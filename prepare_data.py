"""Turn co-registered Sentinel-1 / Sentinel-2 rasters into normalised tiles.

Inputs (same grid, same shape):
  --s1  raster with bands [VV, VH]          (GeoTIFF or .npy of shape (2,H,W))
  --s2  raster with bands [R, G, B] (B4,B3,B2)  (GeoTIFF or .npy of shape (3,H,W))

Outputs <out>/{train,val,test}/{sar,opt}/NNNNN.npy and <out>/meta.json.
SAR tiles hold 3 channels in [-1,1]: [vv_dB, vh_dB, vh-vv ratio_dB]; pick which
ones to feed the model at train time with --channels.
Splits are spatial (horizontal bands of the scene) so tiles never overlap
across splits.  Training is still unpaired (see dataset.py).
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter

VV_RANGE = (-25.0, 0.0)
VH_RANGE = (-30.0, -5.0)
RATIO_RANGE = (-20.0, 0.0)
OPT_MAX = 0.3  # reflectance clipped to [0, 0.3] then mapped to [-1, 1]


def read_raster(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    try:
        import rasterio
    except ImportError:
        raise SystemExit("Reading GeoTIFFs needs rasterio (pip install rasterio), "
                         "or pass .npy arrays instead.")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def lee_filter(img, size):
    """Classic Lee speckle filter on linear-power data."""
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img * img, size)
    var = np.maximum(sq_mean - mean * mean, 0)
    overall = np.var(img)
    w = var / (var + overall + 1e-12)
    return mean + w * (img - mean)


def scale(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1) * 2 - 1


def preprocess_s1(s1, units, lee):
    lin = [(10 ** (b / 10) if units == "db" else b) for b in s1[:2]]
    valid = np.ones(lin[0].shape, bool)
    for b in lin:
        valid &= np.isfinite(b) & (b > 0)
    out_db = []
    for b in lin:
        fill = np.median(b[valid]) if valid.any() else 1.0
        b = np.where(valid, b, fill)
        if lee and lee > 1:
            b = lee_filter(b, lee)
        out_db.append(10 * np.log10(np.maximum(b, 1e-6)))
    vv, vh = out_db
    chans = np.stack([scale(vv, *VV_RANGE), scale(vh, *VH_RANGE),
                      scale(vh - vv, *RATIO_RANGE)]).astype(np.float32)
    chans[:, ~valid] = 0.0
    return chans, valid


def preprocess_s2(s2, divisor, mode):
    rgb = s2[:3] / divisor
    valid = np.all(np.isfinite(rgb), axis=0)
    rgb = np.nan_to_num(rgb)
    if mode == "gray":  # lunar optical is effectively grayscale albedo
        ch = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])[None]
    else:
        ch = rgb
    ch = (np.clip(ch / OPT_MAX, 0, 1) * 2 - 1).astype(np.float32)
    ch[:, ~valid] = 0.0
    return ch, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--s1-units", choices=["db", "linear"], default="db")
    ap.add_argument("--lee-window", type=int, default=5, help="0 disables speckle filter")
    ap.add_argument("--s2-scale", type=float, default=10000.0)
    ap.add_argument("--optical-mode", choices=["gray", "rgb"], default="gray")
    ap.add_argument("--min-valid", type=float, default=0.95)
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    a = ap.parse_args()

    s1, s2 = read_raster(a.s1), read_raster(a.s2)
    if s1.shape[0] < 2 or s2.shape[0] < 3:
        raise SystemExit("need S1 with >=2 bands (VV,VH) and S2 with >=3 bands (R,G,B)")
    if s1.shape[1:] != s2.shape[1:]:
        raise SystemExit(f"S1 {s1.shape[1:]} and S2 {s2.shape[1:]} must share one grid; "
                         "export both from the same region/CRS/scale.")
    sar, v1 = preprocess_s1(s1, a.s1_units, a.lee_window)
    opt, v2 = preprocess_s2(s2, a.s2_scale, a.optical_mode)
    valid = v1 & v2
    _, H, W = sar.shape

    t = a.tile
    # Split by tile-row so every split gets whole rows even on small scenes.
    ys = list(range(0, H - t + 1, a.stride))
    n = len(ys)
    if n < 3:
        raise SystemExit(f"scene too small: only {n} tile rows; need >= 3 for train/val/test")
    f0, f1 = a.split[0] / sum(a.split), a.split[1] / sum(a.split)
    min_va = -(-t // a.stride)  # val band must be at least one tile tall
    n_va = max(min_va, int(round(n * f1)))
    n_tr = max(1, min(int(round(n * f0)), n - n_va - 1))
    if n_tr + n_va >= n:
        raise SystemExit("scene too small for this tile/stride; use a larger scene or smaller tile")
    bounds = (ys[n_tr], ys[n_tr + n_va])  # pixel row where val / test begin
    counts = {"train": 0, "val": 0, "test": 0}
    for sp in counts:
        for sub in ("sar", "opt"):
            os.makedirs(os.path.join(a.out, sp, sub), exist_ok=True)

    skipped = 0
    for y in ys:
        if y + t <= bounds[0]:
            sp = "train"
        elif y >= bounds[0] and y + t <= bounds[1]:
            sp = "val"
        elif y >= bounds[1]:
            sp = "test"
        else:
            continue  # tile straddles a split boundary: drop to avoid leakage
        for x in range(0, W - t + 1, a.stride):
            if valid[y:y + t, x:x + t].mean() < a.min_valid:
                skipped += 1
                continue
            n = counts[sp]
            np.save(os.path.join(a.out, sp, "sar", f"{n:05d}.npy"), sar[:, y:y + t, x:x + t])
            np.save(os.path.join(a.out, sp, "opt", f"{n:05d}.npy"), opt[:, y:y + t, x:x + t])
            counts[sp] += 1

    meta = {"tile": t, "stride": a.stride, "optical_mode": a.optical_mode,
            "sar_channels": ["vv", "vh", "ratio"], "ranges_db": {
                "vv": VV_RANGE, "vh": VH_RANGE, "ratio": RATIO_RANGE},
            "optical_clip_reflectance": OPT_MAX, "lee_window": a.lee_window,
            "counts": counts, "skipped_low_valid": skipped}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
