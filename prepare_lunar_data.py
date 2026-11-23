"""Prepare lunar SAR and optical rasters into normalized tiles for CycleGAN.

Designed for Lunar instruments:
  - SAR: Chandrayaan-2 DFSAR (Hybrid-pol / Stokes) or LRO Mini-RF (S1, CPR / SC, OC)
  - Optical: LRO LROC NAC (grayscale radiance/reflectance) or ShadowCam

Inputs:
  SAR: either a multi-band raster via --sar, OR separate single-band rasters
       via --sar-ch1 and --sar-ch2 (e.g. S1 and CPR).
  Optical: single-band (or multi-band) grayscale raster via --opt.

Outputs:
  <out>/{train,val,test}/{sar,opt}/NNNNN.npy and <out>/meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter


def read_raster(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        return arr
    try:
        import rasterio
    except ImportError:
        raise SystemExit(
            "Reading GeoTIFFs requires rasterio (`pip install rasterio`), "
            "or convert files to .npy arrays."
        )
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def lee_filter(img, size):
    """Lee speckle filter on linear-scale power raster."""
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img * img, size)
    var = np.maximum(sq_mean - mean * mean, 0)
    overall = np.var(img)
    w = var / (var + overall + 1e-12)
    return mean + w * (img - mean)


def normalize_to_minus1_1(arr, valid_mask, p_low=1.0, p_high=99.0, manual_range=None):
    """Normalize array values to [-1, 1] using robust percentiles or manual range."""
    valid_vals = arr[valid_mask]
    if valid_vals.size == 0:
        return np.zeros_like(arr), (0.0, 1.0)
    
    if manual_range is not None:
        lo, hi = manual_range
    else:
        lo = float(np.percentile(valid_vals, p_low))
        hi = float(np.percentile(valid_vals, p_high))
        if hi <= lo:
            hi = lo + 1e-5

    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
    norm[~valid_mask] = 0.0
    return norm.astype(np.float32), (lo, hi)


def main():
    ap = argparse.ArgumentParser(description="Prepare lunar SAR and optical tiles.")
    # SAR options
    ap.add_argument("--sar", help="Single multi-band SAR file (GeoTIFF or .npy).")
    ap.add_argument("--sar-ch1", help="SAR primary channel (e.g. S1 or Same-Sense circular).")
    ap.add_argument("--sar-ch2", help="SAR secondary channel (e.g. CPR or Opposite-Sense circular).")
    ap.add_argument("--sar-names", default="s1,cpr", help="Comma-separated channel names (e.g. s1,cpr or sc,oc).")
    ap.add_argument("--lee-window", type=int, default=3, help="Lee speckle filter window size (0 to disable).")

    # Optical options
    ap.add_argument("--opt", required=True, help="LROC NAC / ShadowCam grayscale optical raster.")

    # PSR (Permanently Shadowed Region) filtering options
    ap.add_argument("--psr-mask", help="Path to PSR mask (1=PSR, 0=sunlit) or LOLA illumination GeoTIFF/.npy.")
    ap.add_argument("--min-psr-fraction", type=float, default=0.5,
                    help="Minimum fraction of tile area that must be inside PSR (default: 0.5 = 50%%).")
    ap.add_argument("--psr-is-illumination", action="store_true",
                    help="Set if mask is a LOLA illumination percentage (0-100%%); pixels with illumination <= --psr-thresh are treated as PSR.")
    ap.add_argument("--psr-thresh", type=float, default=0.0,
                    help="Illumination threshold for PSR when --psr-is-illumination is set (default: 0.0%%).")

    # General & tiling options
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--tile", type=int, default=256, help="Tile size in pixels.")
    ap.add_argument("--stride", type=int, default=256, help="Tile stride in pixels.")
    ap.add_argument("--min-valid", type=float, default=0.90, help="Min fraction of valid pixels per tile.")
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"), help="Train/val/test fractions.")
    a = ap.parse_args()

    # Load SAR
    if a.sar:
        sar_arr = read_raster(a.sar)
    elif a.sar_ch1:
        ch1 = read_raster(a.sar_ch1)
        if a.sar_ch2:
            ch2 = read_raster(a.sar_ch2)
            if ch1.shape[1:] != ch2.shape[1:]:
                sys.exit(f"SAR ch1 {ch1.shape} and ch2 {ch2.shape} have mismatched spatial dimensions.")
            sar_arr = np.concatenate([ch1[:1], ch2[:1]], axis=0)
        else:
            sar_arr = ch1[:1]
    else:
        sys.exit("Must supply either --sar (multi-band raster) or --sar-ch1 (and optional --sar-ch2).")

    # Load Optical
    opt_arr = read_raster(a.opt)
    if opt_arr.ndim == 2:
        opt_arr = opt_arr[None]
    elif opt_arr.shape[0] > 1:
        # If multi-band (RGB), convert to lunar luminance
        opt_arr = (0.2126 * opt_arr[0] + 0.7152 * opt_arr[1] + 0.0722 * opt_arr[2])[None]

    # Verify spatial alignment
    if sar_arr.shape[1:] != opt_arr.shape[1:]:
        sys.exit(
            f"Dimension mismatch: SAR {sar_arr.shape[1:]} vs Optical {opt_arr.shape[1:]}.\n"
            "Both rasters must be co-registered on the exact same pixel grid (same CRS, resolution, bounds)."
        )

    _, H, W = sar_arr.shape
    t = a.tile
    channel_names = [n.strip().lower() for n in a.sar_names.split(",") if n.strip()]

    # Load PSR mask if provided
    psr_mask_arr = None
    if a.psr_mask:
        p_raw = read_raster(a.psr_mask)
        if p_raw.ndim == 3:
            p_raw = p_raw[0]
        if p_raw.shape != (H, W):
            sys.exit(f"Dimension mismatch: PSR mask {p_raw.shape} does not match scene {(H, W)}.")
        if a.psr_is_illumination:
            psr_mask_arr = (p_raw <= a.psr_thresh) & np.isfinite(p_raw)
        else:
            psr_mask_arr = (p_raw > 0.5) & np.isfinite(p_raw)
        print(f"Loaded PSR mask: {psr_mask_arr.sum()} pixels ({psr_mask_arr.mean() * 100:.2f}%% of scene) in PSR.")

    # Validity masks
    sar_valid = np.all(np.isfinite(sar_arr), axis=0)
    opt_valid = np.all(np.isfinite(opt_arr), axis=0)
    overall_valid = sar_valid & opt_valid

    # Preprocess SAR channels
    sar_proc = []
    sar_ranges = {}
    for idx in range(sar_arr.shape[0]):
        ch = sar_arr[idx]
        ch_name = channel_names[idx] if idx < len(channel_names) else f"ch{idx}"
        
        # Lee filter for speckle reduction if requested
        if a.lee_window > 1:
            ch_finite = np.where(sar_valid, ch, np.median(ch[sar_valid]) if sar_valid.any() else 0.0)
            ch = lee_filter(ch_finite, a.lee_window)

        ch_norm, r_range = normalize_to_minus1_1(ch, sar_valid, p_low=1.0, p_high=99.0)
        sar_proc.append(ch_norm)
        sar_ranges[ch_name] = r_range

    sar_tiles_stack = np.stack(sar_proc, axis=0)

    # Preprocess Optical channel
    opt_norm, opt_range = normalize_to_minus1_1(opt_arr[0], opt_valid, p_low=1.0, p_high=99.0)
    opt_tiles_stack = opt_norm[None]

    # Collect candidate tiles matching validity and PSR mask criteria
    skipped_low_valid = 0
    skipped_non_psr = 0
    candidate_tiles = []

    for y in range(0, H - t + 1, a.stride):
        for x in range(0, W - t + 1, a.stride):
            if overall_valid[y:y + t, x:x + t].mean() < a.min_valid:
                skipped_low_valid += 1
                continue
            if psr_mask_arr is not None:
                psr_frac = float(psr_mask_arr[y:y + t, x:x + t].mean())
                if psr_frac < a.min_psr_fraction:
                    skipped_non_psr += 1
                    continue
            candidate_tiles.append((y, x))

    if not candidate_tiles:
        sys.exit(
            f"No tiles found matching criteria!\n"
            f"Skipped {skipped_non_psr} tiles because PSR coverage was below {a.min_psr_fraction * 100:.1f}%%.\n"
            f"Try lowering --min-psr-fraction or checking that your PSR mask aligns with the SAR/optical data."
        )

    print(f"Found {len(candidate_tiles)} valid tiles (filtered out {skipped_non_psr} non-PSR tiles, {skipped_low_valid} nodata tiles).")

    # Spatial partitioning by coordinate ordering so tiles in splits are distinct spatial blocks
    candidate_tiles.sort(key=lambda pos: (pos[0], pos[1]))
    n_total = len(candidate_tiles)
    f0, f1 = a.split[0] / sum(a.split), a.split[1] / sum(a.split)
    n_tr = max(1, int(round(n_total * f0)))
    n_va = max(1 if n_total >= 3 else 0, int(round(n_total * f1)))
    if n_tr + n_va >= n_total and n_total >= 3:
        n_tr = n_total - n_va - 1

    partition = {
        "train": candidate_tiles[:n_tr],
        "val": candidate_tiles[n_tr:n_tr + n_va] if n_total >= 2 else candidate_tiles[:n_tr],
        "test": candidate_tiles[n_tr + n_va:] if n_total >= 3 else candidate_tiles[:n_tr]
    }

    counts = {"train": 0, "val": 0, "test": 0}
    for sp in counts:
        for sub in ("sar", "opt"):
            os.makedirs(os.path.join(a.out, sp, sub), exist_ok=True)

    for sp, tiles in partition.items():
        for idx, (y, x) in enumerate(tiles):
            np.save(os.path.join(a.out, sp, "sar", f"{idx:05d}.npy"),
                    sar_tiles_stack[:, y:y + t, x:x + t])
            np.save(os.path.join(a.out, sp, "opt", f"{idx:05d}.npy"),
                    opt_tiles_stack[:, y:y + t, x:x + t])
            counts[sp] += 1

    meta = {
        "tile": t,
        "stride": a.stride,
        "optical_mode": "gray",
        "sar_channels": channel_names[:sar_arr.shape[0]],
        "sar_ranges": sar_ranges,
        "optical_range": opt_range,
        "lee_window": a.lee_window,
        "psr_filtered": a.psr_mask is not None,
        "min_psr_fraction": a.min_psr_fraction if a.psr_mask else None,
        "counts": counts,
        "skipped_non_psr": skipped_non_psr,
        "skipped_low_valid": skipped_low_valid
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Lunar dataset preparation complete:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
