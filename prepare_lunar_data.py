"""Prepare lunar SAR and optical datasets into normalized tiles for CycleGAN.

Supports two input workflows:
  1. Full-Scene Rasters:
     Cuts large SAR GeoTIFF/npy rasters and LROC/ShadowCam optical rasters into tiles
     with optional PSR mask filtering (via --sar or --sar-ch1/--sar-ch2 and --opt).
  2. Pre-extracted Patch Stacks:
     Unpacks pre-cut 4D stacks (like dataset_moon with train/val/test patches.npy),
     normalizes all radar channels into [-1, 1], and pairs with optical or surrogate
     targets (via --patches-dir).

Outputs:
  <out>/{train,val,test}/{sar,opt}/NNNNN.npy and <out>/meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

DEFAULT_DFSAR_CHANNELS = [
    "east_cpr", "east_srd", "east_trt",
    "west_cpr", "west_srd", "west_trt"
]


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


def make_surrogate_optical(sar_patch):
    """Generate a surrogate lunar grayscale optical patch (1, H, W) in [-1, 1]
    from SAR surface roughness & CPR features when optical matching is pending."""
    cpr = sar_patch[0]
    srd = sar_patch[1] if sar_patch.shape[0] > 1 else sar_patch[0]
    albedo = 0.12 + 0.03 * gaussian_filter(srd, 15)
    rim_boulders = np.clip(cpr * 0.2, 0.0, 0.2)
    opt = np.clip(albedo + rim_boulders, 0.0, 0.35)
    norm = np.clip(opt / 0.3, 0.0, 1.0) * 2.0 - 1.0
    return norm[None].astype(np.float32)


def _clear_npy(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                try:
                    os.remove(os.path.join(folder, f))
                except OSError:
                    pass


def process_psr_dataset(a):
    """Workflow 3: Prepare dataset_PSR (co-registered West SAR/Optical + East Optical).

    Expected layout under --psr-dir:
      WEST_PATCHES/patches.npy  (N_w, 4, H, W): [opt, west_cpr, west_srd, west_trt]
      EAST_PATCHES/patches.npy  (N_e, 4, H, W): [opt, east_cpr, east_srd, east_trt] (optional; all-NaN SAR dropped)
    Uses spatially disjoint row splits on overlapping 2D tile grids so train and test share 0 pixels.
    """
    west_p = os.path.join(a.psr_dir, "WEST_PATCHES", "patches.npy")
    east_p = os.path.join(a.psr_dir, "EAST_PATCHES", "patches.npy")
    if not os.path.exists(west_p):
        sys.exit(f"Missing PSR West patches file: {west_p}")

    west = np.load(west_p).astype(np.float32)  # (N_w, 4, H, W)
    east = np.load(east_p).astype(np.float32) if os.path.exists(east_p) else None

    ch_names = ["west_cpr", "west_srd", "west_trt"]
    n_w = west.shape[0]

    # Spatially disjoint split on the 49-patch 50%-overlap PSR grid (rows 0..3 train, row 4 val buffer, rows 5..6 test)
    if n_w == 49:
        tr_idx = list(range(0, 26))
        va_idx = list(range(26, 34))
        te_idx = list(range(34, 49))
    else:
        n_tr = int(round(n_w * a.split[0]))
        n_va = int(round(n_w * a.split[1]))
        tr_idx = list(range(0, n_tr))
        va_idx = list(range(n_tr, min(n_w, n_tr + n_va)))
        te_idx = list(range(min(n_w, n_tr + n_va), n_w))

    # Extract West SAR [1:4] and Optical [0:1]
    west_sar = west[:, 1:4]
    west_opt = west[:, 0:1]

    # Compute normalization bounds strictly from train split (NaN-safe)
    sar_ranges = {}
    for c, name in enumerate(ch_names):
        tr_ch = west_sar[tr_idx, c]
        valid = np.isfinite(tr_ch)
        lo = float(np.percentile(tr_ch[valid], 1.0)) if valid.any() else 0.0
        hi = float(np.percentile(tr_ch[valid], 99.0)) if valid.any() else 1.0
        if hi <= lo:
            hi = lo + 1e-4
        sar_ranges[name] = (lo, hi)
        print(f"  PSR {name:10s}: [{lo:.4f}, {hi:.4f}] -> mapped to [-1, 1]")

    # Include valid East optical patches in training optical pool (Domain B)
    east_opt = None
    if east is not None:
        east_opt_raw = east[:, 0:1]
        if np.isfinite(east_opt_raw).any():
            east_opt = east_opt_raw

    tr_opt_pool = [west_opt[tr_idx]] + ([east_opt] if east_opt is not None else [])
    tr_opt_cat = np.concatenate(tr_opt_pool, axis=0)
    opt_valid = np.isfinite(tr_opt_cat)
    opt_lo = float(np.percentile(tr_opt_cat[opt_valid], 1.0))
    opt_hi = float(np.percentile(tr_opt_cat[opt_valid], 99.0))
    if opt_hi <= opt_lo:
        opt_hi = opt_lo + 1e-4
    opt_range = (opt_lo, opt_hi)
    print(f"  PSR optical  : [{opt_lo:.4f}, {opt_hi:.4f}] -> mapped to [-1, 1]")

    def norm_sar_batch(batch):
        out = np.zeros_like(batch, dtype=np.float32)
        for c, name in enumerate(ch_names):
            lo, hi = sar_ranges[name]
            ch = batch[:, c]
            valid = np.isfinite(ch)
            norm = np.clip((np.where(valid, ch, lo) - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
            norm[~valid] = 0.0  # neutral mid-range fill for NoData radar edges
            out[:, c] = norm
        return out

    def norm_opt_batch(batch):
        valid = np.isfinite(batch)
        norm = np.clip((np.where(valid, batch, opt_lo) - opt_lo) / (opt_hi - opt_lo), 0.0, 1.0) * 2.0 - 1.0
        norm[~valid] = -1.0
        return norm.astype(np.float32)

    splits_map = {
        "train": (norm_sar_batch(west_sar[tr_idx]), norm_opt_batch(tr_opt_cat)),
        "val": (norm_sar_batch(west_sar[va_idx]), norm_opt_batch(west_opt[va_idx])),
        "test": (norm_sar_batch(west_sar[te_idx]), norm_opt_batch(west_opt[te_idx])),
    }

    counts_sar, counts_opt = {}, {}
    for sp, (s_arr, o_arr) in splits_map.items():
        sar_dir = os.path.join(a.out, sp, "sar")
        opt_dir = os.path.join(a.out, sp, "opt")
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        _clear_npy(sar_dir)
        _clear_npy(opt_dir)

        for i in range(s_arr.shape[0]):
            np.save(os.path.join(sar_dir, f"{i:05d}.npy"), s_arr[i])
        for j in range(o_arr.shape[0]):
            np.save(os.path.join(opt_dir, f"{j:05d}.npy"), o_arr[j])
        counts_sar[sp] = int(s_arr.shape[0])
        counts_opt[sp] = int(o_arr.shape[0])

    meta = {
        "source": "PSR co-registered patches (spatially disjoint row split)",
        "psr_dir": a.psr_dir,
        "sar_channels": ch_names,
        "sar_ranges": sar_ranges,
        "optical_range": opt_range,
        "counts_sar": counts_sar,
        "counts_opt": counts_opt,
        "has_real_optical": True,
        "split_indices": {"train_west": tr_idx, "val_west": va_idx, "test_west": te_idx},
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPrepared PSR dataset in '{a.out}':")
    print(json.dumps({k: v for k, v in meta.items() if k != "split_indices"}, indent=2))


def process_patches_stack(a):
    """Workflow 2: Unpack and normalize 4D pre-extracted patches (e.g. dataset_moon)."""
    train_path = os.path.join(a.patches_dir, "train", "patches.npy")
    val_path = os.path.join(a.patches_dir, "val", "patches.npy")
    test_path = os.path.join(a.patches_dir, "test", "patches.npy")

    for p in (train_path, val_path, test_path):
        if not os.path.exists(p):
            sys.exit(f"Missing patches file: {p}")

    print(f"Loading patch stacks from '{a.patches_dir}'...")
    train_raw = np.load(train_path)
    val_raw = np.load(val_path)
    test_raw = np.load(test_path)

    n_ch = train_raw.shape[1]
    if a.sar_names:
        ch_names = [n.strip().lower() for n in a.sar_names.split(",") if n.strip()]
    elif n_ch == 6:
        ch_names = DEFAULT_DFSAR_CHANNELS
    else:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    print(f"Loaded: train={train_raw.shape}, val={val_raw.shape}, test={test_raw.shape} ({n_ch} channels)")

    # Compute normalization ranges from training split
    ranges = {}
    for c in range(n_ch):
        name = ch_names[c] if c < len(ch_names) else f"ch{c}"
        data = train_raw[:, c, :, :]
        lo = float(np.percentile(data, 1.0))
        hi = float(np.percentile(data, 99.0))
        if hi <= lo:
            hi = lo + 1e-4
        ranges[name] = (lo, hi)
        print(f"  {name:10s}: [{lo:.4f}, {hi:.4f}] -> mapped to [-1, 1]")

    # Check if real optical patches exist under <patches_dir>/optical
    opt_base = os.path.join(a.patches_dir, "optical")
    has_real_optical = False
    opt_splits = {}
    opt_range = None

    if os.path.isdir(opt_base):
        train_opt_p = os.path.join(opt_base, "train", "patches.npy")
        val_opt_p = os.path.join(opt_base, "val", "patches.npy")
        test_opt_p = os.path.join(opt_base, "test", "patches.npy")
        if os.path.exists(train_opt_p) and os.path.exists(val_opt_p) and os.path.exists(test_opt_p):
            has_real_optical = True
            print(f"Found real optical dataset in '{opt_base}'!")
            tr_opt = np.load(train_opt_p).astype(np.float32)
            va_opt = np.load(val_opt_p).astype(np.float32)
            te_opt = np.load(test_opt_p).astype(np.float32)

            if tr_opt.ndim == 3:
                tr_opt = tr_opt[:, None]
                va_opt = va_opt[:, None]
                te_opt = te_opt[:, None]

            opt_lo = float(np.percentile(tr_opt, 1.0))
            opt_hi = float(np.percentile(tr_opt, 99.0))
            if opt_hi <= opt_lo:
                opt_hi = opt_lo + 1.0
            opt_range = (opt_lo, opt_hi)
            print(f"  Optical normalization range: [{opt_lo:.4f}, {opt_hi:.4f}] -> mapped to [-1, 1]")
            print(f"  Loaded optical counts: train={len(tr_opt)}, val={len(va_opt)}, test={len(te_opt)}")

            opt_splits["train"] = np.clip((tr_opt - opt_lo) / (opt_hi - opt_lo), 0.0, 1.0) * 2.0 - 1.0
            opt_splits["val"] = np.clip((va_opt - opt_lo) / (opt_hi - opt_lo), 0.0, 1.0) * 2.0 - 1.0
            opt_splits["test"] = np.clip((te_opt - opt_lo) / (opt_hi - opt_lo), 0.0, 1.0) * 2.0 - 1.0

    # Normalize splits
    splits = {"train": train_raw, "val": val_raw, "test": test_raw}
    counts_sar = {"train": 0, "val": 0, "test": 0}
    counts_opt = {"train": 0, "val": 0, "test": 0}

    for sp, raw_arr in splits.items():
        sar_dir = os.path.join(a.out, sp, "sar")
        opt_dir = os.path.join(a.out, sp, "opt")
        os.makedirs(sar_dir, exist_ok=True)
        if has_real_optical or a.surrogate_opt:
            os.makedirs(opt_dir, exist_ok=True)

        # Clear any existing .npy files from older runs in this split
        for f in os.listdir(sar_dir):
            if f.endswith(".npy"):
                try: os.remove(os.path.join(sar_dir, f))
                except OSError: pass
        if os.path.exists(opt_dir):
            for f in os.listdir(opt_dir):
                if f.endswith(".npy"):
                    try: os.remove(os.path.join(opt_dir, f))
                    except OSError: pass

        norm_arr = np.empty_like(raw_arr, dtype=np.float32)
        for c in range(n_ch):
            name = ch_names[c] if c < len(ch_names) else f"ch{c}"
            lo, hi = ranges[name]
            norm_arr[:, c, :, :] = np.clip((raw_arr[:, c, :, :] - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0

        for i in range(norm_arr.shape[0]):
            np.save(os.path.join(sar_dir, f"{i:05d}.npy"), norm_arr[i])
            counts_sar[sp] += 1

        if has_real_optical:
            sp_opt = opt_splits[sp]
            for j in range(sp_opt.shape[0]):
                np.save(os.path.join(opt_dir, f"{j:05d}.npy"), sp_opt[j])
                counts_opt[sp] += 1
        elif a.surrogate_opt:
            for i in range(norm_arr.shape[0]):
                opt_tile = make_surrogate_optical(norm_arr[i])
                np.save(os.path.join(opt_dir, f"{i:05d}.npy"), opt_tile)
                counts_opt[sp] += 1

    meta = {
        "source": "pre-extracted patch stacks",
        "patches_dir": a.patches_dir,
        "sar_channels": ch_names[:n_ch],
        "sar_ranges": ranges,
        "optical_range": opt_range,
        "counts_sar": counts_sar,
        "counts_opt": counts_opt,
        "has_real_optical": has_real_optical,
        "has_surrogate_optical": (not has_real_optical) and a.surrogate_opt
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPrepared {sum(counts_sar.values())} SAR tiles and {sum(counts_opt.values())} Optical tiles in '{a.out}':")
    print(json.dumps(meta, indent=2))


def process_full_rasters(a):
    """Workflow 1: Slice large full-scene GeoTIFF/npy rasters into tiles with PSR filtering."""
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
        sys.exit("Must supply either --patches-dir, OR --sar, OR --sar-ch1 (and optional --sar-ch2).")

    # Load Optical
    if not a.opt:
        sys.exit("Full-scene raster mode requires --opt (path to optical raster).")

    opt_arr = read_raster(a.opt)
    if opt_arr.ndim == 2:
        opt_arr = opt_arr[None]
    elif opt_arr.shape[0] > 1:
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
        print(f"Loaded PSR mask: {psr_mask_arr.sum()} pixels ({psr_mask_arr.mean() * 100:.2f}% of scene) in PSR.")

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

    # Collect candidate tiles
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
            f"Skipped {skipped_non_psr} non-PSR tiles and {skipped_low_valid} nodata tiles."
        )

    print(f"Found {len(candidate_tiles)} valid tiles (filtered out {skipped_non_psr} non-PSR tiles, {skipped_low_valid} nodata tiles).")

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
        "source": "full-scene rasters",
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


def main():
    ap = argparse.ArgumentParser(description="Prepare lunar SAR and optical tiles.")
    # Mode 1: Pre-extracted patch stacks (e.g. dataset_moon or dataset_PSR)
    ap.add_argument("--patches-dir", help="Path to pre-extracted patches folder (like dataset_moon with train/val/test patches.npy).")
    ap.add_argument("--psr-dir", help="Path to PSR dataset folder (like dataset_PSR with WEST_PATCHES/EAST_PATCHES).")
    ap.add_argument("--surrogate-opt", action="store_true", help="Generate surrogate optical targets for testing when ShadowCam matching is pending.")

    # Mode 2: Full-scene rasters
    ap.add_argument("--sar", help="Single multi-band SAR file (GeoTIFF or .npy).")
    ap.add_argument("--sar-ch1", help="SAR primary channel (e.g. S1 or Same-Sense circular).")
    ap.add_argument("--sar-ch2", help="SAR secondary channel (e.g. CPR or Opposite-Sense circular).")
    ap.add_argument("--sar-names", help="Comma-separated channel names (e.g. s1,cpr or east_cpr,east_srd,...).")
    ap.add_argument("--lee-window", type=int, default=3, help="Lee speckle filter window size (0 to disable).")
    ap.add_argument("--opt", help="LROC NAC / ShadowCam grayscale optical raster.")

    # PSR filtering options
    ap.add_argument("--psr-mask", help="Path to PSR mask (1=PSR, 0=sunlit) or LOLA illumination GeoTIFF/.npy.")
    ap.add_argument("--min-psr-fraction", type=float, default=0.5,
                    help="Minimum fraction of tile area that must be inside PSR (default: 0.5 = 50%%).")
    ap.add_argument("--psr-is-illumination", action="store_true",
                    help="Set if mask is a LOLA illumination percentage (0-100%%); pixels with illumination <= --psr-thresh are treated as PSR.")
    ap.add_argument("--psr-thresh", type=float, default=0.0,
                    help="Illumination threshold for PSR when --psr-is-illumination is set (default: 0.0%%).")

    # Tiling & general options
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--tile", type=int, default=256, help="Tile size in pixels.")
    ap.add_argument("--stride", type=int, default=256, help="Tile stride in pixels.")
    ap.add_argument("--min-valid", type=float, default=0.90, help="Min fraction of valid pixels per tile.")
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"), help="Train/val/test fractions.")
    a = ap.parse_args()

    if a.psr_dir:
        process_psr_dataset(a)
    elif a.patches_dir:
        process_patches_stack(a)
    else:
        process_full_rasters(a)


if __name__ == "__main__":
    main()
