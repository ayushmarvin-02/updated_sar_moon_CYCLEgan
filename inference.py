"""Dedicated Lunar PSR SAR-to-Optical Inference Pipeline.

Synthesizes high-resolution optical imagery from polarimetric Synthetic
Aperture Radar (SAR) data (Chandrayaan-2 DFSAR / LRO Mini-RF) in lunar
Permanently Shadowed Regions (PSRs) using trained CycleGAN generators.

Usage examples:
  1. Infer on a single test patch from dataset_moon:
     python inference.py --input dataset_moon/test/patches.npy --index 0 --out runs/inference

  2. Infer on all patches in a test stack:
     python inference.py --input dataset_moon/test/patches.npy --all --out runs/inference

  3. Infer on an individual .npy tile:
     python inference.py --input data/moon_dfsar/test/sar/00000.npy --out runs/inference

  4. Infer on a large full-scene GeoTIFF raster:
     python inference.py --input path/to/sar_scene.tif --out runs/inference --save-tif

  5. Python API:
     from inference import LunarOpticalSynthesizer
     synth = LunarOpticalSynthesizer("runs/real_optical_100/last.pt")
     optical_img = synth.predict(sar_patch)
"""
import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

from dataset import CHANNEL_INDEX, parse_channels
from models import ResnetGenerator
from utils import to_img

# Optional rasterio import for GeoTIFF support
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Optional scipy filter for Lee speckle filter
try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


DEFAULT_CKPT_CANDIDATES = [
    "runs/psr_cyclegan_100/last.pt",
    "runs/real_optical_100/last.pt",
    "runs/real_optical_50/last.pt",
    "runs/epochs_50/last.pt",
]

DEFAULT_META_CANDIDATES = [
    "data/moon_psr/meta.json",
    "data/moon_dfsar/meta.json",
    "data/dfsar/meta.json",
    "data/lunar_synthetic/meta.json",
]


def find_default_checkpoint():
    for c in DEFAULT_CKPT_CANDIDATES:
        if os.path.exists(c):
            return c
    return DEFAULT_CKPT_CANDIDATES[0]


def find_default_meta(cfg_data=None):
    if cfg_data and os.path.exists(os.path.join(cfg_data, "meta.json")):
        return os.path.join(cfg_data, "meta.json")
    for m in DEFAULT_META_CANDIDATES:
        if os.path.exists(m):
            return m
    return None


def lee_filter(img, size=3):
    """Speckle reduction on linear SAR intensity."""
    if not HAS_SCIPY or size <= 1:
        return img
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img * img, size)
    var = np.maximum(sq_mean - mean * mean, 0)
    overall = np.var(img)
    w = var / (var + overall + 1e-12)
    return mean + w * (img - mean)


def create_side_by_side_visual(sar_tile, optical_tile, title=None):
    """Create an annotated side-by-side comparison image:
       [ SAR Input (Radar) | Synthesized Optical (PSR) ]
    """
    # sar_tile: (C, H, W) in [-1, 1] or uint8
    if isinstance(sar_tile, torch.Tensor):
        sar_tile = sar_tile.detach().cpu().numpy()
    if isinstance(optical_tile, torch.Tensor):
        optical_tile = optical_tile.detach().cpu().numpy()

    # Format SAR visual (first channel or false color if >=2 channels)
    if sar_tile.dtype != np.uint8:
        sar_norm = np.clip((sar_tile + 1.0) / 2.0, 0.0, 1.0)
        if sar_norm.shape[0] >= 2:
            # False-color composite: R=ch0 (CPR), G=ch1 (SRD), B=ch0*0.5
            r = sar_norm[0]
            g = sar_norm[1]
            b = np.clip(sar_norm[0] * 0.6 + sar_norm[1] * 0.4, 0.0, 1.0)
            sar_img = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        else:
            ch0 = (sar_norm[0] * 255).astype(np.uint8)
            sar_img = np.stack([ch0, ch0, ch0], axis=-1)
    else:
        sar_img = sar_tile

    # Format Optical visual (Grayscale H, W -> 3-channel H, W, 3)
    if optical_tile.dtype != np.uint8:
        opt_norm = np.clip((optical_tile + 1.0) / 2.0, 0.0, 1.0)
        if opt_norm.ndim == 3:
            opt_norm = opt_norm[0]
        opt_uint8 = (opt_norm * 255).astype(np.uint8)
    else:
        opt_uint8 = optical_tile
    if opt_uint8.ndim == 2:
        opt_img = np.stack([opt_uint8, opt_uint8, opt_uint8], axis=-1)
    else:
        opt_img = opt_uint8

    h, w, _ = sar_img.shape
    banner_h = 32
    sep_w = 4
    canvas_w = w * 2 + sep_w
    canvas_h = h + banner_h

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = 25  # Dark banner background
    canvas[banner_h:, :w] = sar_img
    canvas[banner_h:, w:w + sep_w] = 80  # Separator line
    canvas[banner_h:, w + sep_w:] = opt_img

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    # Header labels
    label_sar = "INPUT SAR RADAR (CPR / SRD)"
    label_opt = "SYNTHESIZED OPTICAL (PSR)"
    draw.text((10, 8), label_sar, fill=(180, 220, 255))
    draw.text((w + sep_w + 10, 8), label_opt, fill=(255, 230, 180))

    if title:
        draw.text((canvas_w // 2 - 40, 8), f"[{title}]", fill=(200, 200, 200))

    return pil_img


class LunarOpticalSynthesizer:
    """Full inference wrapper for SAR-to-Optical lunar cycle synthesis."""

    def __init__(self, ckpt_path=None, meta_path=None, device=None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ckpt_path = ckpt_path or find_default_checkpoint()
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        print(f"Loading Lunar Optical Synthesizer from: {self.ckpt_path}")
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.cfg = self.ckpt.get("args", {})

        self.in_ch = self.cfg.get("in_ch", 4)
        self.out_ch = self.cfg.get("out_ch", 1)
        self.ngf = self.cfg.get("ngf", 64)
        self.n_blocks = self.cfg.get("n_blocks", 9)
        self.channel_spec = self.cfg.get("channels", "east_cpr,east_srd,west_cpr,west_srd")

        # Parse target channel names
        self.target_channel_names = parse_channels(self.channel_spec)
        print(f"  Architecture: ResNet-{self.n_blocks} (ngf={self.ngf})")
        print(f"  SAR Input Channels ({self.in_ch}): {self.target_channel_names}")
        print(f"  Optical Output Channels: {self.out_ch}")
        print(f"  Device: {self.device}")

        # Initialize G_AB (SAR -> Optical)
        self.net_G = ResnetGenerator(self.in_ch, self.out_ch, self.ngf, self.n_blocks)
        self.net_G.to(self.device)
        self.net_G.load_state_dict(self.ckpt["G_AB"])
        self.net_G.eval()

        # Load normalization metadata
        self.meta_path = meta_path or find_default_meta(self.cfg.get("data"))
        self.meta = None
        self.sar_ranges = {}
        if self.meta_path and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path) as f:
                    self.meta = json.load(f)
                self.sar_ranges = self.meta.get("sar_ranges", {})
                print(f"  Loaded calibration metadata from: {self.meta_path}")
            except Exception as e:
                print(f"  Warning: failed to read meta.json ({e})")

        # Warm up generator
        self._warmup()

    def _warmup(self):
        """Execute one dry run to warm up CUDA kernels."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_ch, 256, 256, device=self.device)
            _ = self.net_G(dummy)

    def prepare_sar_patch(self, raw_patch, channel_names=None, denoise=0):
        """Prepares a raw SAR patch (C, H, W) for generator input.
        - Handles NaN NoData pixels safely (neutral 0.0 fill in [-1, 1]).
        - Applies optional Lee speckle filtering.
        - Maps available channels to target model channels.
        - Normalizes values to [-1, 1].
        """
        arr = np.array(raw_patch, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]  # (1, H, W)

        # Apply speckle filter if requested (NaN-safe)
        if denoise > 1 and HAS_SCIPY:
            for c in range(arr.shape[0]):
                valid_c = np.isfinite(arr[c])
                if valid_c.any():
                    med = float(np.median(arr[c][valid_c]))
                    filtered = lee_filter(np.where(valid_c, arr[c], med), size=denoise)
                    arr[c] = np.where(valid_c, filtered, np.nan)

        # Check if already normalized to [-1, 1] using finite values only
        valid_all = np.isfinite(arr)
        if valid_all.any():
            fin_vals = arr[valid_all]
            vmin, vmax = float(fin_vals.min()), float(fin_vals.max())
            is_already_normalized = (vmin >= -1.05 and vmax <= 1.05 and (vmin < -0.05 or vmax > 0.05))
        else:
            is_already_normalized = True

        # Channel mapping
        if channel_names is None:
            if arr.shape[0] == 6:
                channel_names = [
                    "east_cpr", "east_srd", "east_trt",
                    "west_cpr", "west_srd", "west_trt"
                ]
            elif arr.shape[0] == 3:
                # 3-channel PSR/DFSAR look layout: [cpr, srd, trt]
                channel_names = ["west_cpr", "west_srd", "west_trt"]
            elif arr.shape[0] == 2:
                channel_names = ["cpr", "srd"]
            elif arr.shape[0] == len(self.target_channel_names):
                channel_names = self.target_channel_names
            else:
                channel_names = [f"ch{i}" for i in range(arr.shape[0])]

        # Single-look West fallback aliases if a 4-channel dual-look model receives West-only SAR
        alias_fallback = {
            "east_cpr": "west_cpr", "east_srd": "west_srd", "east_trt": "west_trt",
            "west_cpr": "east_cpr", "west_srd": "east_srd", "west_trt": "east_trt",
            "cpr": "west_cpr", "srd": "west_srd", "trt": "west_trt",
        }

        # Extract required channels
        selected_channels = []
        for target_name in self.target_channel_names:
            if target_name in channel_names:
                ch_data = arr[channel_names.index(target_name)]
            elif target_name in alias_fallback and alias_fallback[target_name] in channel_names:
                ch_data = arr[channel_names.index(alias_fallback[target_name])]
            elif target_name in CHANNEL_INDEX and CHANNEL_INDEX[target_name] < arr.shape[0]:
                ch_data = arr[CHANNEL_INDEX[target_name]]
            else:
                idx = min(len(selected_channels), arr.shape[0] - 1)
                ch_data = arr[idx]

            valid_ch = np.isfinite(ch_data)
            if not valid_ch.any():
                ch_norm = np.zeros_like(ch_data, dtype=np.float32)
            elif not is_already_normalized:
                if target_name in self.sar_ranges:
                    lo, hi = self.sar_ranges[target_name]
                elif target_name in alias_fallback and alias_fallback[target_name] in self.sar_ranges:
                    lo, hi = self.sar_ranges[alias_fallback[target_name]]
                else:
                    fin_ch = ch_data[valid_ch]
                    lo = float(np.percentile(fin_ch, 1.0))
                    hi = float(np.percentile(fin_ch, 99.0))
                    if hi <= lo:
                        hi = lo + 1e-4
                ch_norm = np.clip((np.where(valid_ch, ch_data, lo) - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
                ch_norm[~valid_ch] = 0.0
            else:
                ch_norm = np.clip(np.where(valid_ch, ch_data, 0.0), -1.0, 1.0)
                ch_norm[~valid_ch] = 0.0

            selected_channels.append(ch_norm.astype(np.float32))

        normalized_stack = np.stack(selected_channels, axis=0)  # (in_ch, H, W)
        return normalized_stack

    def synthesize_tile(self, sar_normalized):
        """Runs single-tile forward pass (in_ch, H, W) -> (out_ch, H, W) in [-1, 1]."""
        t_in = torch.from_numpy(sar_normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                t_out = self.net_G(t_in)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0
            else:
                t0 = time.perf_counter()
                t_out = self.net_G(t_in)
                latency_ms = (time.perf_counter() - t0) * 1000.0

        out_np = t_out.squeeze(0).cpu().numpy()
        return out_np, latency_ms

    def synthesize_raster(self, sar_arr, tile_size=256, stride=192, denoise=0):
        """Synthesizes large / arbitrary-sized rasters using sliding-window inference
        with smooth cosine-weighted overlap blending to eliminate seam artifacts.
        """
        sar_norm = self.prepare_sar_patch(sar_arr, denoise=denoise)
        _, H, W = sar_norm.shape

        # If exactly tile size, run direct forward pass
        if H == tile_size and W == tile_size:
            out_tile, lat = self.synthesize_tile(sar_norm)
            return out_tile, lat

        # Prepare accumulator arrays for seamless blending
        out_accum = np.zeros((self.out_ch, H, W), dtype=np.float32)
        weight_accum = np.zeros((H, W), dtype=np.float32)

        # 2D Hann / Cosine window for seamless edge blending
        w1d = np.hanning(tile_size)
        w2d = np.outer(w1d, w1d).astype(np.float32)
        w2d = np.maximum(w2d, 1e-4)

        y_steps = list(range(0, max(1, H - tile_size + 1), stride))
        if y_steps[-1] != H - tile_size:
            y_steps.append(H - tile_size)

        x_steps = list(range(0, max(1, W - tile_size + 1), stride))
        if x_steps[-1] != W - tile_size:
            x_steps.append(W - tile_size)

        total_tiles = len(y_steps) * len(x_steps)
        total_time_ms = 0.0

        print(f"Synthesizing large raster ({W}x{H}) across {total_tiles} sliding tiles...")

        for y in y_steps:
            for x in x_steps:
                patch = sar_norm[:, y:y + tile_size, x:x + tile_size]
                out_p, lat = self.synthesize_tile(patch)
                total_time_ms += lat

                for c in range(self.out_ch):
                    out_accum[c, y:y + tile_size, x:x + tile_size] += out_p[c] * w2d
                weight_accum[y:y + tile_size, x:x + tile_size] += w2d

        out_accum /= np.maximum(weight_accum, 1e-5)
        out_accum = np.clip(out_accum, -1.0, 1.0)
        return out_accum, total_time_ms

    def predict(self, input_data, index=None, denoise=0, psr_skip_opt=False):
        """Unified prediction interface accepting:
        - file path (.npy, .tif, .tiff)
        - numpy array (2D, 3D, or 4D stack)
        """
        # Load from file if path is given
        geo_profile = None
        input_path_str = input_data if isinstance(input_data, str) else ""
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Input file not found: {input_data}")
            if input_data.endswith(".npy"):
                arr = np.load(input_data)
            elif input_data.lower().endswith((".tif", ".tiff")):
                if not HAS_RASTERIO:
                    raise ImportError("Reading GeoTIFF requires rasterio (`pip install rasterio`).")
                with rasterio.open(input_data) as src:
                    arr = src.read().astype(np.float32)
                    geo_profile = src.profile.copy()
            else:
                # Try loading standard image
                img = Image.open(input_data).convert("L")
                arr = np.array(img, dtype=np.float32)[None]
        else:
            arr = np.array(input_data)

        # Handle dataset_PSR 4-channel [opt, cpr, srd, trt] layout cleanly
        ch_axis = 1 if arr.ndim == 4 else 0
        if arr.ndim in (3, 4) and arr.shape[ch_axis] == 4:
            path_hint = ("west_patches" in input_path_str.lower() or "east_patches" in input_path_str.lower())
            if psr_skip_opt or path_hint:
                print("Detected PSR [optical, cpr, srd, trt] layout -> extracting SAR channels [1:4].")
                arr = arr[:, 1:4] if arr.ndim == 4 else arr[1:4]

        # Handle 4D patch stack (N, C, H, W)
        if arr.ndim == 4:
            if index is not None and str(index).lower() != "all":
                idx = int(index)
                if idx < 0 or idx >= len(arr):
                    raise IndexError(f"Patch index {idx} out of range [0, {len(arr)-1}].")
                sar_patch = arr[idx]
                if not np.isfinite(sar_patch).any():
                    raise ValueError(f"Patch {idx} contains 100% NaN SAR pixels (no radar coverage).")
                sar_norm = self.prepare_sar_patch(sar_patch, denoise=denoise)
                out_opt, lat = self.synthesize_tile(sar_norm)
                return {
                    "optical_norm": out_opt,
                    "optical_uint8": ((np.clip(out_opt[0], -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8),
                    "sar_norm": sar_norm,
                    "latency_ms": lat,
                    "count": 1,
                    "geo_profile": None,
                }
            else:
                # Process all valid patches in stack (skipping all-NaN patches)
                results = []
                total_lat = 0.0
                skipped_nan = 0
                for i in range(len(arr)):
                    s_patch = arr[i]
                    if not np.isfinite(s_patch).any():
                        skipped_nan += 1
                        continue
                    s_norm = self.prepare_sar_patch(s_patch, denoise=denoise)
                    o_opt, lat = self.synthesize_tile(s_norm)
                    total_lat += lat
                    results.append({
                        "index": i,
                        "optical_norm": o_opt,
                        "optical_uint8": ((np.clip(o_opt[0], -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8),
                        "sar_norm": s_norm,
                        "latency_ms": lat,
                    })
                if skipped_nan > 0:
                    print(f"Skipped {skipped_nan} patches with 100% NaN SAR data; synthesized {len(results)} valid patches.")
                return {
                    "batch": results,
                    "count": len(results),
                    "total_latency_ms": total_lat,
                    "mean_latency_ms": total_lat / max(1, len(results)),
                    "geo_profile": None,
                }

        # Handle single raster (C, H, W) or (H, W)
        if arr.ndim == 2:
            arr = arr[None]

        sar_norm = self.prepare_sar_patch(arr, denoise=denoise)
        _, H, W = sar_norm.shape

        if H == 256 and W == 256:
            out_opt, lat = self.synthesize_tile(sar_norm)
        else:
            out_opt, lat = self.synthesize_raster(arr, tile_size=256, denoise=denoise)

        opt_uint8 = ((np.clip(out_opt[0], -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
        return {
            "optical_norm": out_opt,
            "optical_uint8": opt_uint8,
            "sar_norm": sar_norm,
            "latency_ms": lat,
            "count": 1,
            "geo_profile": geo_profile,
        }


def save_inference_result(res, out_dir, base_name="sample", save_npy=False,
                          save_tif=False, visualize=True):
    """Saves synthesized optical images, side-by-side grids, and metadata."""
    os.makedirs(out_dir, exist_ok=True)
    saved_files = []

    # Case A: Batch of patches
    if "batch" in res:
        print(f"\nSaving {len(res['batch'])} synthesized patches to: {out_dir}")
        for item in res["batch"]:
            idx = item["index"]
            opt_uint8 = item["optical_uint8"]
            opt_path = os.path.join(out_dir, f"{base_name}_{idx:04d}_optical.png")
            Image.fromarray(opt_uint8).save(opt_path)
            saved_files.append(opt_path)

            if visualize:
                grid = create_side_by_side_visual(
                    item["sar_norm"], opt_uint8, title=f"Patch #{idx}"
                )
                grid_path = os.path.join(out_dir, f"{base_name}_{idx:04d}_comparison.png")
                grid.save(grid_path)
                saved_files.append(grid_path)

            if save_npy:
                npy_path = os.path.join(out_dir, f"{base_name}_{idx:04d}_optical.npy")
                np.save(npy_path, item["optical_norm"])
                saved_files.append(npy_path)

        print(f"Batch inference complete: {res['count']} patches in {res['total_latency_ms']:.1f} ms "
              f"({res['mean_latency_ms']:.2f} ms/tile -> {1000.0/res['mean_latency_ms']:.1f} tiles/sec)")
        return saved_files

    # Case B: Single patch or full raster
    opt_uint8 = res["optical_uint8"]
    opt_path = os.path.join(out_dir, f"{base_name}_optical.png")
    Image.fromarray(opt_uint8).save(opt_path)
    saved_files.append(opt_path)

    if visualize:
        grid = create_side_by_side_visual(res["sar_norm"], opt_uint8, title=base_name)
        grid_path = os.path.join(out_dir, f"{base_name}_comparison.png")
        grid.save(grid_path)
        saved_files.append(grid_path)

    if save_npy:
        npy_path = os.path.join(out_dir, f"{base_name}_optical.npy")
        np.save(npy_path, res["optical_norm"])
        saved_files.append(npy_path)

    if save_tif and HAS_RASTERIO:
        tif_path = os.path.join(out_dir, f"{base_name}_optical.tif")
        profile = res.get("geo_profile") or {}
        h, w = opt_uint8.shape
        profile.update(
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=rasterio.uint8,
            nodata=None
        )
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(opt_uint8, 1)
        saved_files.append(tif_path)

    print(f"\nInference Results Saved:")
    for f in saved_files:
        print(f"  -> {f}")
    print(f"Latency: {res['latency_ms']:.2f} ms (Throughput: {1000.0/max(res['latency_ms'], 1e-3):.1f} FPS)")
    return saved_files


def main():
    p = argparse.ArgumentParser(
        description="Lunar SAR-to-Optical PSR Inference Pipeline."
    )
    p.add_argument("--input", "-i", required=True,
                   help="Path to SAR patch/stack (.npy), GeoTIFF (.tif/.tiff), or image.")
    p.add_argument("--ckpt", "-c", default=None,
                   help=f"Path to model checkpoint .pt (default: {find_default_checkpoint()}).")
    p.add_argument("--meta", "-m", default=None,
                   help="Path to meta.json with channel calibration ranges.")
    p.add_argument("--out", "-o", default="runs/inference",
                   help="Output directory (default: runs/inference).")
    p.add_argument("--index", default="0",
                   help="Patch index to predict from a 4D array (e.g. 0, 1, or 'all'). Default: 0.")
    p.add_argument("--all", action="store_true",
                   help="Synthesize all patches if input is a 4D stack.")
    p.add_argument("--denoise", type=int, default=0,
                   help="Lee filter window size (0 to disable). Default: 0.")
    p.add_argument("--save-npy", action="store_true",
                   help="Save synthesized optical output as raw float32 .npy in [-1, 1].")
    p.add_argument("--save-tif", action="store_true",
                   help="Save synthesized optical output as GeoTIFF (if rasterio installed).")
    p.add_argument("--no-visualize", action="store_true",
                   help="Disable generating side-by-side SAR vs Optical comparison images.")
    p.add_argument("--psr-skip-opt", action="store_true",
                   help="Slice channels [1:4] when passing 4-channel [opt, cpr, srd, trt] PSR stacks.")
    p.add_argument("--device", default=None,
                   help="Device: 'cuda', 'cpu', or None (auto-detect).")
    args = p.parse_args()

    # Determine index argument
    index_param = "all" if args.all else args.index

    # Base name for outputs
    input_base = os.path.splitext(os.path.basename(args.input))[0]

    # Initialize synthesizer
    synth = LunarOpticalSynthesizer(
        ckpt_path=args.ckpt,
        meta_path=args.meta,
        device=args.device,
    )

    # Run prediction
    res = synth.predict(
        args.input,
        index=index_param,
        denoise=args.denoise,
        psr_skip_opt=args.psr_skip_opt,
    )

    # Save output
    save_inference_result(
        res=res,
        out_dir=args.out,
        base_name=input_base,
        save_npy=args.save_npy,
        save_tif=args.save_tif,
        visualize=not args.no_visualize,
    )


if __name__ == "__main__":
    main()
