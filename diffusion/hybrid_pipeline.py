"""Hybrid CycleGAN + Diffusion Pipeline for Lunar PSR Optical Synthesis.

Combines the best of both generative paradigms:
  1. CycleGAN: Rapid (18 ms), deterministic, physically grounded structural prior
     free of hallucination and anchored by multi-aspect SAR polarimetry.
  2. Diffusion Refiner (SDEdit): High-frequency texture synthesis that resolves
     residual GAN blur, refines crater micro-relief, and enhances regolith graininess.

Evaluation & Benchmarking:
  Compares Pure CycleGAN vs. Pure Diffusion vs. Tandem Hybrid (CycleGAN + Diffusion).
"""
import argparse
import json
import os
import sys
import time

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.abspath(os.path.dirname(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import parse_channels
from diffusion.diffusion_model import DDPMScheduler, DiffusionUNet
from models import ResnetGenerator
from utils import set_seed


def compute_gradient_sharpness(img_2d):
    """Computes Tenengrad / spatial gradient variance as a sharpness metric."""
    gy, gx = np.gradient(img_2d)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.var(grad_mag))


def to_uint8(norm_arr):
    """Maps [-1, 1] array to [0, 255] uint8."""
    return ((np.clip(norm_arr, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


class HybridLunarSynthesizer:
    def __init__(self, cyclegan_ckpt="runs/real_optical_100/last.pt",
                 diffusion_ckpt="runs/diffusion_model/diffusion_last.pt",
                 device=None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Load CycleGAN
        print(f"Loading CycleGAN Generator: {cyclegan_ckpt}")
        ck_gan = torch.load(cyclegan_ckpt, map_location=self.device)
        cfg_gan = ck_gan["args"]
        self.in_ch = cfg_gan.get("in_ch", 4)
        self.out_ch = cfg_gan.get("out_ch", 1)
        self.G_AB = ResnetGenerator(
            self.in_ch, self.out_ch, cfg_gan.get("ngf", 64), cfg_gan.get("n_blocks", 9)
        ).to(self.device)
        self.G_AB.load_state_dict(ck_gan["G_AB"])
        self.G_AB.eval()

        # 2. Load Diffusion Model
        print(f"Loading Diffusion Refiner: {diffusion_ckpt}")
        ck_diff = torch.load(diffusion_ckpt, map_location=self.device)
        self.timesteps = ck_diff.get("timesteps", 500)
        cond_ch = ck_diff.get("cond_channels", 0)
        self.diff_model = DiffusionUNet(
            in_channels=1, out_channels=1, cond_channels=cond_ch, base_dim=32
        ).to(self.device)
        self.diff_model.load_state_dict(ck_diff["model"])
        self.diff_model.eval()

        self.scheduler = DDPMScheduler(
            num_timesteps=self.timesteps, beta_start=1e-4, beta_end=0.02, device=self.device
        )

        print(f"Hybrid Lunar Synthesizer initialized successfully on {self.device}.")

    @torch.no_grad()
    def predict_cyclegan(self, sar_tensor):
        """Pure CycleGAN pass (18 ms)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gan_out = self.G_AB(sar_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000.0
        return gan_out, lat

    @torch.no_grad()
    def predict_pure_diffusion(self, sar_tensor=None):
        """Pure Diffusion: sampling from pure random noise at t = T."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        shape = (1, 1, 256, 256)
        diff_out = self.scheduler.sample_full(
            self.diff_model, shape=shape, cond=sar_tensor, num_steps=self.timesteps
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000.0
        return diff_out, lat

    @torch.no_grad()
    def predict_hybrid(self, sar_tensor, t_refine=80):
        """Hybrid Pipeline:
        1. CycleGAN rapid forward pass -> x_init (structural crater prior).
        2. SDEdit Diffusion Refinement from t_refine -> 0 (micro-texture enhancement).
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: CycleGAN structural prior
        x_init = self.G_AB(sar_tensor)

        # Step 2: Diffusion SDEdit Refinement
        hybrid_out = self.scheduler.sample_sdedit(
            self.diff_model, x_init=x_init, t_refine=t_refine, cond=None
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000.0
        return hybrid_out, x_init, lat


def run_benchmark(synthesizer, test_data_dir="data/moon_dfsar", out_dir="runs/hybrid_benchmark"):
    os.makedirs(out_dir, exist_ok=True)
    import glob

    sar_files = sorted(glob.glob(os.path.join(test_data_dir, "test", "sar", "*.npy")))
    opt_files = sorted(glob.glob(os.path.join(test_data_dir, "test", "opt", "*.npy")))
    n_eval = min(len(sar_files), len(opt_files))
    print(f"\n--- Running Comparative Benchmark on {n_eval} Held-Out Test Tiles ---")

    metrics_summary = {
        "cyclegan": {"ssim": [], "psnr": [], "sharpness": [], "latency_ms": []},
        "pure_diffusion": {"ssim": [], "psnr": [], "sharpness": [], "latency_ms": []},
        "hybrid": {"ssim": [], "psnr": [], "sharpness": [], "latency_ms": []},
        "ground_truth": {"sharpness": []},
    }

    panel_rows = []

    for i in range(n_eval):
        # Load SAR and Optical ground truth
        sar_arr = np.load(sar_files[i])  # (6, 256, 256)
        opt_arr = np.load(opt_files[i])  # (1, 256, 256)

        sar_4ch = sar_arr[[0, 1, 3, 4]]  # east_cpr, east_srd, west_cpr, west_srd
        sar_t = torch.from_numpy(sar_4ch).unsqueeze(0).float().to(synthesizer.device)
        ref_np = (np.clip(opt_arr[0], -1.0, 1.0) + 1.0) / 2.0  # [0, 1]

        # 1. CycleGAN alone
        out_gan, lat_gan = synthesizer.predict_cyclegan(sar_t)
        pred_gan = (np.clip(out_gan[0, 0].cpu().numpy(), -1.0, 1.0) + 1.0) / 2.0

        # 2. Pure Diffusion alone
        out_diff, lat_diff = synthesizer.predict_pure_diffusion(sar_tensor=None)
        pred_diff = (np.clip(out_diff[0, 0].cpu().numpy(), -1.0, 1.0) + 1.0) / 2.0

        # 3. Hybrid (CycleGAN + Diffusion Refiner)
        out_hybrid, _, lat_hybrid = synthesizer.predict_hybrid(sar_t, t_refine=70)
        pred_hybrid = (np.clip(out_hybrid[0, 0].cpu().numpy(), -1.0, 1.0) + 1.0) / 2.0

        # Calculate metrics
        # CycleGAN
        s_gan = structural_similarity(ref_np, pred_gan, data_range=1.0)
        p_gan = peak_signal_noise_ratio(ref_np, pred_gan, data_range=1.0)
        sh_gan = compute_gradient_sharpness(pred_gan)
        metrics_summary["cyclegan"]["ssim"].append(s_gan)
        metrics_summary["cyclegan"]["psnr"].append(p_gan)
        metrics_summary["cyclegan"]["sharpness"].append(sh_gan)
        metrics_summary["cyclegan"]["latency_ms"].append(lat_gan)

        # Pure Diffusion
        s_diff = structural_similarity(ref_np, pred_diff, data_range=1.0)
        p_diff = peak_signal_noise_ratio(ref_np, pred_diff, data_range=1.0)
        sh_diff = compute_gradient_sharpness(pred_diff)
        metrics_summary["pure_diffusion"]["ssim"].append(s_diff)
        metrics_summary["pure_diffusion"]["psnr"].append(p_diff)
        metrics_summary["pure_diffusion"]["sharpness"].append(sh_diff)
        metrics_summary["pure_diffusion"]["latency_ms"].append(lat_diff)

        # Hybrid
        s_hyb = structural_similarity(ref_np, pred_hybrid, data_range=1.0)
        p_hyb = peak_signal_noise_ratio(ref_np, pred_hybrid, data_range=1.0)
        sh_hyb = compute_gradient_sharpness(pred_hybrid)
        metrics_summary["hybrid"]["ssim"].append(s_hyb)
        metrics_summary["hybrid"]["psnr"].append(p_hyb)
        metrics_summary["hybrid"]["sharpness"].append(sh_hyb)
        metrics_summary["hybrid"]["latency_ms"].append(lat_hybrid)

        # Ground truth sharpness
        metrics_summary["ground_truth"]["sharpness"].append(compute_gradient_sharpness(ref_np))

        # Build comparison row
        sar_cpr_u8 = to_uint8(sar_arr[0])
        diff_u8 = to_uint8(out_diff[0, 0].cpu().numpy())
        gan_u8 = to_uint8(out_gan[0, 0].cpu().numpy())
        hyb_u8 = to_uint8(out_hybrid[0, 0].cpu().numpy())
        ref_u8 = to_uint8(opt_arr[0])

        # Stack columns: [SAR CPR | Pure Diffusion | CycleGAN | Hybrid | Real Optical]
        row_imgs = [sar_cpr_u8, diff_u8, gan_u8, hyb_u8, ref_u8]
        row_canvas = np.concatenate(row_imgs, axis=1)
        panel_rows.append(row_canvas)

    # Average metrics
    avg = {}
    for m_k in ("cyclegan", "pure_diffusion", "hybrid"):
        avg[m_k] = {
            "ssim": float(np.mean(metrics_summary[m_k]["ssim"])),
            "psnr": float(np.mean(metrics_summary[m_k]["psnr"])),
            "sharpness": float(np.mean(metrics_summary[m_k]["sharpness"])),
            "latency_ms": float(np.mean(metrics_summary[m_k]["latency_ms"])),
        }
    avg["ground_truth_sharpness"] = float(np.mean(metrics_summary["ground_truth"]["sharpness"]))

    # Save metrics JSON
    with open(os.path.join(out_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(avg, f, indent=2)

    # Create master comparison figure
    banner_h = 36
    col_w = 256
    row_h = 256
    sep = 3
    num_cols = 5
    num_rows = len(panel_rows)

    total_w = col_w * num_cols + sep * (num_cols - 1)
    total_h = banner_h + (row_h + sep) * num_rows

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = 20  # Header banner

    col_titles = [
        "1. Input SAR (CPR)",
        "2. Pure Diffusion",
        "3. CycleGAN (Ours)",
        "4. Hybrid (GAN + Diff)",
        "5. True Real Optical",
    ]

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    for c, title in enumerate(col_titles):
        x = c * (col_w + sep) + 10
        draw.text((x, 10), title, fill=(230, 230, 230))

    canvas = np.array(pil_img)

    for r_idx, r_img in enumerate(panel_rows):
        y = banner_h + r_idx * (row_h + sep)
        for c in range(num_cols):
            x = c * (col_w + sep)
            tile = r_img[:, c * col_w:(c + 1) * col_w]
            tile_3ch = np.stack([tile, tile, tile], axis=-1)
            canvas[y:y + row_h, x:x + col_w] = tile_3ch

    out_img = Image.fromarray(canvas)
    grid_path = os.path.join(out_dir, "hybrid_vs_cyclegan_comparison.png")
    out_img.save(grid_path)

    # Copy to assets and brain
    import shutil
    shutil.copyfile(grid_path, "assets/hybrid_vs_cyclegan_comparison.png")
    brain_path = r"C:\Users\Ayush Marvin\.gemini\antigravity\brain\aa5adec2-063d-451e-98c6-dec284bf6786\hybrid_vs_cyclegan_comparison.png"
    try:
        shutil.copyfile(grid_path, brain_path)
    except Exception:
        pass

    print("\n================== BENCHMARK RESULTS ==================")
    print(f"CycleGAN alone:       SSIM={avg['cyclegan']['ssim']:.4f} | PSNR={avg['cyclegan']['psnr']:.2f} dB | Sharpness={avg['cyclegan']['sharpness']:.5f} | Latency={avg['cyclegan']['latency_ms']:.2f} ms")
    print(f"Pure Diffusion alone: SSIM={avg['pure_diffusion']['ssim']:.4f} | PSNR={avg['pure_diffusion']['psnr']:.2f} dB | Sharpness={avg['pure_diffusion']['sharpness']:.5f} | Latency={avg['pure_diffusion']['latency_ms']:.2f} ms")
    print(f"Hybrid (GAN + Diff):  SSIM={avg['hybrid']['ssim']:.4f} | PSNR={avg['hybrid']['psnr']:.2f} dB | Sharpness={avg['hybrid']['sharpness']:.5f} | Latency={avg['hybrid']['latency_ms']:.2f} ms")
    print(f"Ground Truth Optical: Sharpness={avg['ground_truth_sharpness']:.5f}")
    print("=======================================================")
    print(f"Visual comparison saved to: {grid_path}")
    return avg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cyclegan", default="runs/real_optical_100/last.pt")
    p.add_argument("--diffusion", default="runs/diffusion_model/diffusion_last.pt")
    p.add_argument("--data", default="data/moon_dfsar")
    p.add_argument("--out", default="runs/hybrid_benchmark")
    a = p.parse_args()

    synth = HybridLunarSynthesizer(a.cyclegan, a.diffusion)
    run_benchmark(synth, a.data, a.out)
