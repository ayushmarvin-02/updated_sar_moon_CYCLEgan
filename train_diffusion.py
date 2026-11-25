"""Train Lunar Diffusion Model & Texture Refiner on real lunar optical data.

Optimizes standard DDPM objective:
  L_diff = E_{x_0, eps, t} [ || eps - eps_theta(x_t, t, cond) ||^2 ]

Supports:
  1. Unconditional lunar regolith texture diffusion (for SDEdit stochastic refinement).
  2. Conditional diffusion conditioned on SAR / CycleGAN structural prior.
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diffusion_model import DDPMScheduler, DiffusionUNet
from models import ResnetGenerator
from utils import set_seed


class LunarOpticalDataset(Dataset):
    """Loads normalized lunar optical tiles with spatial augmentations."""
    def __init__(self, opt_dir, sar_dir=None, channels_idx=None, augment=True):
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.npy")))
        if not self.opt_files:
            raise FileNotFoundError(f"No optical tiles found in {opt_dir}")
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.npy"))) if sar_dir else []
        self.channels_idx = channels_idx or [0, 1, 3, 4]
        self.augment = augment

    def __len__(self):
        return len(self.opt_files)

    def __getitem__(self, idx):
        # Load optical
        opt = np.load(self.opt_files[idx])  # (1, 256, 256) in [-1, 1]
        if opt.ndim == 2:
            opt = opt[None]

        sar = None
        if self.sar_files:
            sar_idx = idx % len(self.sar_files)
            sar_raw = np.load(self.sar_files[sar_idx])
            sar = sar_raw[self.channels_idx] if sar_raw.shape[0] > max(self.channels_idx) else sar_raw

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                opt = opt[:, :, ::-1]
                if sar is not None:
                    sar = sar[:, :, ::-1]
            if random.random() < 0.5:
                opt = opt[:, ::-1, :]
                if sar is not None:
                    sar = sar[:, ::-1, :]
            k = random.randint(0, 3)
            if k > 0:
                opt = np.rot90(opt, k, axes=(1, 2))
                if sar is not None:
                    sar = np.rot90(sar, k, axes=(1, 2))

        opt = np.ascontiguousarray(opt)
        item = {"optical": torch.from_numpy(opt).float()}
        if sar is not None:
            sar = np.ascontiguousarray(sar)
            item["sar"] = torch.from_numpy(sar).float()

        return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/moon_dfsar", help="Root data dir.")
    p.add_argument("--out", default="runs/diffusion_model", help="Checkpoint output dir.")
    p.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=500, help="Total diffusion timesteps.")
    p.add_argument("--cond_sar", action="store_true", help="Condition directly on 4-channel SAR.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    opt_dir = os.path.join(args.data, "train", "opt")
    sar_dir = os.path.join(args.data, "train", "sar") if args.cond_sar else None

    ds = LunarOpticalDataset(opt_dir, sar_dir=sar_dir, augment=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    cond_ch = 4 if args.cond_sar else 0
    model = DiffusionUNet(in_channels=1, out_channels=1, cond_channels=cond_ch, base_dim=32).to(dev)
    scheduler = DDPMScheduler(num_timesteps=args.timesteps, beta_start=1e-4, beta_end=0.02, device=dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"Training Lunar Diffusion Model on {len(ds)} tiles:")
    print(f"  Device: {dev}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Conditioning channels: {cond_ch}")
    print(f"  Epochs: {args.epochs} (Batch size: {args.batch_size})")

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            x_0 = batch["optical"].to(dev)
            b = x_0.shape[0]

            # Sample random diffusion timestep t uniformly
            t = torch.randint(0, args.timesteps, (b,), device=dev).long()

            # Sample Gaussian noise
            noise = torch.randn_like(x_0)

            # Forward noising q(x_t | x_0)
            x_t = scheduler.add_noise(x_0, noise, t)

            # Condition
            cond = batch["sar"].to(dev) if args.cond_sar else None

            # Model predicts noise eps_theta(x_t, t, cond)
            noise_pred = model(x_t, t, cond=cond)

            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}]  Diffusion MSE Loss: {avg_loss:.5f}  "
                  f"Elapsed: {time.time() - t_start:.1f}s")

    # Save model and config
    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "timesteps": args.timesteps,
        "cond_channels": cond_ch,
    }
    ckpt_path = os.path.join(args.out, "diffusion_last.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Diffusion model training complete! Saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
