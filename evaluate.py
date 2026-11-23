"""Evaluate a trained generator on the held-out spatial test split.

The Earth proxy IS co-registered, so although training is unpaired we can score
against the true optical tile.  Because an unpaired model does not reproduce exact
radiometry, structure-oriented metrics (SSIM, correlation) matter more than PSNR.
Reported next to a constant "mean training image" baseline for context.
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import PairedDataset
from models import ResnetGenerator
from utils import save_grid, to_img


def to01(t):
    return ((t.clamp(-1, 1) + 1) / 2).cpu().numpy()  # (C,H,W) in [0,1]


def metrics(pred, ref):
    p, r = np.moveaxis(pred, 0, -1), np.moveaxis(ref, 0, -1)
    ch_axis = -1 if p.shape[-1] > 1 else None
    if ch_axis is None:
        p, r = p[..., 0], r[..., 0]
    ssim = structural_similarity(r, p, data_range=1.0, channel_axis=ch_axis)
    psnr = peak_signal_noise_ratio(r, p, data_range=1.0)
    a, b = pred.mean(0).ravel(), ref.mean(0).ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else 0.0
    return ssim, psnr, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_samples", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = ck["args"]
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"eval_{a.split}")
    os.makedirs(out, exist_ok=True)

    G_AB = ResnetGenerator(cfg["in_ch"], cfg["out_ch"], cfg["ngf"], cfg["n_blocks"]).to(dev)
    G_BA = ResnetGenerator(cfg["out_ch"], cfg["in_ch"], cfg["ngf"], cfg["n_blocks"]).to(dev)
    G_AB.load_state_dict(ck["G_AB"])
    G_BA.load_state_dict(ck["G_BA"])
    G_AB.eval()
    G_BA.eval()

    ds = PairedDataset(a.data, a.split, cfg["channels"])

    # constant baseline: mean of (a subsample of) the training optical tiles
    train_files = sorted(glob(os.path.join(a.data, "train", "opt", "*.npy")))
    step = max(1, len(train_files) // 200)
    mean_img = np.mean([np.load(f) for f in train_files[::step]], axis=0)
    mean_img = np.clip((mean_img + 1) / 2, 0, 1)
    mean_pix = mean_img.mean(axis=(1, 2), keepdims=True)

    rows, res, base, cyc = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            A, B = s["A"][None].to(dev), s["B"][None].to(dev)
            fake_B = G_AB(A)
            rec_A = G_BA(fake_B)
            pred, ref = to01(fake_B[0]), to01(B[0])
            res.append(metrics(pred, ref))
            base.append(metrics(np.broadcast_to(mean_pix, ref.shape).copy(), ref))
            cyc.append(torch.nn.functional.l1_loss(rec_A, A).item())
            if i < a.n_samples:
                rows.append([to_img(A[0], True), to_img(fake_B[0]), to_img(B[0]),
                             to_img(rec_A[0], True)])

    res, base = np.array(res), np.array(base)
    summary = {
        "n_tiles": len(ds), "channels": cfg["channels"],
        "model": {"ssim": res[:, 0].mean(), "psnr_db": res[:, 1].mean(),
                  "corr": res[:, 2].mean(), "ssim_std": res[:, 0].std()},
        "mean_image_baseline": {"ssim": base[:, 0].mean(), "psnr_db": base[:, 1].mean()},
        "cycle_l1_sar": float(np.mean(cyc)),
    }
    summary = json.loads(json.dumps(summary, default=float))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        save_grid(rows, os.path.join(out, "samples.png"))  # SAR | generated | true optical | cycle-rec SAR
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
