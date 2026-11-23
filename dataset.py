"""Datasets for the prepared tile folders produced by prepare_data.py.

Layout:  <root>/<split>/sar/*.npy   (3, H, W) float32 in [-1, 1]: [vv, vh, ratio]
         <root>/<split>/opt/*.npy   (1 or 3, H, W) float32 in [-1, 1]
"""
import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNEL_INDEX = {
    # Earth proxy (Sentinel-1)
    "vv": 0, "vh": 1, "ratio": 2,
    # Lunar SAR aliases (Mini-RF / DFSAR)
    "s1": 0, "cpr": 1, "s2": 1,
    "sc": 0, "oc": 1,
    # Generic index aliases
    "ch0": 0, "ch1": 1, "ch2": 2,
}


def parse_channels(spec):
    names = [c.strip().lower() for c in spec.split(",") if c.strip()]
    for n in names:
        if n not in CHANNEL_INDEX:
            raise ValueError(f"unknown channel '{n}', choose from {list(CHANNEL_INDEX)}")
    return names


def _augment(x):
    """Random flips / 90-degree rotations on a (C,H,W) array."""
    if random.random() < 0.5:
        x = x[:, :, ::-1]
    if random.random() < 0.5:
        x = x[:, ::-1, :]
    k = random.randrange(4)
    if k:
        x = np.rot90(x, k, axes=(1, 2))
    return np.ascontiguousarray(x)


def _random_crop(x, size):
    _, h, w = x.shape
    if size >= h and size >= w:
        return x
    y = random.randint(0, h - size)
    xx = random.randint(0, w - size)
    return x[:, y:y + size, xx:xx + size]


class _TileSet:
    def __init__(self, root, split, channels):
        d = os.path.join(root, split)
        self.sar_files = sorted(glob(os.path.join(d, "sar", "*.npy")))
        self.opt_files = sorted(glob(os.path.join(d, "opt", "*.npy")))
        if not self.sar_files or not self.opt_files:
            raise FileNotFoundError(f"no tiles found under {d}")
        self.names = parse_channels(channels) if isinstance(channels, str) else list(channels)
        self.idx = [CHANNEL_INDEX[n] for n in self.names]
        self.opt_channels = np.load(self.opt_files[0], mmap_mode="r").shape[0]

    def load_sar(self, i):
        return np.load(self.sar_files[i])[self.idx]

    def load_opt(self, i):
        return np.load(self.opt_files[i])


class UnpairedDataset(Dataset):
    """Domain A (SAR) index is sequential; domain B (optical) index is random,
    so the generators never see co-registered pairs during training."""

    def __init__(self, root, split="train", channels="vv,vh", crop=256, augment=True):
        self.t = _TileSet(root, split, channels)
        self.crop, self.augment = crop, augment
        self.opt_channels = self.t.opt_channels
        self.sar_channels = len(self.t.idx)

    def __len__(self):
        return len(self.t.sar_files)

    def __getitem__(self, i):
        a = self.t.load_sar(i)
        b = self.t.load_opt(random.randrange(len(self.t.opt_files)))
        if self.augment:
            a, b = _augment(a), _augment(b)
        a, b = _random_crop(a, self.crop), _random_crop(b, self.crop)
        return {"A": torch.from_numpy(np.ascontiguousarray(a)),
                "B": torch.from_numpy(np.ascontiguousarray(b))}


class PairedDataset(Dataset):
    """Evaluation only: SAR tile i with the co-registered optical tile i
    (valid for the Earth proxy where S1/S2 share a grid)."""

    def __init__(self, root, split="test", channels="vv,vh"):
        self.t = _TileSet(root, split, channels)
        if len(self.t.sar_files) != len(self.t.opt_files):
            raise ValueError("paired evaluation needs equal SAR/optical tile counts")
        self.opt_channels = self.t.opt_channels
        self.sar_channels = len(self.t.idx)

    def __len__(self):
        return len(self.t.sar_files)

    def __getitem__(self, i):
        return {"A": torch.from_numpy(self.t.load_sar(i).copy()),
                "B": torch.from_numpy(self.t.load_opt(i).copy())}
