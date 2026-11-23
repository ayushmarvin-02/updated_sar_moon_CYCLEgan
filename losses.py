"""Losses and helpers for CycleGAN training."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """LSGAN objective (MSE against 1/0 targets)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class ImagePool:
    """History buffer of generated images to stabilise the discriminators."""

    def __init__(self, size=50):
        self.size = size
        self.imgs = []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.imgs) < self.size:
                self.imgs.append(img)
                out.append(img)
            elif random.random() > 0.5:
                k = random.randrange(self.size)
                old = self.imgs[k].clone()
                self.imgs[k] = img
                out.append(old)
            else:
                out.append(img)
        return torch.cat(out, 0)


def _sobel_mag(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_consistency(sar, opt):
    """EXPERIMENTAL structure prior: edge-strength maps of the SAR input and the
    generated optical image should agree (after per-image normalisation).
    Helps discourage hallucinated structure; weight defaults to 0 (off)."""
    s = sar.mean(1, keepdim=True)
    o = opt.mean(1, keepdim=True)
    ms, mo = _sobel_mag(s), _sobel_mag(o)
    ms = ms / (ms.mean((2, 3), keepdim=True) + 1e-6)
    mo = mo / (mo.mean((2, 3), keepdim=True) + 1e-6)
    return F.l1_loss(ms, mo)
