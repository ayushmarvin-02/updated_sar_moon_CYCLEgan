"""Small shared helpers."""
import random

import numpy as np
import torch
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_img(t, first_channel_only=False):
    """(C,H,W) tensor in [-1,1] -> HxWx3 uint8. SAR is shown as its first channel."""
    t = t.detach().cpu().float().clamp(-1, 1)
    t = (t + 1) / 2
    if first_channel_only or t.shape[0] == 1:
        t = t[:1].repeat(3, 1, 1)
    elif t.shape[0] == 2:
        t = torch.cat([t, t[:1] * 0], 0)
    elif t.shape[0] > 3:
        t = t[:3]
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def save_grid(rows, path):
    """rows: list of lists of HxWx3 uint8 images; saved as one PNG."""
    row_imgs = [np.concatenate(r, axis=1) for r in rows]
    Image.fromarray(np.concatenate(row_imgs, axis=0)).save(path)
