"""Generate a fake co-registered S1/S2 scene so the pipeline can be tested
without downloading anything.  NOT real data - for smoke tests only."""
import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter


def norm(x):
    return (x - x.min()) / (np.ptp(x) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw_synthetic")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.size
    base = norm(gaussian_filter(rng.standard_normal((n, n)), 12))   # "land cover"
    rough = norm(gaussian_filter(rng.standard_normal((n, n)), 4))   # "roughness"

    vv_lin = 0.02 + 0.25 * rough * (0.3 + base)
    vh_lin = vv_lin * (0.08 + 0.15 * base)
    speck = lambda: rng.gamma(4.0, 1 / 4.0, (n, n))                 # 4-look speckle
    s1 = np.stack([10 * np.log10(vv_lin * speck()), 10 * np.log10(vh_lin * speck())])

    r = 800 + 3000 * base + 400 * rough
    g = 900 + 2500 * base + 300 * rough
    b = 700 + 2000 * base
    s2 = np.stack([r, g, b])

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "s1.npy"), s1.astype(np.float32))
    np.save(os.path.join(a.out, "s2.npy"), s2.astype(np.float32))
    print("wrote", a.out, s1.shape, s2.shape)


if __name__ == "__main__":
    main()
