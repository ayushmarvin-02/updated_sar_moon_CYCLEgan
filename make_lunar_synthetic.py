"""Generate synthetic co-registered lunar SAR and optical scenes with realistic craters.

Used for smoke-testing the lunar pipeline without downloading PDS products.
Simulates:
  - Synthetic topography with impact craters (bowl depression + raised rim)
  - Lunar optical (LROC NAC style): directional solar hillshade + dark crater floor shadows
  - Lunar SAR (Mini-RF / DFSAR style):
      * S1 (total backscatter): strong reflections on crater rims and slopes
      * CPR (circular polarization ratio): elevated on rough/bouldery crater rims and ejecta
      * Multi-look radar speckle noise
"""
import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw_lunar_synthetic")
    ap.add_argument("--size", type=int, default=1024, help="Grid size (H=W).")
    ap.add_argument("--n_craters", type=int, default=35, help="Number of impact craters.")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n = a.size

    # Background terrain (undulating topography)
    topo = gaussian_filter(rng.standard_normal((n, n)), 30) * 150.0

    # Add impact craters: inverted paraboloids with raised rims
    cpr_base = np.full((n, n), 0.3, dtype=np.float32)  # typical lunar regolith CPR ~ 0.3
    y_coords, x_coords = np.indices((n, n))

    # Track Permanently Shadowed Regions (PSRs):
    # Deep interiors of craters where low polar grazing sunlight never reaches
    psr_mask = np.zeros((n, n), dtype=np.float32)

    for _ in range(a.n_craters):
        cx = rng.uniform(0.1 * n, 0.9 * n)
        cy = rng.uniform(0.1 * n, 0.9 * n)
        radius = rng.uniform(15, 80)
        depth = radius * 0.25

        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        in_crater = dist < radius
        in_rim = (dist >= radius) & (dist < radius * 1.4)

        # Crater cavity
        cavity = np.maximum(0, depth * (1.0 - (dist / radius) ** 2))
        topo -= cavity

        # Raised rim & ejecta blanket
        rim_height = depth * 0.25 * (1.0 - (dist - radius) / (0.4 * radius))
        topo[in_rim] += rim_height[in_rim]

        # Roughness / ejecta enhances CPR
        cpr_boost = np.clip(1.5 * (1.0 - (dist - radius) / (0.8 * radius)), 0, 1.8)
        cpr_base[dist < radius * 1.8] += cpr_boost[dist < radius * 1.8]

        # Deep crater floor qualifies as PSR
        if radius > 25:
            psr_mask[dist < radius * 0.60] = 1.0

    # Optical: grazing solar illumination (simulating lunar polar low sun angle)
    # Gradient of topography
    dy, dx = np.gradient(topo)
    sun_azimuth = np.radians(45.0)  # coming from top-left
    sun_elevation = np.radians(15.0)  # low polar sun angle
    sun_vec = np.array([np.cos(sun_elevation) * np.cos(sun_azimuth),
                        np.cos(sun_elevation) * np.sin(sun_azimuth),
                        np.sin(sun_elevation)])

    # Surface normals
    normals = np.stack([-dx, -dy, np.ones_like(topo)], axis=-1)
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm_len + 1e-8)

    # Hillshade dot product
    illumination = np.clip(np.sum(normals * sun_vec, axis=-1), 0.0, 1.0)
    albedo = 0.12 + 0.04 * gaussian_filter(rng.standard_normal((n, n)), 20)  # lunar albedo ~ 0.12
    optical = np.clip(albedo * illumination, 0.0, 0.4).astype(np.float32)

    # In LROC NAC, PSRs are pitch black (zero direct sunlight)
    optical_lroc = optical.copy()
    optical_lroc[psr_mask > 0.5] = 0.0

    # ShadowCam: ultra-sensitive secondary illumination from crater rims / earthshine inside PSRs
    secondary_light = gaussian_filter(illumination, 25) * 0.25
    shadowcam_opt = np.where(psr_mask > 0.5,
                             albedo * secondary_light * 4.0,
                             optical).astype(np.float32)

    # SAR: Backscatter (S1) and Circular Polarization Ratio (CPR)
    # Radar is active microwave: it penetrates total optical darkness unaffected by shadows!
    sar_look_azimuth = np.radians(225.0)
    sar_vec = np.array([np.cos(np.radians(45.0)) * np.cos(sar_look_azimuth),
                        np.cos(np.radians(45.0)) * np.sin(sar_look_azimuth),
                        np.sin(np.radians(45.0))])
    slope_sar = np.clip(np.sum(normals * sar_vec, axis=-1), 0.05, 1.0)

    # 4-look gamma speckle noise
    speckle = lambda: rng.gamma(4.0, 1.0 / 4.0, (n, n)).astype(np.float32)

    # S1 in linear power -> converted to typical lunar radar backscatter
    s1_lin = (0.01 + 0.15 * slope_sar + 0.1 * (cpr_base - 0.3)) * speckle()
    cpr = np.clip(cpr_base * speckle(), 0.05, 3.0).astype(np.float32)

    # Save rasters
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "sar_s1.npy"), s1_lin[None].astype(np.float32))
    np.save(os.path.join(a.out, "sar_cpr.npy"), cpr[None].astype(np.float32))
    np.save(os.path.join(a.out, "lroc_opt.npy"), optical_lroc[None].astype(np.float32))
    np.save(os.path.join(a.out, "shadowcam_opt.npy"), shadowcam_opt[None].astype(np.float32))
    np.save(os.path.join(a.out, "psr_mask.npy"), psr_mask[None].astype(np.float32))

    print(f"Generated synthetic lunar dataset with PSRs in '{a.out}':")
    print(f"  SAR S1:        shape {s1_lin[None].shape}, range [{s1_lin.min():.4f}, {s1_lin.max():.4f}]")
    print(f"  SAR CPR:       shape {cpr[None].shape}, range [{cpr.min():.4f}, {cpr.max():.4f}]")
    print(f"  LROC Optical:  shape {optical_lroc[None].shape}, range [{optical_lroc.min():.4f}, {optical_lroc.max():.4f}]")
    print(f"  ShadowCam Opt: shape {shadowcam_opt[None].shape}, range [{shadowcam_opt.min():.4f}, {shadowcam_opt.max():.4f}]")
    print(f"  PSR Mask:      shape {psr_mask[None].shape}, coverage {psr_mask.mean() * 100:.2f}%%")
    print(f"  SAR CPR: shape {cpr[None].shape}, range [{cpr.min():.4f}, {cpr.max():.4f}]")
    print(f"  Optical: shape {optical[None].shape}, range [{optical.min():.4f}, {optical.max():.4f}]")


if __name__ == "__main__":
    main()
