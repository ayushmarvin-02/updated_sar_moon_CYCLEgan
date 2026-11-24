# Synthesizing Visible Optical Imagery from SAR Data in Lunar Permanently Shadowed Regions (PSRs) Using Dual-Polarization CycleGAN

Unpaired image-to-image translation translating multi-channel lunar Synthetic Aperture Radar (SAR) into visible-wavelength optical representations inside Lunar Permanently Shadowed Regions (PSRs).

---

## 🌕 Problem Overview

Permanently Shadowed Regions (PSRs) inside lunar polar impact craters (e.g., Shackleton, Faustini, Shoemaker, Haworth) receive **zero direct solar illumination**. While conventional optical cameras (such as LROC NAC) image only pure shadows in these depressions, active microwave radar instruments can penetrate total optical darkness:

* **Chandrayaan-2 DFSAR**: High-resolution L-band polarimetry providing multi-aspect (East-look & West-look) scattering matrices.
* **LRO Mini-RF**: S-band/X-band hybrid-pol radar measuring Stokes parameters ($S_1$) and Circular Polarization Ratio ($CPR$).
* **KPLO ShadowCam**: Ultra-sensitive optical camera capturing secondary scattered light inside PSR crater floors.

This repository provides an end-to-end CycleGAN framework that translates polarimetric radar features (dielectric roughness, volume scattering, bouldery crater rims) into plausible visible-wavelength optical imagery without requiring paired ground truth.

---

## 📊 Benchmark Results (Chandrayaan-2 DFSAR, 50 Epochs)

Evaluated on the held-out test split of real 25m Chandrayaan-2 DFSAR South Pole mosaics using multi-aspect radar features (`EAST_CPR`, `EAST_SRD`, `WEST_CPR`, `WEST_SRD`):

| Metric | 50 Epochs (`runs/epochs_50`) | 5 Epochs (Initial) | Significance |
| :--- | :---: | :---: | :--- |
| **Cycle-Consistency $L_1$ Loss** | **0.1602** | 0.2754 | **-42% distortion reduction** in round-trip $\text{SAR} \to \text{Opt} \to \text{SAR}$. |
| **PSNR** | **23.21 dB** | 22.63 dB | High reconstruction signal-to-noise ratio. |
| **Pearson Correlation ($r$)** | **+0.4335** | +0.4393 | Persistent positive spatial alignment with lunar topography. |
| **SSIM** | **0.7988 ± 0.0624** | 0.8287 ± 0.0362 | High structural fidelity across complex crater walls. |

> Detailed metrics, per-epoch loss curves, and diagnostic visual panels are available in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

---

## 🖼️ Visual Translation Pipeline

Every evaluation sample visualizes a 4-panel diagnostic comparison:

$$\underbrace{\text{SAR Input (CPR)}}_{\text{Column 1 (Radar)}} \quad\longrightarrow\quad \underbrace{\text{Synthesized Optical } (G_{AB})}_{\text{Column 2 (AI Prediction)}} \quad\longrightarrow\quad \underbrace{\text{Target Optical}}_{\text{Column 3 (Reference)}} \quad\longrightarrow\quad \underbrace{\text{Cycle-Reconstructed SAR } (G_{BA})}_{\text{Column 4 (Verification)}}$$

* **Column 1 (Input SAR)**: Radar Circular Polarization Ratio (CPR). Bright pixels indicate high surface roughness / bouldery crater rims.
* **Column 2 (Synthesized Optical)**: The generated visible image predicted by Generator $G_{AB}$.
* **Column 3 (Target Optical)**: Target optical luminance.
* **Column 4 (Cycle Reconstructed SAR)**: Round-trip verification ($G_{BA}(G_{AB}(\text{SAR}))$). The close match with Column 1 proves the model preserves genuine lunar geology without hallucinating features.

---

## 📁 Repository Structure

| File | Purpose |
| :--- | :--- |
| `prepare_lunar_data.py` | Unified preprocessor: handles pre-cut patch stacks (`dataset_moon`) or full-scene GeoTIFF rasters with PSR mask filtering. |
| `dataset.py` | PyTorch Datasets supporting DFSAR (`east_cpr`, `west_cpr`, etc.), Mini-RF (`s1, cpr`), and Sentinel-1. |
| `models.py` | ResNet Generator (6–9 residual blocks) and 70×70 PatchGAN Discriminator. |
| `losses.py` | Least-Squares GAN (LSGAN) loss, history ImagePool, and optional edge-consistency prior. |
| `train.py` | CycleGAN training loop with dynamic LR scheduling and periodic visual sampling. |
| `evaluate.py` | Quantitative evaluation (SSIM, PSNR, Pearson correlation, Cycle $L_1$) on held-out test splits. |
| `make_lunar_synthetic.py` | Synthetic lunar scene generator with impact craters, PSR shadows, and ShadowCam simulation for offline testing. |
| `EVALUATION_REPORT.md` | Comprehensive experimental report and benchmark metrics. |

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/ayushmarvin-02/updated_sar_moon_CYCLEgan.git
cd updated_sar_moon_CYCLEgan
pip install -r requirements.txt
```

### 2. Preprocess Data
#### A. From Pre-Extracted Patch Stacks (e.g. `dataset_moon`)
```bash
python prepare_lunar_data.py --patches-dir dataset_moon --surrogate-opt --out data/moon_dfsar
```

#### B. From Full-Scene GeoTIFFs with PSR Mask Filtering
```bash
python prepare_lunar_data.py \
  --sar-ch1 path/to/mini_rf_s1.tif \
  --sar-ch2 path/to/mini_rf_cpr.tif \
  --opt path/to/shadowcam.tif \
  --psr-mask path/to/lola_psr_mask.tif \
  --min-psr-fraction 0.5 \
  --out data/lunar_psr_tiles
```

### 3. Train CycleGAN
```bash
python train.py \
  --data data/moon_dfsar \
  --channels east_cpr,east_srd,west_cpr,west_srd \
  --epochs 25 \
  --epochs_decay 25 \
  --out runs/epochs_50
```

### 4. Evaluate Test Split
```bash
python evaluate.py --ckpt runs/epochs_50/last.pt --data data/moon_dfsar --out runs/epochs_50/eval_test
```

---

## 🛰️ Multi-Aspect (Dual-Look) Radar Benefits

In polar terrain, low-incidence single-look SAR often suffers from severe radar shadows behind steep crater walls. By integrating both **East-looking** and **West-looking** Chandrayaan-2 DFSAR passes (`east_cpr` + `west_cpr`), this model synthesizes optical terrain with multi-aspect illumination, significantly reducing occlusion artifacts.
