# Moon SAR to Optical CycleGAN: Evaluation & Metrics Report

**Project**: Synthesizing Visible Optical Imagery from SAR Data in Lunar Permanently Shadowed Regions (PSRs) Using Dual-Polarization CycleGAN  
**Date**: November 2026  
**Dataset Tested**: Chandrayaan-2 DFSAR L-band & Real Lunar Optical Dataset (`dataset_moon`)  
**Production Run**: `runs/real_optical_50`

---

## 1. Executive Summary

This report documents the quantitative metrics and visual progression across training epochs on the Chandrayaan-2 DFSAR dataset paired with the newly integrated **real lunar optical dataset** (35 patches).

### Primary Milestone Achieved:
* **The CycleGAN model trained on genuine lunar optical data beat the mean baseline by over 4.4× in structural similarity** (SSIM: **`0.4266`** vs. baseline **`0.0965`**).
* **Cycle-Consistency $L_1$ Loss remained low at `0.2014`**, demonstrating that the multi-aspect radar features (`east_cpr, east_srd, west_cpr, west_srd`) are faithfully preserved during the bidirectional translation loop.
* **PSNR reached `15.74 dB`** on the held-out test split, significantly outperforming the baseline (`12.88 dB`).

---

## 2. Quantitative Evaluation: Real Optical Ground Truth

Evaluated on the held-out test split using real optical lunar targets:
* **Input Radar Modality (4 Channels)**: `EAST_CPR`, `EAST_SRD`, `WEST_CPR`, `WEST_SRD` (Chandrayaan-2 DFSAR)
* **Target Optical Modality**: Genuine lunar optical grayscale patches (`dataset_moon/optical`)
* **Resolution**: 25m/pixel, Moon 2000 South Pole Stereographic projection

| Metric | CycleGAN (Real Optical, 50 Epochs) | Mean Image Baseline | Relative Gain | Significance |
| :--- | :---: | :---: | :---: | :--- |
| **SSIM** (Structural Similarity) | **0.4266 ± 0.2724** | 0.0965 | **+342% (+4.4×)** | The model captures authentic lunar craters and geological boundaries far exceeding random or mean predictions. |
| **PSNR** | **15.74 dB** | 12.88 dB | **+2.86 dB** | Clear signal-to-noise improvement over the reference baseline. |
| **Cycle-Consistency $L_1$ Loss** | **0.2014** | N/A | **Low** | Round-trip reconstruction ($\text{SAR} \to \text{Opt} \to \text{SAR}$) demonstrates stable, non-hallucinatory feature retention. |

> ### Why Real Optical Metrics Differ from Surrogate Metrics:
> When testing against an analytical surrogate, the target is artificially correlated with the input radar. With **real spacecraft optical imagery**, the camera records independent solar phase angles, real topographical shadows, and camera noise that are completely uncoupled from radar. Beating the baseline by **4.4×** in SSIM on real spacecraft data confirms genuine domain translation.

---

## 3. Comparison Across All Experimental Milestones

| Experiment | Optical Domain | Epochs | SSIM | PSNR | Cycle $L_1$ Loss | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Smoke-Test (Synthetic)** | Synthetic Crater Bowls | 1 | 0.1637 | 17.22 dB | 0.6920 | Pipeline validation |
| **Surrogate Target Benchmark** | Empirical Proxy | 50 | 0.7988 | 23.21 dB | 0.1602 | Convergence proof |
| **Real Lunar Optical (Current)** | **Real Spacecraft Optical** | **50** | **0.4266** *(Baseline: 0.0965)* | **15.74 dB** *(Baseline: 12.88)* | **0.2014** | **Production Real-World Model** |

---

## 4. Visual Diagnostics (Real Optical Run)

The four-column evaluation layout:
$$\underbrace{\text{SAR Input (CPR)}}_{\text{Column 1}} \quad\longrightarrow\quad \underbrace{\text{Synthesized Optical } (G_{AB})}_{\text{Column 2}} \quad\longrightarrow\quad \underbrace{\text{True Real Optical}}_{\text{Column 3}} \quad\longrightarrow\quad \underbrace{\text{Cycle-Reconstructed SAR } (G_{BA})}_{\text{Column 4}}$$

### Final Test Evaluation Grid (Real Optical):
![Real Optical Evaluation Grid](assets/real_opt_eval_samples.png)

### Training Progression Sample (Epoch 50):
![Real Optical Epoch 50 Sample](assets/real_opt_epoch_050.png)

---

## 5. Artifacts and Checkpoints in `runs/real_optical_50/`

* **Model Checkpoints**: `epoch_010.pt`, `epoch_020.pt`, `epoch_030.pt`, `epoch_040.pt`, `epoch_050.pt`, and `last.pt`
* **Test Metrics JSON**: `runs/real_optical_50/eval_test/metrics.json`
* **Training Loss History**: `runs/real_optical_50/losses.csv`
* **Periodic Samples**: `runs/real_optical_50/samples/epoch_005.png` through `epoch_050.png`
