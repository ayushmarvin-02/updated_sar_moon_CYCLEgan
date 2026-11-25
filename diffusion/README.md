# 🌀 Lunar Diffusion & Hybrid Synthesis Module

This directory contains the standalone **Denoising Diffusion Probabilistic Model (DDPM)** and the **Tandem Hybrid Pipeline (CycleGAN + Diffusion SDEdit Refinement)** for lunar Permanently Shadowed Region (PSR) optical synthesis.

It is kept separate from the core CycleGAN code to keep the repository modular.

---

## 📁 Directory Structure

| File | Description |
| :--- | :--- |
| `diffusion_model.py` | Standalone PyTorch DDPM implementation: time-conditioned U-Net, sinusoidal embeddings, residual blocks, and linear/cosine `DDPMScheduler`. |
| `train_diffusion.py` | Training script for training diffusion models on lunar optical regolith patches. |
| `hybrid_pipeline.py` | Tandem CycleGAN + Diffusion inference & evaluation script using SDEdit stochastic refinement. |

---

## 🚀 Usage

### 1. Train the Diffusion Model
```bash
python diffusion/train_diffusion.py --data data/moon_dfsar --out runs/diffusion_model --epochs 120
```

### 2. Run the Hybrid Benchmark
Compares Pure CycleGAN, Pure Diffusion, and the Hybrid combination on held-out lunar test tiles:
```bash
python diffusion/hybrid_pipeline.py \
  --cyclegan runs/real_optical_100/last.pt \
  --diffusion runs/diffusion_model/diffusion_last.pt \
  --data data/moon_dfsar \
  --out runs/hybrid_benchmark
```

### 3. Programmatic Usage (Python)
```python
from diffusion import HybridLunarSynthesizer

synth = HybridLunarSynthesizer(
    cyclegan_ckpt="runs/real_optical_100/last.pt",
    diffusion_ckpt="runs/diffusion_model/diffusion_last.pt"
)

# Run hybrid synthesis (CycleGAN prior + SDEdit diffusion refinement)
hybrid_opt, cyclegan_opt, latency_ms = synth.predict_hybrid(sar_tensor, t_refine=70)
```
