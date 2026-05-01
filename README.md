# 🧠 DDPM — Diffusion Model for High-Resolution Face Generation

> **Denoising Diffusion Probabilistic Model** trained on CelebA-HQ 256×256 from scratch using pure PyTorch. Dual-GPU optimised for Kaggle T4×2.

---

## 📌 Overview

This project implements a complete **DDPM pipeline** — forward diffusion, reverse denoising, image generation, and reconstruction — without relying on any pretrained diffusion library. Every component is built using base PyTorch.

The model learns to generate photorealistic 256×256 human faces by training on the [CelebA-HQ dataset](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only), progressively denoising images from pure Gaussian noise.
<img width="1357" height="281" alt="image" src="https://github.com/user-attachments/assets/37e42303-2e21-4c3f-ac13-015fb442450c" />
<img width="1262" height="433" alt="image" src="https://github.com/user-attachments/assets/dc20e9ec-81c4-4d32-ac3b-3a9088b24151" />

---

## ✨ Features

- **Full DDPM pipeline** — forward noising + learned reverse denoising
- **Custom U-Net** with GroupNorm, ResBlocks, and bottleneck self-attention
- **EMA (Exponential Moving Average)** model for sharper, more stable samples
- **Mixed Precision Training** (AMP) with GradScaler for memory efficiency
- **Dual GPU** via `nn.DataParallel` — fully utilises Kaggle T4×2
- **Cosine LR Schedule** for smooth convergence
- **Checkpoint resume** — safe to resume from any interrupted session
- **Image reconstruction** from partially-noised target images
- **Quantitative evaluation** — PSNR & SSIM over 5 images
- **Per-epoch visual previews** of generated samples

---

## 🗂️ Project Structure

```
ddpm_celebahq.ipynb        ← Main Kaggle notebook (all-in-one)
checkpoints/
  ckpt_latest.pt           ← Latest checkpoint (resumes automatically)
  ckpt_epoch_010.pt        ← Milestone saves (every 10 epochs)
  ckpt_epoch_020.pt
  losses.csv               ← Epoch-wise loss + LR log
  generated/               ← 5 generated output images
  target.png               ← Reconstruction target
  reconstructed.png        ← Reconstruction output
```

---

## 🏗️ Architecture

### U-Net Backbone

```
Input (3 × 256 × 256)
      │
   Conv 3×3 → 64 ch
      │
  [Encoder]
  ResBlock ×2 → 64ch  →  Downsample (stride-2 conv)
  ResBlock ×2 → 128ch →  Downsample
  ResBlock ×2 → 256ch →  Downsample
      │
  [Bottleneck]
  ResBlock → Self-Attention → ResBlock   (at 32×32 spatial)
      │
  [Decoder]
  Upsample → ResBlock ×2 → 256ch  (+ skip from encoder)
  Upsample → ResBlock ×2 → 128ch  (+ skip)
  Upsample → ResBlock ×2 → 64ch   (+ skip)
      │
  GroupNorm → SiLU → Conv 3×3
      │
Output (3 × 256 × 256)  ← predicted noise ε
```

**Channel progression:** 64 → 128 → 256 (as required)

**Time conditioning:** Sinusoidal embeddings projected through a 2-layer MLP, injected into every ResBlock via AdaGN (scale + shift on GroupNorm output).

### Why GroupNorm over BatchNorm?

BatchNorm normalises across the batch. In DDPM the batch contains images at *different* noise levels — normalising across them corrupts the signal. GroupNorm normalises within each sample, making it the correct choice for diffusion.

---

## 📐 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image size | 256 × 256 | Assignment recommended |
| Timesteps T | 400 | Quality/speed balance (200–500 range) |
| β start | 1e-4 | Standard DDPM |
| β end | 2e-2 | Standard DDPM |
| Batch size | 16 | 8 per GPU, AMP enabled |
| Learning rate | 2e-4 | Standard AdamW for diffusion |
| Weight decay | 1e-4 | L2 regularisation |
| EMA β | 0.999 | Smooth parameter tracking |
| Epochs | 25 | Sufficient convergence on CelebA-HQ |
| Grad clip | 1.0 | Prevents gradient explosion |
| LR schedule | Cosine | Smooth decay |

---

## 🔢 Diffusion Mathematics

### Forward Process  q(x_t | x_0)

```
x_t = √ᾱ_t · x₀  +  √(1 - ᾱ_t) · ε       where  ε ~ N(0, I)
```

The **cumulative product** `ᾱ_t = ∏ αₛ` lets us jump to any noise level in one step, without iterating through all intermediate timesteps.

### Reverse Process  p_θ(x_{t-1} | x_t)

The model predicts the noise `ε_θ(x_t, t)`, from which we recover:

```
x₀_pred = (x_t - √(1 - ᾱ_t) · ε_θ) / √ᾱ_t

x_{t-1} = posterior_mean(x₀_pred, x_t)  +  σ_t · z
```

### Training Objective

```
L = E_{x₀, ε, t} [ || ε  -  ε_θ(x_t, t) ||² ]
```

Simple MSE between predicted and actual noise.

---

## 🚀 Running on Kaggle

### Requirements

The notebook runs entirely within the Kaggle environment. No local installation needed.

```
Platform : Kaggle Notebooks
Accelerator : GPU T4 × 2
Dataset : CelebA-HQ 256 (denislukovnikov/celebahq256-images-only)
Runtime : Python 3.10, PyTorch ≥ 2.0
```

### Dataset Path

```python
DATA_DIR = '/kaggle/input/celebahq256-images-only/data256x256'
```

### Steps

1. Create a new Kaggle notebook
2. Add the CelebA-HQ dataset from the dataset panel
3. Set accelerator to **GPU T4 × 2**
4. Paste / upload `ddpm_celebahq.ipynb`
5. Run all cells in order — sections are independent and clearly labelled

---

## 📊 Notebook Sections

| Section | Description |
|---------|-------------|
| 1 | Environment setup, GPU detection |
| 2 | Hyperparameters (single place to edit) |
| 3 | Dataset, dataloader, 6 sample images |
| 4 | Beta schedule, forward diffusion, 8-step visualisation + SNR plot |
| 5 | U-Net architecture, param count, sanity forward pass |
| 6 | Training loop — AMP, EMA, cosine LR, CSV logging, per-epoch preview |
| 7 | DDPM sampler with intermediate step collection |
| 8 | 5 generated images from pure noise |
| 9 | 5 reverse diffusion step visualisations |
| 10 | Image reconstruction — partial noising + reverse |
| 11 | PSNR & SSIM evaluation |
| 12 | Loss + LR curve from CSV |

---

## 📈 Results

### Training

The model trains stably with monotonically decreasing MSE loss. Per-epoch image previews show progression from noise toward coherent faces.

### Image Generation

5 diverse faces generated entirely from Gaussian noise using the full 400-step reverse process. The EMA model is used at inference for improved sharpness.
<img width="713" height="720" alt="image" src="https://github.com/user-attachments/assets/71451a3d-98dc-4edb-b6ef-f48d8562ede4" />
<img width="621" height="620" alt="image" src="https://github.com/user-attachments/assets/ddd05d73-10c3-4de3-83e2-39767f948fcd" />
<img width="311" height="311" alt="image" src="https://github.com/user-attachments/assets/d9907bb9-9ab2-4640-a291-0b6027d0e4d4" />

### Image Reconstruction

A target face is noised to `t = 300` (preserving low-frequency structure), then reverse-diffused back. The output resembles the target identity.
<img width="1345" height="463" alt="image" src="https://github.com/user-attachments/assets/27eca8b6-0936-46a3-8fb4-402cb6934a7d" />

### Quantitative Metrics

| Metric | Score |
|--------|-------|
| PSNR | ~22–26 dB |
| SSIM | ~0.65–0.80 |

Scores vary with training duration and noise level used for reconstruction (`t = 250–300` recommended).

---

## 🔍 Key Design Decisions

**Why self-attention at bottleneck only?**
At 32×32 spatial resolution (after 3 downsamples from 256×256), attention over 1024 positions is computationally affordable. Adding attention at higher resolutions would increase memory quadratically with no meaningful gain for faces at this scale.

**Why partial noising for reconstruction?**
Noising to `T-1` (full noise) produces a sample completely disconnected from the target — it's just generation. Noising to `t = 300` preserves enough low-frequency face structure for the reverse process to reconstruct something recognisable.

**Why no gradient accumulation?**
AMP halves memory, DataParallel doubles throughput. Combined, batch=16 fits comfortably without the complexity of accumulation.

---

## 📚 References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Karras et al. (2019) — [Progressive Growing of GANs / CelebA-HQ](https://arxiv.org/abs/1710.10196)
- Ronneberger et al. (2015) — [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## 👤 Author

Built as part of a Deep Learning assignment on generative modelling.
Platform: Kaggle | Dataset: CelebA-HQ 256×256 | Framework: PyTorch
