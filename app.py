import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import torchvision.transforms as T
import io

# ==========================================
# 1. Configuration & Setup
# ==========================================
st.set_page_config(page_title="Diffusion App", layout="wide")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
IMG_SIZE = 128
TIMESTEPS = 400

# ==========================================
# 2. Diffusion Math & Schedules
# ==========================================
@st.cache_data
def get_schedules():
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = alpha_bar.sqrt()
    sqrt_1mab = (1.0 - alpha_bar).sqrt()
    alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])
    posterior_var = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
    return betas, alphas, alpha_bar, sqrt_ab, sqrt_1mab, alpha_bar_prev, posterior_var

betas, alphas, alpha_bar, sqrt_ab, sqrt_1mab, alpha_bar_prev, posterior_var = get_schedules()

def q_sample(x0, t, noise=None):
    """Forward process to add noise to an image."""
    if noise is None:
        noise = torch.randn_like(x0)
    t_cpu = t.cpu()
    s = sqrt_ab[t_cpu].view(-1, 1, 1, 1).to(x0.device)
    s1 = sqrt_1mab[t_cpu].view(-1, 1, 1, 1).to(x0.device)
    return s * x0 + s1 * noise, noise

# ==========================================
# 3. Model Architecture (Exact Match)
# ==========================================
def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        sc, sh = self.t_proj(t_emb).chunk(2, dim=1)
        h = h * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, -1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1)
        attn = torch.softmax((q.transpose(1, 2) @ k) * (C ** -0.5), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)

class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        c1, c2, c3 = base, base * 2, base * 4

        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))
        self.init_conv = nn.Conv2d(in_ch, c1, 3, padding=1)

        self.d1a, self.d1b = ResBlock(c1, c1, t_dim), ResBlock(c1, c1, t_dim)
        self.ds1 = nn.Conv2d(c1, c1, 4, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock(c1, c2, t_dim), ResBlock(c2, c2, t_dim)
        self.ds2 = nn.Conv2d(c2, c2, 4, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock(c2, c3, t_dim), ResBlock(c3, c3, t_dim)
        self.ds3 = nn.Conv2d(c3, c3, 4, stride=2, padding=1)

        self.mid1 = ResBlock(c3, c3, t_dim)
        self.attn = Attention(c3)
        self.mid2 = ResBlock(c3, c3, t_dim)

        self.us3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c3, c3, 3, padding=1))
        self.u3a, self.u3b = ResBlock(c3 + c3, c3, t_dim), ResBlock(c3, c3, t_dim)
        self.us2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c3, c2, 3, padding=1))
        self.u2a, self.u2b = ResBlock(c2 + c2, c2, t_dim), ResBlock(c2, c2, t_dim)
        self.us1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c2, c1, 3, padding=1))
        self.u1a, self.u1b = ResBlock(c1 + c1, c1, t_dim), ResBlock(c1, c1, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, in_ch, 3, padding=1))

    def forward(self, x, t):
        te = self.t_mlp(sinusoidal_emb(t, self.t_dim))
        x = self.init_conv(x)
        x1 = self.d1b(self.d1a(x, te), te)
        x2 = self.d2b(self.d2a(self.ds1(x1), te), te)
        x3 = self.d3b(self.d3a(self.ds2(x2), te), te)
        xm = self.mid2(self.attn(self.mid1(self.ds3(x3), te)), te)
        xu3 = self.u3b(self.u3a(torch.cat([self.us3(xm), x3], 1), te), te)
        xu2 = self.u2b(self.u2a(torch.cat([self.us2(xu3), x2], 1), te), te)
        xu1 = self.u1b(self.u1a(torch.cat([self.us1(xu2), x1], 1), te), te)
        return self.out(xu1)

# ==========================================
# 4. Helpers & Loading
# ==========================================
@st.cache_resource
def load_model():
    model = UNet(in_ch=3, base=64, t_dim=256)
    # Ensure map_location is set so it works on machines without GPUs
    weights = torch.load('ema_weights_only.pt', map_location=DEVICE)
    clean_weights = {}
    for key, value in weights.items():
        if key.startswith('module.'):
            clean_weights[key[7:]] = value  # [7:] removes the first 7 characters ("module.")
        else:
            clean_weights[key] = value
    model.load_state_dict(clean_weights)
    model.to(DEVICE)
    model.eval()
    return model

def to_pil(t):
    """Converts a [-1, 1] tensor to a PIL Image"""
    img_np = ((t.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def preprocess_image(image):
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    return tfm(image.convert('RGB')).unsqueeze(0).to(DEVICE)

# ==========================================
# 5. Core Inference Loops (With Streamlit Progress)
# ==========================================
@torch.no_grad()
def generate_images(model, n=1, pbar=None):
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    
    for i, t_val in enumerate(reversed(range(TIMESTEPS))):
        t_b = torch.full((n,), t_val, device=DEVICE, dtype=torch.long)
        
        # Note: autocast removed to ensure CPU compatibility, add back if strict GPU
        eps = model(x, t_b) 

        a = alphas[t_val].to(DEVICE)
        ab = alpha_bar[t_val].to(DEVICE)
        b = betas[t_val].to(DEVICE)

        x0p = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        c1 = (alpha_bar_prev[t_val].sqrt() * b) / (1 - ab)
        c2 = (a.sqrt() * (1 - alpha_bar_prev[t_val])) / (1 - ab)
        x = c1 * x0p + c2 * x

        if t_val > 0:
            x = x + posterior_var[t_val].to(DEVICE).sqrt() * torch.randn_like(x)
            
        if pbar: pbar.progress((i + 1) / TIMESTEPS)
        
    return [to_pil(img) for img in x]

@torch.no_grad()
def reconstruct_image(model, target_tensor, noise_t=300, pbar=None):
    x, _ = q_sample(target_tensor, torch.tensor([noise_t]))
    x = x.to(DEVICE)
    noisy_pil = to_pil(x) # Save intermediate noisy image for display

    for i, t_val in enumerate(reversed(range(noise_t))):
        t_b = torch.full((1,), t_val, device=DEVICE, dtype=torch.long)
        eps = model(x, t_b)

        a = alphas[t_val].to(DEVICE)
        ab = alpha_bar[t_val].to(DEVICE)
        b = betas[t_val].to(DEVICE)
        
        x0p = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        c1 = (alpha_bar_prev[t_val].sqrt() * b) / (1 - ab)
        c2 = (a.sqrt() * (1 - alpha_bar_prev[t_val])) / (1 - ab)
        x = c1 * x0p + c2 * x
        
        if t_val > 0:
            x = x + posterior_var[t_val].to(DEVICE).sqrt() * torch.randn_like(x)
            
        if pbar: pbar.progress((i + 1) / noise_t)
        
    return noisy_pil, to_pil(x)

# ==========================================
# 6. Streamlit UI
# ==========================================
st.title("DDPM Face Generator & Reconstructor")
model = load_model()

# Sidebar Navigation
mode = st.sidebar.radio("Select Task:", ["Generate from Noise", "Reconstruct Image"])

if mode == "Generate from Noise":
    st.header("Generate High-Resolution Faces")
    cols = st.columns([1, 3])
    with cols[0]:
        num_images = st.slider("Number of Images", 1, 4, 1)
        generate_btn = st.button("Generate", type="primary")
        
    if generate_btn:
        progress_text = st.empty()
        pbar = st.progress(0)
        progress_text.text("Denoising from pure noise...")
        
        # Generate
        images = generate_images(model, n=num_images, pbar=pbar)
        
        # Display
        progress_text.text("Done!")
        img_cols = st.columns(num_images)
        for idx, img in enumerate(images):
            img_cols[idx].image(img, use_container_width=True)

elif mode == "Reconstruct Image":
    st.header("Reconstruct Face from Target")
    uploaded_file = st.file_uploader("Upload a face image (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file)
        target_tensor = preprocess_image(raw_img)
        
        noise_level = st.slider("Noise Timestep (T)", min_value=50, max_value=399, value=300, step=10)
        
        if st.button("Reconstruct", type="primary"):
            progress_text = st.empty()
            pbar = st.progress(0)
            progress_text.text(f"Applying noise to T={noise_level} and reconstructing...")
            
            # Reconstruct
            noisy_img, recon_img = reconstruct_image(model, target_tensor, noise_t=noise_level, pbar=pbar)
            
            # Display
            progress_text.text("Done!")
            col1, col2, col3 = st.columns(3)
            col1.image(raw_img.resize((IMG_SIZE, IMG_SIZE)), caption="Original Target", use_container_width=True)
            col2.image(noisy_img, caption=f"Noised (T={noise_level})", use_container_width=True)
            col3.image(recon_img, caption="Reconstructed Output", use_container_width=True)