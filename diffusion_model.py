"""Self-contained PyTorch Denoising Diffusion Probabilistic Model (DDPM) & Refiner.

Designed for Lunar Optical Synthesis and Hybrid CycleGAN-Diffusion Refinement:
  1. DDPMScheduler: Linear/cosine noise scheduling, forward noising, and reverse sampling.
  2. DiffusionUNet: Time-conditioned U-Net with sinusoidal embeddings, residual blocks,
     and concatenation conditioning (SAR radar or CycleGAN prior).
  3. SDEdit Refiner: Stochastic differential editing that refines CycleGAN structural
     priors by injecting calibrated noise and reverse-denoising high-frequency micro-textures.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal time-step embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Convolutional block conditioned on time embeddings."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Inject time embedding
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_proj

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual_conv(x)


class DiffusionUNet(nn.Module):
    """U-Net architecture for noise prediction in diffusion process."""
    def __init__(self, in_channels=1, out_channels=1, cond_channels=0,
                 base_dim=32, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        total_in_ch = in_channels + cond_channels

        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        dims = [base_dim * m for m in dim_mults]

        # Initial projection
        self.init_conv = nn.Conv2d(total_in_ch, dims[0], 3, padding=1)

        # Downsampling path
        self.down1 = ResidualBlock(dims[0], dims[0], time_dim)
        self.pool1 = nn.Conv2d(dims[0], dims[1], 4, stride=2, padding=1)

        self.down2 = ResidualBlock(dims[1], dims[1], time_dim)
        self.pool2 = nn.Conv2d(dims[1], dims[2], 4, stride=2, padding=1)

        self.down3 = ResidualBlock(dims[2], dims[2], time_dim)
        self.pool3 = nn.Conv2d(dims[2], dims[3], 4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = ResidualBlock(dims[3], dims[3], time_dim)
        self.mid2 = ResidualBlock(dims[3], dims[3], time_dim)

        # Upsampling path
        self.up1 = nn.ConvTranspose2d(dims[3], dims[2], 4, stride=2, padding=1)
        self.upblock1 = ResidualBlock(dims[2] + dims[2], dims[2], time_dim)

        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 4, stride=2, padding=1)
        self.upblock2 = ResidualBlock(dims[1] + dims[1], dims[1], time_dim)

        self.up3 = nn.ConvTranspose2d(dims[1], dims[0], 4, stride=2, padding=1)
        self.upblock3 = ResidualBlock(dims[0] + dims[0], dims[0], time_dim)

        # Final projection to noise prediction
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, dims[0]), dims[0]),
            nn.SiLU(),
            nn.Conv2d(dims[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        t_emb = self.time_mlp(t)

        x0 = self.init_conv(x)
        d1 = self.down1(x0, t_emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        d3 = self.down3(p2, t_emb)
        p3 = self.pool3(d3)

        m = self.mid1(p3, t_emb)
        m = self.mid2(m, t_emb)

        u1 = self.up1(m)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.upblock1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.upblock2(u2, t_emb)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.upblock3(u3, t_emb)

        return self.final_conv(u3)


class DDPMScheduler:
    """Discrete Denoising Diffusion Probabilistic Model scheduler."""
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 device=torch.device("cpu")):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute diffusion calculations
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, x_start, noise, t):
        """Forward diffusion step: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps"""
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def step(self, model_output, t_idx, sample):
        """Single reverse diffusion step from x_t to x_{t-1}."""
        beta_t = self.betas[t_idx]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_idx]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_idx]

        # Predicted mean mu_t
        pred_mean = sqrt_recip_alpha_t * (
            sample - (beta_t / sqrt_one_minus_alpha_bar_t) * model_output
        )

        if t_idx == 0:
            return pred_mean
        else:
            var = self.posterior_variance[t_idx]
            noise = torch.randn_like(sample)
            return pred_mean + torch.sqrt(torch.clamp(var, min=1e-20)) * noise

    @torch.no_grad()
    def sample_full(self, model, shape, cond=None, num_steps=None):
        """Full reverse diffusion sampling from pure noise at t = T down to t = 0."""
        model.eval()
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        steps = list(reversed(range(num_steps or self.num_timesteps)))
        for t_val in steps:
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            eps_pred = model(img, t, cond=cond)
            img = self.step(eps_pred, t_val, img)

        return torch.clamp(img, -1.0, 1.0)

    @torch.no_grad()
    def sample_sdedit(self, model, x_init, t_refine=150, cond=None):
        """SDEdit / Hybrid Refiner:
        1. Takes CycleGAN output x_init.
        2. Injects calibrated Gaussian noise up to intermediate timestep t_refine.
        3. Reverse-denoises from t_refine -> 0 to enhance high-frequency textures
           while preserving CycleGAN's macro crater geometry.
        """
        model.eval()
        device = self.device
        b = x_init.shape[0]

        # 1. Forward noising up to t_refine
        t_start_tensor = torch.full((b,), t_refine, device=device, dtype=torch.long)
        noise = torch.randn_like(x_init)
        x_t = self.add_noise(x_init, noise, t_start_tensor)

        # 2. Reverse denoising steps from t_refine down to 0
        for t_val in reversed(range(t_refine + 1)):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            eps_pred = model(x_t, t, cond=cond)
            x_t = self.step(eps_pred, t_val, x_t)

        return torch.clamp(x_t, -1.0, 1.0)
