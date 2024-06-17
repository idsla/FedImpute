from typing import Tuple

import torch


class DiffImpute(torch.nn.Module):

    def __init__(self, timesteps=1000):
        super(DiffImpute, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps)

        # define alphas
        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def compute_loss(self, inputs: Tuple[torch.Tensor, ...]):
        pass

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        if t[0] != 0:
            return (
                    sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
            ).to(self.device)
        else:
            return (sqrt_alphas_cumprod_t * x_start).to(self.device)

    def undo(self, x_out, t):
        betas_t = extract(self.betas, t, x_out.shape)
        x_in_est = torch.sqrt(1 - betas_t) * x_out + torch.sqrt(betas_t) * torch.randn_like(
            x_out
        )
        return x_in_est

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * apply_model(model, x, time=t).squeeze(1)
                / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def apply_model(model, x_num, x_cat=None, time=None):
    if isinstance(model, denoise_models.FTTransformer):
        return model(x_num, x_cat, time)
    else:
        assert x_cat is None
        return model(x_num, time)
