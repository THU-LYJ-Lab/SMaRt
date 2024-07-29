# python3.8
"""Contains the Gaussian Diffusion for EDM."""

__all__ = ['GaussianDiffusion']

import torch


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
    ):
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        step_indices = torch.arange(num_timesteps, dtype=torch.float64)
        self.t_steps = (sigma_max ** (1 / rho) 
                        + step_indices / (num_timesteps - 1) * 
                        (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps = torch.as_tensor(self.t_steps).flip(dims=[0])
        
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.rand_like(x_0)
        assert noise.shape == x_0.shape

        sigma = self.t_steps.to(device=t.device)[t].float()
        while len(sigma.shape) < len(x_0.shape):
            sigma = sigma[..., None]
        sigma = sigma.expand(x_0.shape)

        return x_0 + sigma * noise

    def __call__(self, net, images, t, labels=None, augment_pipe=None, return_logits=False):
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        sigma = self.t_steps.to(device=t.device)[t].float()
        while len(sigma.shape) < len(y.shape):
            sigma = sigma[..., None]
        sigma = sigma.expand(y.shape)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma[:,:1,:1,:1], labels, augment_labels=augment_labels)
        loss = (D_yn - y) ** 2
        if return_logits:
            return loss, D_yn, y
        return loss
