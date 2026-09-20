"""Euler--Maruyama sampling for latent velocity fields."""

from __future__ import annotations

from dataclasses import field
from typing import Callable

from configgle import Fig, Makeable
from torch import Tensor

import torch


def apply_time_shift(t: Tensor, latent_dim: int, shift_base: int = 4096) -> Tensor:
    shift = (latent_dim / shift_base) ** 0.5
    return ((shift * t) / (1 + (shift - 1) * t)).clamp(0, 1)


class EulerMaruyamaSampler:
    class Config(Fig["EulerMaruyamaSampler"]):
        num_steps: int = 50
        cfg_scale: float = 1.0
        cls_cfg_scale: float = 1.0
        time_shifting: bool = True
        shift_base: int = 4096

    def __init__(self, config: Config) -> None:
        self.config = config

    @torch.no_grad()
    def __call__(self, model: Callable[..., tuple[Tensor, list[Tensor], Tensor]], latents: Tensor, labels: Tensor, cls_latents: Tensor) -> Tensor:
        cfg = self.config
        times = torch.linspace(1, 0, cfg.num_steps + 1, device=latents.device, dtype=torch.float64)
        if cfg.time_shifting:
            times = apply_time_shift(times, latents[0].numel(), cfg.shift_base)
        x, cls_x = latents.double(), cls_latents.double()
        for current, nxt in zip(times[:-1], times[1:], strict=True):
            dt = nxt - current
            t = torch.full((x.shape[0],), current, device=x.device, dtype=latents.dtype)
            velocity, _, cls_velocity = model(x.to(latents.dtype), t, labels, cls_token=cls_x.to(latents.dtype), return_projections=False)
            x = x + velocity.double() * dt
            cls_x = cls_x + cls_velocity.double() * dt
            if nxt > 0:
                noise = torch.randn_like(x)
                cls_noise = torch.randn_like(cls_x)
                diffusion = 2 * current.clamp_min(0)
                x = x + diffusion.sqrt() * noise * (-dt).sqrt()
                cls_x = cls_x + diffusion.sqrt() * cls_noise * (-dt).sqrt()
        return x.to(latents.dtype)
