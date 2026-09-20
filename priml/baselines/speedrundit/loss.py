"""Flow-matching objective used by the upstream SpeedRunDiT recipe."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Literal

import math

from configgle import Fig
from torch import Tensor

import torch
from torch.nn import functional as F

from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.math.diffusion.schedule import log_snr_from_log_time_per_logit
from priml.math.diffusion.target import target_rectified_flow


def mean_flat(x: Tensor) -> Tensor:
    return x.flatten(1).mean(1)


class SpeedrunDiTLoss:
    """Compute denoising, projection, class, and CFM terms."""

    class Config(Fig["SpeedrunDiTLoss"]):
        path_type: Literal["linear", "cosine"] = "linear"
        weighting: Literal["uniform", "lognormal"] = "uniform"
        cfm_weighting: Literal["uniform", "linear"] = "uniform"
        apply_time_shift: bool = True
        shift_base: int = 4096
        projection_coeff: float = 0.5
        class_coeff: float = 0.03
        cfm_coeff: float = 0.05
        _: KW_ONLY

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, model: SpeedrunDiT, *, media: Tensor, label: Tensor, cls_token: Tensor, features: list[Tensor] | None = None, time: Tensor | None = None, noise: Tensor | None = None) -> dict[str, Tensor]:
        b = media.shape[0]
        if time is None:
            if self.config.weighting == "uniform":
                time = torch.rand((b,), device=media.device, dtype=media.dtype)
            else:
                sigma = torch.randn((b,), device=media.device, dtype=media.dtype).exp()
                time = sigma / (1 + sigma)
        if self.config.apply_time_shift:
            shift = math.sqrt(media[0].numel() / self.config.shift_base)
            time = (shift * time) / (1 + (shift - 1) * time)
            time = time.clamp(0, 1)
        noise = torch.randn_like(media) if noise is None else noise
        noise_cls = torch.randn_like(cls_token)
        alpha, sigma, d_alpha, d_sigma = _interpolant(time, self.config.path_type)
        alpha_x = alpha.reshape(b, 1, 1, 1)
        sigma_x = sigma.reshape(b, 1, 1, 1)
        d_alpha_x = d_alpha.reshape(b, 1, 1, 1)
        d_sigma_x = d_sigma.reshape(b, 1, 1, 1)
        x_t = alpha_x * media + sigma_x * noise
        cls_t = alpha[:, None] * cls_token + sigma[:, None] * noise_cls
        target = d_alpha_x * media + d_sigma_x * noise
        if self.config.path_type == "linear":
            log_sigma = time.clamp_min(torch.finfo(time.dtype).tiny).log().reshape(b, 1, 1, 1)
            log_snr = log_snr_from_log_time_per_logit(log_sigma)
            target = target_rectified_flow(
                torch.zeros_like(media), x_t, log_snr, log_sigma,
                x_original=media, eps_original=noise,
            ).target
            assert target is not None
        cls_target = d_alpha[:, None] * cls_token + d_sigma[:, None] * noise_cls
        output, projections, cls_output = model(x_t, time, label, cls_token=cls_t)
        result = {"denoising": mean_flat((output - target) ** 2), "class": mean_flat((cls_output - cls_target) ** 2)}
        projection = torch.zeros((), device=media.device, dtype=media.dtype)
        if features and projections:
            for expected, actual in zip(features, projections, strict=False):
                n = min(expected.shape[1], actual.shape[1])
                projection = projection - F.normalize(expected[:, :n].detach(), dim=-1).mul(F.normalize(actual[:, :n], dim=-1)).sum(-1).mean()
        cfm_target = target.roll(1, 0)
        cfm = -((output - cfm_target) ** 2)
        if self.config.cfm_weighting == "linear":
            cfm = cfm * time.reshape(b, *([1] * (cfm.ndim - 1)))
        result["projection"] = projection.reshape(1)
        result["cfm"] = cfm.flatten(1).mean(1)
        result["loss"] = result["denoising"] + self.config.class_coeff * result["class"] + self.config.projection_coeff * result["projection"] + self.config.cfm_coeff * result["cfm"]
        return result


def _interpolant(t: Tensor, path_type: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if path_type == "linear":
        one = torch.ones_like(t)
        return 1 - t, t, -one, one
    angle = t * math.pi / 2
    return torch.cos(angle), torch.sin(angle), -math.pi / 2 * torch.sin(angle), math.pi / 2 * torch.cos(angle)
