"""Generation: integrate the learned velocity from noise back to data.

The step is :func:`priml.math.diffusion.ddpm_ddim` at its own defaults -- this
model predicts ``v = eps - x``, which is what ``target_rectified_flow``
decomposes, and rectified flow is already that function's partner corruption.
``eta`` chooses between a deterministic DDIM trajectory and the full DDPM
posterior.

What is written here is the loop, because this model carries two coupled
streams: the latent and the class token share a time grid and each feeds the
other's velocity, so they advance together. Time is carried as ``log_snr`` and
handed back to the model as ``t`` through the schedule's stated inverse.

References:
  https://arxiv.org/abs/2010.02502
    Song et al. 2020, "Denoising Diffusion Implicit Models" (the eta family).
  https://arxiv.org/abs/2401.08740
    Ma et al. 2024, "SiT", Section 4 on SDE sampling.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, NamedTuple

import torch

from configgle import Fig

from priml.baselines.speedrundit.loss import resolution_time_shift
from priml.math.diffusion import (
    TargetFn,
    ddpm_ddim,
    log_sigma_from_log_snr_per_rectified_flow,
    log_snr_from_log_time_per_logit,
    log_time_from_log_snr_per_logit,
    target_rectified_flow,
)


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.speedrundit.model import SpeedrunDiT
    from priml.math.custom_types import TensorableFn


__all__ = ["EulerMaruyamaSampler"]


class EulerMaruyamaSampler:
    """Integrates the learned velocity from noise to data."""

    class Output(NamedTuple):
        """One sampling run."""

        media: Tensor
        """Generated latents, ``[batch, channels, size, size]``."""

        cls_token: Tensor
        """Generated class features, ``[batch, channels_cls]``."""

    class Config(Fig["EulerMaruyamaSampler"]):
        """Configuration for EulerMaruyamaSampler."""

        _: KW_ONLY

        num_steps: int = 250
        """Integration steps; the reported FID is measured at this count."""

        last_time: float = 0.04
        """Time the grid stops at. Zero is unreachable: ``log_snr`` there is
        infinite, and the drift is stiffest in the last steps anyway."""

        eta: float = 1.0
        """Stochasticity: zero is DDIM, one is the DDPM posterior."""

        target_fn: TargetFn = target_rectified_flow
        """Decomposes the model output into clean signal and noise.

        Must agree with what the model was trained to predict; this one reads
        a velocity, which is what the objective's default path targets."""

        corruption_fn: TensorableFn = log_sigma_from_log_snr_per_rectified_flow
        """Maps log SNR to log sigma; the path's own corruption rule."""

        time_shift_base: int | None = 4096
        """Element count at which the time shift is the identity; ``None``
        disables shifting. Must match the objective's, or the sampler walks a
        curve the model was never fit to."""

        guidance: float = 1.0
        """Classifier-free guidance strength; one disables the second pass."""

        @property
        def uses_guidance(self) -> bool:
            """Whether an unconditional pass is needed.

            Returns:
              needed: True when ``guidance`` is not exactly one.

            """
            return self.guidance != 1.0

    def __init__(self, config: Config) -> None:
        self.config = config

    def log_snr(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Build the descending log-SNR grid.

        Args:
          shape: Shape of one sample, without the batch axis.
          device: Device the grid is built on.

        Returns:
          log_snr: ``[num_steps + 1]``, increasing as time descends.

        """
        cfg = self.config
        times = torch.linspace(
            1.0,
            cfg.last_time,
            cfg.num_steps,
            dtype=torch.float64,
            device=device,
        )
        if cfg.time_shift_base is not None:
            times = resolution_time_shift(
                times,
                shape=shape,
                base=cfg.time_shift_base,
            )
        return log_snr_from_log_time_per_logit(times.log())

    @torch.no_grad()
    def __call__(
        self,
        model: SpeedrunDiT,
        media: Tensor,
        label: Tensor,
        cls_token: Tensor,
    ) -> Output:
        """Integrate from noise to data.

        Args:
          model: The trained velocity field.
          media: Initial latent noise, ``[batch, channels, size, size]``.
          label: Class indices to condition on, ``[batch]``.
          cls_token: Initial class-token noise, ``[batch, channels_cls]``.

        Returns:
          output: The integrated latent and class token.

        """
        grid = self.log_snr(tuple(media.shape[1:]), media.device)
        latent, cls = media, cls_token
        steps = grid.shape[0] - 1

        for index in range(steps):
            curr, following = grid[index], grid[index + 1]
            time = self._times(curr, latent.shape[0], latent.dtype, latent.device)
            velocity, velocity_cls = self._velocity(model, latent, time, label, cls)
            # The final step takes the posterior mean and adds no noise, as
            # Ho et al. prescribe; every earlier one is stochastic.
            noisy = index < steps - 1
            latent = self._advance(velocity, latent, curr, following, noisy=noisy)
            cls = self._advance(velocity_cls, cls, curr, following, noisy=noisy)
        return EulerMaruyamaSampler.Output(media=latent, cls_token=cls)

    def _advance(
        self,
        velocity: Tensor,
        state: Tensor,
        log_snr_curr: Tensor,
        log_snr_next: Tensor,
        *,
        noisy: bool,
    ) -> Tensor:
        """Take one shared diffusion step.

        Args:
          velocity: The model's prediction for this stream.
          state: Current noisy state.
          log_snr_curr: Log SNR now.
          log_snr_next: Log SNR after the step.
          noisy: Whether to add the step's sampling noise.

        Returns:
          state: The advanced state.

        """
        cfg = self.config
        _, mean, log_std = ddpm_ddim(
            velocity,
            state,
            log_snr_curr,
            log_snr_next,
            corruption_fn=cfg.corruption_fn,
            target_fn=cfg.target_fn,
            eta=cfg.eta,
        )
        if not noisy:
            return mean
        return mean + torch.exp(log_std) * torch.randn_like(state)

    def _times(
        self,
        log_snr: Tensor,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Recover the flow time the model was conditioned on."""
        time = log_time_from_log_snr_per_logit(log_snr).exp()
        return torch.full((batch,), float(time), dtype=dtype, device=device)

    def _velocity(
        self,
        model: SpeedrunDiT,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate both velocities, optionally with guidance.

        Args:
          model: The trained velocity field.
          media: Current latents.
          time: Current times.
          label: Class indices.
          cls_token: Current class token.

        Returns:
          velocity: Latent velocity.
          velocity_cls: Class-token velocity.

        """
        conditional = model(media, time, label, cls_token)
        if not self.config.uses_guidance:
            return conditional.velocity, conditional.cls_velocity
        # Two forwards, not one doubled batch: the unconditional branch also
        # discards the sparse path, so the two take different routes.
        unconditional = model(media, time, label, cls_token, uncond=True)
        weight = self.config.guidance
        return (
            torch.lerp(unconditional.velocity, conditional.velocity, weight),
            torch.lerp(unconditional.cls_velocity, conditional.cls_velocity, weight),
        )
