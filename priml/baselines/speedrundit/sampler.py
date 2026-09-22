"""Euler--Maruyama sampling of the SR-DiT velocity field.

On not using ``priml.math.diffusion.sample``
    That driver carries ONE state tensor: ``sample_iter`` threads a single
    ``x_curr`` through ``model_fn`` and ``onestep_fn``. This model diffuses two
    coupled streams -- the latent and the class token -- which share a time
    grid and have to advance together, because the class token is an input to
    the velocity the latent reads. Packing them into one tensor would mean
    reshaping at every step and unpacking inside the model, which buys nothing
    and hides the coupling. So the loop is written here, and the shared
    driver stays the right answer for the single-stream case it serves.

    What IS reused is the path itself: the drift below is expressed through
    the same :class:`~priml.baselines.speedrundit.loss.InterpolantFn` the
    objective trains against, so a change of path moves both together instead
    of leaving the sampler integrating a curve the model was never fit to.

References:
  https://arxiv.org/abs/2401.08740
    Ma et al. 2024, "SiT: Exploring Flow and Diffusion-based Generative
    Models with Scalable Interpolant Transformers", Section 4 on SDE sampling.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, NamedTuple

import torch

from configgle import Fig

from priml.baselines.speedrundit.loss import (
    InterpolantFn,
    linear_path,
    resolution_time_shift,
)


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.speedrundit.model import SpeedrunDiT


__all__ = ["EulerMaruyamaSampler"]


class EulerMaruyamaSampler:
    """Integrates the learned velocity backwards from noise to data."""

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

        last_step: float = 0.04
        """Time the uniform grid stops at before the final jump to zero.

        Integrating all the way to zero on a uniform grid spends its last and
        largest steps where the drift is stiffest; stopping short and taking
        one exact step to zero is what keeps the tail stable."""

        interpolant: InterpolantFn = linear_path
        """Probability path; must match the one the model was trained on."""

        time_shift_base: int | None = 4096
        """Element count at which the time shift is the identity; ``None``
        disables shifting."""

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

    def times(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Build the descending time grid.

        Args:
          shape: Shape of one sample, without the batch axis.
          device: Device the grid is built on.

        Returns:
          times: ``[num_steps + 1]`` float64 times from one down to zero.

        """
        cfg = self.config
        grid = torch.linspace(
            1.0,
            cfg.last_step,
            cfg.num_steps,
            dtype=torch.float64,
            device=device,
        )
        grid = torch.cat([grid, grid.new_zeros(1)])
        if cfg.time_shift_base is not None:
            grid = resolution_time_shift(grid, shape=shape, base=cfg.time_shift_base)
        return grid

    @torch.no_grad()
    def __call__(
        self,
        model: SpeedrunDiT,
        media: Tensor,
        label: Tensor,
        cls_token: Tensor,
    ) -> Output:
        """Integrate from noise to data.

        The state is carried in float64 while the model is called in its own
        dtype: the accumulation runs for hundreds of steps and is the caller's,
        so it is stated here rather than inherited from whatever the model
        happens to emit.

        Args:
          model: The trained velocity field.
          media: Initial latent noise, ``[batch, channels, size, size]``.
          label: Class indices to condition on, ``[batch]``.
          cls_token: Initial class-token noise, ``[batch, channels_cls]``.

        Returns:
          output: The integrated latent and class token, in ``media``'s dtype.

        """
        cfg = self.config
        dtype = media.dtype
        grid = self.times(tuple(media.shape[1:]), media.device)
        state = media.double()
        state_cls = cls_token.double()

        for current, following in zip(grid[:-1], grid[1:], strict=True):
            step = following - current
            time = torch.full(
                (state.shape[0],),
                float(current),
                device=state.device,
                dtype=dtype,
            )
            velocity, velocity_cls = self._velocity(
                model,
                state.to(dtype),
                time,
                label,
                state_cls.to(dtype),
            )
            # Reverse-time diffusion coefficient of the variance-exploding SDE
            # that shares this path's marginals.
            diffusion = 2 * current.clamp_min(0)
            drift = velocity.double() - 0.5 * diffusion * self._score(
                state,
                velocity.double(),
                current,
            )
            drift_cls = velocity_cls.double() - 0.5 * diffusion * self._score(
                state_cls,
                velocity_cls.double(),
                current,
            )
            state = state + drift * step
            state_cls = state_cls + drift_cls * step
            if following > 0:
                scale = (diffusion * step.abs()).sqrt()
                state = state + scale * torch.randn_like(state)
                state_cls = state_cls + scale * torch.randn_like(state_cls)

        return EulerMaruyamaSampler.Output(
            media=state.to(dtype),
            cls_token=state_cls.to(dtype),
        )

    def _velocity(
        self,
        model: SpeedrunDiT,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the velocity, optionally with guidance.

        Guidance runs two separate forwards rather than one doubled batch,
        because the unconditional branch also drops SPRINT's sparse path and
        the two therefore take different routes through the model.

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
        unconditional = model(media, time, label, cls_token, uncond=True)
        strength = self.config.guidance
        blend = lambda cond, uncond: uncond + strength * (cond - uncond)  # noqa: E731
        return (
            blend(conditional.velocity, unconditional.velocity),
            blend(conditional.cls_velocity, unconditional.cls_velocity),
        )

    def _score(self, state: Tensor, velocity: Tensor, time: Tensor) -> Tensor:
        """Recover the score from a velocity prediction.

        Args:
          state: Current state.
          velocity: Predicted velocity at ``time``.
          time: Current time.

        Returns:
          score: The gradient of the log density at ``state``.

        """
        path = self.config.interpolant(time)
        alpha = torch.as_tensor(path.alpha, dtype=state.dtype, device=state.device)
        sigma = torch.as_tensor(path.sigma, dtype=state.dtype, device=state.device)
        d_alpha = torch.as_tensor(path.d_alpha, dtype=state.dtype, device=state.device)
        d_sigma = torch.as_tensor(path.d_sigma, dtype=state.dtype, device=state.device)
        ratio = d_alpha / alpha
        variance = sigma**2 * ratio - d_sigma * sigma
        return (ratio * state - velocity) / variance
