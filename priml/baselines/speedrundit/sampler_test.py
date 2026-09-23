"""Euler--Maruyama sampling: the grid, the shapes, and determinism."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.sampler import EulerMaruyamaSampler
from priml.math.diffusion import target_rectified_flow, target_v


pytestmark = pytest.mark.compute_training

GRID: Final = 4
CHANNELS: Final = 8
TARGET: Final = 16
BATCH: Final = 2


def tiny_model() -> SpeedrunDiT:
    """Build a small model to integrate.

    Returns:
      model: A shrunk SR-DiT in eval mode.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = CHANNELS
    cfg.channels_hidden = 64
    cfg.image_size = GRID
    cfg.num_layers = 5
    cfg.heads = 4
    cfg.num_classes = 10
    cfg.projector_dims = (TARGET,)
    cfg.projector_hidden = 32
    return cfg.make().eval()


def noise() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the sampler's starting state.

    Returns:
      media: Latent noise.
      label: Class indices.
      cls_token: Class-token noise.

    """
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(BATCH, CHANNELS, GRID, GRID, generator=generator),
        torch.randint(10, (BATCH,), generator=generator),
        torch.randn(BATCH, TARGET, generator=generator),
    )


def sampler(**overrides: object) -> EulerMaruyamaSampler:
    """Build a short sampler.

    Args:
      **overrides: Config fields to set.

    Returns:
      sampler: A built sampler.

    """
    config = EulerMaruyamaSampler.Config()
    config.num_steps = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_log_snr_rises_as_time_descends() -> None:
    """Denoising means increasing signal-to-noise, monotonically.

    ``ddpm_ddim_step`` computes ``log1mexp(log_snr_curr - log_snr_next)``,
    which is NaN unless the grid increases; a reversed grid fails here rather
    than producing silent NaNs a hundred steps in.
    """
    grid = sampler().log_snr((CHANNELS, GRID, GRID), torch.device("cpu"))
    assert grid.shape == (4,)
    assert torch.all(grid[1:] > grid[:-1])
    assert torch.all(torch.isfinite(grid))


def test_the_grid_stops_short_of_zero() -> None:
    """``log_snr`` at ``t = 0`` is infinite, so the grid must not reach it."""
    grid = sampler(last_time=0.04).log_snr(
        (CHANNELS, GRID, GRID),
        torch.device("cpu"),
    )
    assert torch.all(torch.isfinite(grid))


def test_the_time_shift_moves_the_grid() -> None:
    """The sampler shifts time the same way the objective trained it to.

    A sampler integrating an unshifted grid would be walking a curve the
    model was never fit to.
    """
    shape = (32, 16, 16)
    shifted = sampler().log_snr(shape, torch.device("cpu"))
    plain = sampler(time_shift_base=None).log_snr(shape, torch.device("cpu"))
    assert not torch.equal(shifted, plain)


def test_sampling_returns_the_shapes_it_was_given() -> None:
    """Generation is a fixed point in shape, whatever the step count."""
    model = tiny_model()
    media, label, cls_token = noise()
    out = sampler()(model, media, label, cls_token)
    assert out.media.shape == media.shape
    assert out.cls_token.shape == cls_token.shape
    assert out.media.dtype == media.dtype


def test_sampling_is_reproducible_under_a_seed() -> None:
    """Two runs on one seed must agree; the SDE draws noise every step."""
    model = tiny_model()
    media, label, cls_token = noise()
    torch.manual_seed(5)
    first = sampler()(model, media, label, cls_token)
    torch.manual_seed(5)
    second = sampler()(model, media, label, cls_token)
    assert torch.equal(first.media, second.media)
    assert torch.equal(first.cls_token, second.cls_token)


def test_different_seeds_give_different_samples() -> None:
    """The stochastic term has to actually be stochastic."""
    model = tiny_model()
    media, label, cls_token = noise()
    torch.manual_seed(1)
    first = sampler()(model, media, label, cls_token)
    torch.manual_seed(2)
    second = sampler()(model, media, label, cls_token)
    assert not torch.equal(first.media, second.media)


def test_sampling_leaves_the_model_untouched() -> None:
    """Generation must not train, and must not flip the model into training."""
    model = tiny_model()
    media, label, cls_token = noise()
    before = [p.detach().clone() for p in model.parameters()]
    _ = sampler()(model, media, label, cls_token)
    assert not model.training
    for left, right in zip(before, model.parameters(), strict=True):
        assert torch.equal(left, right)


def test_guidance_runs_a_second_pass() -> None:
    """A guidance strength other than one changes the trajectory.

    The unconditional branch also drops the sparse path, so the two forwards
    take different routes and cannot be batched into one.
    """
    model = tiny_model()
    media, label, cls_token = noise()
    torch.manual_seed(9)
    plain = sampler()(model, media, label, cls_token)
    torch.manual_seed(9)
    guided = sampler(guidance=2.5)(model, media, label, cls_token)
    assert not torch.equal(plain.media, guided.media)


def test_the_decomposition_defaults_to_the_shared_rectified_flow() -> None:
    """The model predicts a velocity, which is what this function reads.

    Pinned because the objective and the sampler have to agree on what the
    output MEANS: a sampler decomposing a velocity as if it were an epsilon
    prediction produces plausible noise and no error.
    """
    assert sampler().config.target_fn is target_rectified_flow


def test_the_decomposition_is_injected() -> None:
    """A model trained to predict something else swaps this, not the loop."""
    assert sampler(target_fn=target_v).config.target_fn is target_v


def test_eta_selects_between_ddim_and_ddpm() -> None:
    """Zero is deterministic, so two runs agree without seeding."""
    model = tiny_model()
    media, label, cls_token = noise()
    first = sampler(eta=0.0)(model, media, label, cls_token)
    second = sampler(eta=0.0)(model, media, label, cls_token)
    assert torch.equal(first.media, second.media)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
