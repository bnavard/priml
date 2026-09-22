"""Euler--Maruyama sampling: the grid, the shapes, and determinism."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.speedrundit.loss import cosine_path
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.sampler import EulerMaruyamaSampler


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


def test_the_grid_descends_from_one_to_zero() -> None:
    """Integration runs backwards in time and lands exactly on zero.

    Stopping short of zero would leave the sample still noisy; overshooting
    would divide by a vanished sigma.
    """
    grid = sampler().times((CHANNELS, GRID, GRID), torch.device("cpu"))
    assert grid.shape == (5,)
    assert grid[0].item() == pytest.approx(1.0)
    assert grid[-1].item() == 0.0
    assert torch.all(grid[:-1] >= grid[1:])


def test_the_time_shift_moves_the_grid() -> None:
    """The sampler shifts time the same way the objective trained it to.

    A sampler integrating an unshifted grid would be walking a curve the
    model was never fit to.
    """
    shape = (32, 16, 16)
    shifted = sampler().times(shape, torch.device("cpu"))
    plain = sampler(time_shift_base=None).times(shape, torch.device("cpu"))
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


def test_the_path_is_injected() -> None:
    """The sampler integrates whatever path it is handed."""
    assert sampler(interpolant=cosine_path).config.interpolant is cosine_path


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
