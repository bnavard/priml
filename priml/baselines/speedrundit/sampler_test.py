"""Euler--Maruyama sampling: the grid, the guidance routes, and determinism.

Bit-for-bit agreement with the reference's own sampler needs its clone, so it
lives in ``scripts/parity.py``; these pin the contracts it rests on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.speedrundit.loss import cosine_path, linear_path
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.sampler import (
    EulerMaruyamaSampler,
    velocity_to_score,
)
from priml.lib.custom_json import ListCodec


if TYPE_CHECKING:
    from torch import Tensor


pytestmark = pytest.mark.compute_training

GRID: Final = 4
CHANNELS: Final = 8
TARGET: Final = 16
BATCH: Final = 2
CLASSES: Final = 10


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
    cfg.num_classes = CLASSES
    cfg.projector_dims = (TARGET,)
    cfg.projector_hidden = 32
    return cfg.make().eval()


def noise() -> tuple[Tensor, Tensor, Tensor]:
    """Draw the sampler's starting state.

    Returns:
      media: Latent noise.
      label: Class indices.
      cls_token: Class-token noise.

    """
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(BATCH, CHANNELS, GRID, GRID, generator=generator),
        torch.randint(CLASSES, (BATCH,), generator=generator),
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


def recording(
    model: SpeedrunDiT,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[int], bool]]:
    """Record the label and route of every forward the sampler makes.

    Args:
      model: The model to spy on; its forward is wrapped in place.
      monkeypatch: Undoes the wrap when the test ends.

    Returns:
      calls: ``(labels, uncond)`` per forward, appended as they happen.

    """
    calls: list[tuple[list[int], bool]] = []
    forward = model.forward

    def spy(
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
        *,
        uncond: bool = False,
    ) -> SpeedrunDiT.Output:
        calls.append((ListCodec.coerce(label.tolist(), int), uncond))
        return forward(media, time, label, cls_token, uncond=uncond)

    monkeypatch.setattr(model, "forward", spy)
    return calls


def test_the_grid_descends_from_one_to_exactly_zero() -> None:
    """``num_steps`` evaluations need ``num_steps + 1`` points.

    The last one is zero: stopping at ``last_time`` instead would return a
    sample still carrying that much noise.
    """
    times = sampler(time_transform=None).times((CHANNELS, GRID, GRID))
    assert times.shape == (5,)
    assert times.dtype == torch.float64
    assert times[0].item() == 1.0
    assert times[-2].item() == pytest.approx(0.04)
    assert times[-1].item() == 0.0
    assert torch.all(times[1:] < times[:-1])


def test_the_time_shift_moves_the_grid_and_keeps_its_ends() -> None:
    """The sampler shifts time the way the objective trained it to.

    The shift fixes both endpoints, so the grid still starts at pure noise
    and still ends on the data.
    """
    shape = (32, 16, 16)
    shifted = sampler().times(shape)
    plain = sampler(time_transform=None).times(shape)
    assert not torch.equal(shifted, plain)
    assert (shifted[0].item(), shifted[-1].item()) == (1.0, 0.0)


def test_the_score_is_recovered_from_the_velocity_on_both_paths() -> None:
    """Pinned to the closed form: at the true velocity the score is -eps/sigma."""
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(64, generator=generator)
    eps = torch.randn(64, generator=generator)
    t = torch.rand(64, generator=generator) * 0.9 + 0.05
    for path in (linear_path, cosine_path):
        c = path(t)
        state = c.alpha * x + c.sigma * eps
        velocity = c.d_alpha * x + c.d_sigma * eps
        score = velocity_to_score(velocity, state, path, t)
        assert torch.allclose(score, -eps / c.sigma, atol=1e-4), path.__name__


def test_sampling_returns_the_shapes_it_was_given() -> None:
    """Generation is a fixed point in shape and dtype, whatever the steps."""
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


def test_the_final_step_adds_no_noise() -> None:
    """One step is only the deterministic landing, so no seed can move it."""
    model = tiny_model()
    media, label, cls_token = noise()
    torch.manual_seed(1)
    first = sampler(num_steps=1)(model, media, label, cls_token)
    torch.manual_seed(2)
    second = sampler(num_steps=1)(model, media, label, cls_token)
    assert torch.equal(first.media, second.media)


def test_sampling_leaves_the_model_untouched() -> None:
    """Generation must not train, and must not flip the model into training."""
    model = tiny_model()
    media, label, cls_token = noise()
    before = [p.detach().clone() for p in model.parameters()]
    _ = sampler()(model, media, label, cls_token)
    assert not model.training
    for left, right in zip(before, model.parameters(), strict=True):
        assert torch.equal(left, right)


def test_a_training_model_is_refused() -> None:
    """In training the model drops labels and routes tokens at random."""
    model = tiny_model().train()
    media, label, cls_token = noise()
    with pytest.raises(ValueError, match="eval mode"):
        _ = sampler()(model, media, label, cls_token)


def test_unguided_sampling_is_one_conditional_pass_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guidance at one needs no weak branch."""
    model = tiny_model()
    calls = recording(model, monkeypatch)
    media, label, cls_token = noise()
    _ = sampler()(model, media, label, cls_token)
    assert calls == [(ListCodec.coerce(label.tolist(), int), False)] * 4


def test_the_weak_branch_is_the_null_class_with_the_path_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path-drop guidance changes BOTH the label and the route.

    Dropping only the path keeps the class conditioning in the weak branch,
    which is not the unconditional model the guidance extrapolates from.
    """
    model = tiny_model()
    calls = recording(model, monkeypatch)
    media, label, cls_token = noise()
    _ = sampler(guidance=2.5)(model, media, label, cls_token)
    strong, weak = (
        (ListCodec.coerce(label.tolist(), int), False),
        ([CLASSES] * BATCH, True),
    )
    assert calls == [strong, weak] * 4


def test_guidance_applies_only_inside_its_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside the interval a step takes the conditional pass alone."""
    model = tiny_model()
    calls = recording(model, monkeypatch)
    media, label, cls_token = noise()
    guided = sampler(guidance=2.5, guidance_interval=(0.0, 0.5), time_transform=None)
    _ = guided(model, media, label, cls_token)
    times = guided.times((CHANNELS, GRID, GRID))[:-1]
    expected: list[tuple[list[int], bool]] = []
    strong = ListCodec.coerce(label.tolist(), int)
    for t in ListCodec.coerce(times.tolist(), float):
        expected.append((strong, False))
        if t <= 0.5:
            expected.append(([CLASSES] * BATCH, True))
    assert calls == expected
    assert 0 < sum(uncond for _, uncond in calls) < len(times)


def test_guidance_changes_the_sample() -> None:
    """A strength above one has to reach the trajectory."""
    model = tiny_model()
    # Zeroed readouts make both branches predict zero, hiding the guidance.
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    media, label, cls_token = noise()
    torch.manual_seed(9)
    plain = sampler()(model, media, label, cls_token)
    torch.manual_seed(9)
    guided = sampler(guidance=2.5)(model, media, label, cls_token)
    assert not torch.equal(plain.media, guided.media)


def test_guidance_without_a_null_class_is_refused() -> None:
    """A model built with no label dropout has no null row to guide from."""
    cfg = SpeedrunDiT.Config()
    cfg.channels_in, cfg.channels_hidden, cfg.image_size = CHANNELS, 64, GRID
    cfg.num_layers, cfg.heads, cfg.num_classes = 5, 4, CLASSES
    cfg.projector_dims, cfg.projector_hidden = (TARGET,), 32
    cfg.label_embedder.dropout = 0.0
    model = cfg.make().eval()
    media, label, cls_token = noise()
    with pytest.raises(ValueError, match="null class"):
        _ = sampler(guidance=2.5)(model, media, label, cls_token)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
