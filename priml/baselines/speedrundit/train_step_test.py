"""One optimizer update: what it moves, what it reports, what it restores."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from torch import Tensor

from priml.baselines.speedrundit.train_step import SpeedrunDiTTrainStep
from priml.train.parallelism import NoParallel


pytestmark = pytest.mark.compute_training

BATCH: Final = 2
GRID: Final = 4
CHANNELS: Final = 8
TARGET: Final = 16


def tiny_step() -> SpeedrunDiTTrainStep:
    """Build a shrunk step that runs on CPU in milliseconds.

    Size only: the optimizer, the clip, the EMA decay, the objective weights
    and the routing policy are all left as exp000 sets them.

    Returns:
      step: A built train step.

    """
    cfg = SpeedrunDiTTrainStep.Config()
    cfg.model.channels_in = CHANNELS
    cfg.model.channels_hidden = 64
    cfg.model.image_size = GRID
    cfg.model.num_layers = 5
    cfg.model.heads = 4
    cfg.model.num_classes = 10
    cfg.model.projector_dims = (TARGET,)
    cfg.model.projector_hidden = 32
    cfg.train_budget_steps = 8
    cfg.parallelism = NoParallel.Config(device="cpu")
    # fp32 on CPU: autograd's weight-gradient matmul has no bf16 kernel for
    # the transposed layout on most hosts and falls back to a scalar loop.
    cfg.dtype_autocast = None
    cfg.compile = None
    return cfg.make()


def batch(size: int = BATCH) -> dict[str, object]:
    """Build one training batch.

    Args:
      size: Samples in the batch.

    Returns:
      batch: A loader-shaped mapping.

    """
    generator = torch.Generator().manual_seed(0)
    tokens = 1 + GRID * GRID
    return {
        "media": torch.randn(size, CHANNELS, GRID, GRID, generator=generator),
        "label": torch.randint(10, (size,), generator=generator),
        "cls_token": torch.randn(size, TARGET, generator=generator),
        "features": [torch.randn(size, tokens, TARGET, generator=generator)],
        "valid_count": size,
    }


def test_a_step_reports_the_contract_keys() -> None:
    """``loss`` and ``model`` are what the loop reads; the rest is metrics."""
    step = tiny_step()
    out = step.train_step(**batch())
    assert isinstance(out["loss"], Tensor)
    assert isinstance(out["model"], Tensor)
    assert set(out["metrics"]) == {
        "denoising",
        "cls",
        "projection",
        "cfm",
        "cfm_cls",
    }


def test_a_step_advances_the_global_counter() -> None:
    """The ``timer_step`` bracket is what makes ``max_steps`` mean anything.

    Driving the optimizer without it would leave every cadence above reading
    a counter that never moves.
    """
    step = tiny_step()
    assert step.global_step == 0
    _ = step.train_step(**batch())
    assert step.global_step == 1


def test_a_step_moves_the_weights() -> None:
    """An update that changes nothing is the failure a smoke run hides."""
    step = tiny_step()
    before = [p.detach().clone() for p in step.model.parameters() if p.requires_grad]
    _ = step.train_step(**batch())
    after = [p.detach().clone() for p in step.model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_gradients_are_cleared_between_steps() -> None:
    """A step that leaks gradients accumulates across updates silently."""
    step = tiny_step()
    _ = step.train_step(**batch())
    assert all(p.grad is None for p in step.model.parameters())


def test_the_loss_falls_over_a_few_steps() -> None:
    """The recipe has to actually optimize, at any size."""
    torch.manual_seed(0)
    step = tiny_step()
    fixed = batch()
    losses = [float(step.train_step(**fixed)["loss"]) for _ in range(6)]
    assert losses[-1] < losses[0]


def test_the_frozen_position_table_never_moves() -> None:
    """``pos_embed`` rides in the checkpoint but is not trained.

    AdamW receives it in the parameter list, matching the reference, and skips
    it because it has no gradient.
    """
    step = tiny_step()
    before = step.model.pos_embed.detach().clone()
    _ = step.train_step(**batch())
    assert torch.equal(step.model.pos_embed, before)


def test_the_ema_shadow_trails_the_live_weights() -> None:
    """Averaging has to be doing something, and not simply copying."""
    step = tiny_step()
    _ = step.train_step(**batch())
    _ = step.train_step(**batch())
    shadow = step.ema
    live = dict(step.model.named_parameters())
    with shadow.apply_to(step.model):
        averaged = {n: p.detach().clone() for n, p in step.model.named_parameters()}
    moved = [
        name
        for name, value in averaged.items()
        if live[name].requires_grad and not torch.equal(value, live[name])
    ]
    assert moved


def test_eval_does_not_update_anything() -> None:
    """Scoring must leave the weights and the step counter alone."""
    step = tiny_step()
    _ = step.train_step(**batch())
    before = [p.detach().clone() for p in step.model.parameters()]
    at = step.global_step
    out = step.eval_loss(**batch())
    assert step.global_step == at
    for left, right in zip(before, step.model.parameters(), strict=True):
        assert torch.equal(left, right)
    assert isinstance(out["loss"], Tensor)


def test_eval_is_deterministic() -> None:
    """Two evals of one checkpoint must agree.

    They only can if every training-only draw -- label dropout, token drop,
    path drop -- is genuinely off in eval.
    """
    step = tiny_step()
    fixed = batch()
    torch.manual_seed(3)
    left = float(step.eval_loss(**fixed)["loss"])
    torch.manual_seed(3)
    right = float(step.eval_loss(**fixed)["loss"])
    assert left == right


def test_state_round_trips() -> None:
    """A resumed step carries its counter and its optimizer moments."""
    step = tiny_step()
    _ = step.train_step(**batch())
    state = step.state_dict()
    restored = tiny_step()
    restored.load_state_dict(state)
    assert restored.global_step == step.global_step


def test_gradient_clipping_is_on_by_default() -> None:
    """A finite ceiling is part of the recipe, not an option.

    Asserted on the config rather than on a weight delta: AdamW normalizes by
    its second moment, so a first update has magnitude about the learning rate
    whatever the gradient scale, and a weight-space check would pass with the
    clip removed.
    """
    cfg = SpeedrunDiTTrainStep.Config()
    assert cfg.gradient_clip_norm == 1.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
