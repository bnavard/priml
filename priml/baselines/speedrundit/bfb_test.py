"""Bit-for-bit goldens for SR-DiT.

These freeze what ``scripts/parity.py`` established against the pinned
reference. The parity script needs a network and a clone and runs once; these
run on CPU in the ordinary suite and are what catch a regression afterwards.

Three goldens, because they fail for different reasons. ``init`` freezes the
construction order -- reorder two submodules and the RNG stream shifts, and
every weight moves. ``forward`` freezes the op order through routing, rotary
positions, and the value residual. ``five_steps`` freezes the recipe: the
objective, the drawn times and noise, the gradients, the clip, and AdamW's
moments all reach the compared post-state.

Regenerate with ``BFB_REGENERATE=1``; a missing golden is minted AND fails,
which is what forces someone to read it first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from torch import Tensor, nn

from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.testing.bfb import assert_bfb_against_golden


_CWD: Final = Path(__file__).parent.resolve()

GRID: Final = 4
CHANNELS: Final = 8
HIDDEN: Final = 64
HEADS: Final = 4
LAYERS: Final = 6
CLASSES: Final = 10
TARGET: Final = 16
BATCH: Final = 2
STEPS: Final = 5


def miniature() -> SpeedrunDiT.Config:
    """Shrink the recipe without replacing its numerical choices.

    Cut: width, depth, heads, the latent grid, the class count, the projector
    widths. Kept: the routing ratios, the path-drop probability, the value
    residual, the rotary positions, the qk norms, the zeroed modulation, and
    every initialization rule. Four heads rather than two because at batch two
    a head axis of two would be indistinguishable from the batch axis, and a
    transpose between them would not show up here.

    Returns:
      cfg: The golden geometry.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = CHANNELS
    cfg.channels_hidden = HIDDEN
    cfg.image_size = GRID
    cfg.patch_size = 1
    cfg.num_layers = LAYERS
    cfg.heads = HEADS
    cfg.num_classes = CLASSES
    cfg.projector_dims = (TARGET,)
    cfg.projector_hidden = 32
    return cfg


def build_input() -> dict[str, Tensor]:
    """Draw the fixed inputs every golden replays against.

    Returns:
      batch: Latents, times, labels, and class features.

    """
    return {
        "media": torch.randn(BATCH, CHANNELS, GRID, GRID),
        "time": torch.rand(BATCH),
        "label": torch.randint(CLASSES, (BATCH,)),
        "cls_token": torch.randn(BATCH, TARGET),
    }


class _Init(nn.Module):
    """Reports the initialized state as a tensor the harness can compare."""

    def __init__(self, model: SpeedrunDiT) -> None:
        super().__init__()
        self.inner = model

    def forward(self, _: Tensor) -> Tensor:
        """Concatenate every parameter, in construction order.

        Args:
          _: Unused; the harness always passes an input.

        Returns:
          state: One float32 vector of every parameter.

        """
        return torch.cat(
            [p.detach().float().flatten() for p in self.inner.parameters()],
        )


class _Forward(nn.Module):
    """Runs one forward and reports every output as one vector."""

    def __init__(self, model: SpeedrunDiT) -> None:
        super().__init__()
        self.inner = model

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the model in eval mode and flatten its three outputs.

        Eval mode deliberately: routing, label dropout and path drop are all
        stochastic in training, and a golden over them would freeze a draw
        rather than an arithmetic.

        Args:
          batch: Fixed inputs.

        Returns:
          output: Velocity, projections, and class velocity, concatenated.

        """
        self.inner.eval()
        out = self.inner(
            batch["media"],
            batch["time"],
            batch["label"],
            batch["cls_token"],
        )
        parts = [out.velocity.float().flatten(), out.cls_velocity.float().flatten()]
        parts.extend(p.float().flatten() for p in out.projections)
        return torch.cat(parts)


class _FiveSteps(nn.Module):
    """Runs the real recipe for five updates and reports the trajectory.

    Wrapping rather than passing the model directly is what lets the optimizer
    be built AFTER the harness randomizes parameters, against those very
    tensors; an optimizer built earlier would hold a discarded model's.
    """

    def __init__(self, model: SpeedrunDiT) -> None:
        super().__init__()
        self.inner = model

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Take five updates and concatenate what each produced.

        Args:
          batch: Fixed inputs, reused every step.

        Returns:
          trajectory: Losses, gradient norms, and the final weights.

        """
        objective = SpeedrunDiTLoss.Config().make()
        optimizer = torch.optim.AdamW(
            self.inner.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )
        self.inner.train()
        torch.manual_seed(4242)
        pieces: list[Tensor] = []
        for _ in range(STEPS):
            result = objective(
                self.inner,
                media=batch["media"],
                label=batch["label"],
                cls_token=batch["cls_token"],
                features=[batch["features"]],
            )
            result.loss.backward()
            norm = nn.utils.clip_grad_norm_(self.inner.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pieces.append(result.loss.detach().float().reshape(1))
            pieces.append(result.time.detach().float().flatten())
            pieces.append(norm.detach().float().reshape(1))
        pieces.extend(p.detach().float().flatten() for p in self.inner.parameters())
        return torch.cat(pieces)


def test_initialization_bfb() -> None:
    """Freeze the construction order and every initialization rule."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speedrundit_init",
        build_module=lambda: _Init(miniature().make()),
        build_input=lambda: torch.zeros(1),
        seed=0,
    )


def test_forward_bfb() -> None:
    """Freeze the op order of one eval forward."""
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speedrundit_forward",
        build_module=lambda: _Forward(miniature().make()),
        build_input=build_input,
        seed=0,
    )


@pytest.mark.compute_training
def test_five_steps_bfb() -> None:
    """Freeze the recipe: objective, draws, gradients, clip, and AdamW."""

    def inputs() -> dict[str, Tensor]:
        batch = build_input()
        batch["features"] = torch.randn(BATCH, 1 + GRID * GRID, TARGET)
        return batch

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="speedrundit_five_steps",
        build_module=lambda: _FiveSteps(miniature().make()),
        build_input=inputs,
        seed=42,
    )


@pytest.mark.compute_training
def test_the_golden_bites() -> None:
    """A golden nobody has seen fail is not evidence.

    Perturbs one parameter and asserts the comparison notices. Done through
    the harness's own ``run`` so the perturbation travels the compared path
    rather than a path built beside it.
    """

    def perturbed(module: nn.Module, inp: Tensor) -> Tensor:
        with torch.no_grad():
            next(iter(module.parameters())).add_(0.125)
        assert isinstance(module, _Init)
        return module(inp)

    with pytest.raises(AssertionError):
        assert_bfb_against_golden(
            golden_dir=_CWD / "testdata",
            golden_name="speedrundit_init",
            build_module=lambda: _Init(miniature().make()),
            build_input=lambda: torch.zeros(1),
            seed=0,
            run=perturbed,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
