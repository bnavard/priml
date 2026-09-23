"""One SR-DiT optimizer update.

The recipe drives the update itself rather than calling the inherited
:meth:`TrainStep.step`, for one reason: the objective returns six named terms
and the step publishes each of them, so the loss cannot go through the base's
single-tensor ``loss`` slot. Everything else the base owns -- the optimizer,
the gradient clip, the EMA, the autocast policy, the step counter -- is used as
it stands, and the ``with self.timer_step:`` bracket is what advances
``global_step`` so every cadence above still means what it says.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.train.ema import EMA
from priml.train.grad_clip import clip_grad_norm_
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from priml.train.custom_types import EMAProtocol, TrainStepOutput


__all__ = ["SpeedrunDiTTrainStep"]


class SpeedrunDiTTrainStep(TrainStep):
    """Flow-matching update over prepared latents."""

    class Config(
        Makes["SpeedrunDiTTrainStep"],
        TrainStep.Config[SpeedrunDiT.Config],
        kw_only=True,
    ):
        """Configuration for SpeedrunDiTTrainStep."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: SpeedrunDiT.Config = field(default_factory=SpeedrunDiT.Config)
        """The velocity field."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.AdamW,
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                eps=1e-8,
            ),
        )
        """A single AdamW group over every parameter.

        Not a ``CompositeOptimizer``: the reference hands AdamW
        ``model.parameters()`` whole, and a router that skipped the frozen
        position table would build a different parameter list and a different
        optimizer state."""

        gradient_clip_norm: float = 1.0
        """Global gradient-norm ceiling."""

        ema: Makeable[EMAProtocol] = field(
            default_factory=lambda: EMA.Config(
                decay=0.9999,
                update_after_step=0,
                track_buffers=False,
                shadow_kind="module",
            ),
        )
        """Weight-averaging shadow, updated after each optimizer step."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype for the forward; ``None`` runs in full precision."""

        # ---- This recipe's own. ----

        objective: SpeedrunDiTLoss.Config = field(
            default_factory=SpeedrunDiTLoss.Config,
        )
        """The four-term flow-matching objective.

        A slot of its own rather than the base's ``loss``: that one is typed
        to return a single tensor from ``(prediction, **batch)``, and this
        objective drives the model itself so it can build the noised input the
        model consumes."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: SpeedrunDiTTrainStep.Config = config
        self.objective = config.objective.make()

    @property
    @override
    def model(self) -> SpeedrunDiT:
        """The velocity field, narrowed.

        Returns:
          model: The built model.

        """
        built = super().model
        assert isinstance(built, SpeedrunDiT)
        return built

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Take one optimizer step.

        Args:
          **batch: A :class:`~priml.baselines.speedrundit.data.SpeedrunDiTBatch`.

        Returns:
          output: The total loss, a probe of the velocity, and every term.

        """
        self.model.train()
        with self._autocast():
            result = self._evaluate(batch)
        result.loss.backward()
        # Clipped outside the timer bracket: the bracket is what advances
        # ``global_step``, and the clip is not part of the update it counts.
        if self.config.gradient_clip_norm != float("inf"):
            _ = clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
        # This bracket is what advances ``global_step``, so every cadence
        # above -- eval, checkpoint, the schedule horizon -- counts one
        # optimizer update per pass through it.
        with self.timer_step:
            self.apply_learning_rate()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            self.ema(self.model)
        return {
            "loss": result.loss.detach(),
            "model": result.denoising.detach(),
            "metrics": _metrics(result),
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score one batch without updating anything.

        Args:
          **batch: A :class:`~priml.baselines.speedrundit.data.SpeedrunDiTBatch`.

        Returns:
          output: The total loss and a probe of the velocity error.

        """
        was_training = self.model.training
        self.model.eval()
        try:
            with (
                torch.inference_mode(),
                self.ema.apply_to(self.model),
                self._autocast(),
            ):
                result = self._evaluate(batch)
        finally:
            self.model.train(was_training)
        return {
            "loss": result.loss.detach(),
            "model": result.denoising.detach(),
            "metrics": _metrics(result),
        }

    @contextmanager
    def _autocast(self) -> Generator[None]:
        """Enter autocast when the recipe asks for it.

        The base builds this inline inside ``__call__`` and ``call_eval``,
        neither of which this recipe uses: the objective drives the model
        itself, so the context has to be opened here.

        Yields:
          None: With autocast active, or nothing when it is disabled.

        """
        dtype = self.config.dtype_autocast
        if dtype is None:
            with nullcontext():
                yield
            return
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=dtype,
            cache_enabled=self.config.autocast_cache_enabled,
        ):
            yield

    def _evaluate(self, batch: dict[str, object]) -> SpeedrunDiTLoss.Output:
        """Run the objective over one batch.

        Args:
          batch: The raw batch mapping.

        Returns:
          result: Every term of the objective.

        """
        media = batch["media"]
        label = batch["label"]
        cls_token = batch["cls_token"]
        features = batch.get("features", [])
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        assert isinstance(cls_token, Tensor)
        assert isinstance(features, list)
        return self.objective(
            self.model,
            media=media,
            label=label,
            cls_token=cls_token,
            features=cast("list[Tensor]", features),
        )


def _metrics(result: SpeedrunDiTLoss.Output) -> dict[str, float | Tensor]:
    """Publish every term of the objective, keyed for the tracker."""
    return {
        "denoising": result.denoising.detach().mean(),
        "cls": result.cls.detach().mean(),
        "projection": result.projection.detach(),
        "cfm": result.cfm.detach(),
        "cfm_cls": result.cfm_cls.detach(),
    }
