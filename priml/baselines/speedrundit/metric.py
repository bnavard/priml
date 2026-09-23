"""Mean velocity error on held-out latents, the eval cadence's own question.

Not a distributional score: FID needs a full sampling pass, an INVAE decode
and a 50k reference batch, which belong to a script run against a checkpoint
rather than to an eval every ten thousand steps. This costs one forward and
answers whether the field is still improving.

A generative recipe has no class scores, so the per-sample error arrives as
the step's ``model`` output where a classifier would send logits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor

import torch
import torch.distributed as dist

from priml.lib.custom_json import FloatCodec


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["VelocityError"]


class VelocityError:
    """Accumulates the mean per-sample velocity error."""

    class Config(Fig["VelocityError"]):
        """Configuration for VelocityError."""

        key: str = "mse"
        """Name the computed value is published under."""

    class StateDict(TypedDict):
        """Checkpoint payload."""

        total: float
        count: int

    def __init__(self, config: Config) -> None:
        self.key = config.key
        self._total = 0.0
        self._count = 0

    def update(self, logits: Tensor, **batch: object) -> None:
        """Accumulate one batch.

        Args:
          logits: Per-sample velocity error, ``[batch]``.
          **batch: The full batch; ``valid_count`` is read when present.

        """
        valid = batch.get("valid_count")
        errors = logits.detach().flatten()
        # Padding rows are dropped, not averaged in: their error is zero, so
        # counting them would pull the mean down in proportion to how short
        # the final batch happened to be.
        rows = int(valid) if isinstance(valid, int) else errors.numel()
        rows = min(rows, errors.numel())
        if rows <= 0:
            return
        # Summed, not meant per batch: batches differ in width, and a mean of
        # batch means would weight a short final batch like a full one.
        self._total += float(errors[:rows].sum())
        self._count += rows

    def compute(self) -> Mapping[str, object]:
        """Reduce across ranks and report the mean.

        Returns:
          metrics: The mean velocity error, or zero when nothing was seen.

        """
        counts = torch.tensor([self._total, float(self._count)], dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            # NCCL reduces only CUDA tensors; gloo only CPU ones. Move for the
            # former and come back, so ``.tolist()`` works either way.
            if dist.get_backend() != "gloo":
                counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            counts = counts.cpu()
        total, count = (FloatCodec.coerce(value) for value in counts.tolist())
        return {self.key: total / count if count else 0.0}

    def reset(self) -> None:
        """Discard accumulated state."""
        self._total = 0.0
        self._count = 0

    def state_dict(self) -> StateDict:
        """Capture accumulated state.

        Returns:
          state: The running sum and count.

        """
        return {"total": self._total, "count": self._count}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore accumulated state.

        Args:
          state_dict: A payload from :meth:`state_dict`.

        """
        state = cast(VelocityError.StateDict, state_dict)
        self._total = state["total"]
        self._count = state["count"]
