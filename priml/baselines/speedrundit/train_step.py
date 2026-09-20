"""Training step for the SpeedrunDiT flow-matching baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, override

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor, nn

import torch

from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.optimizers import CompositeOptimizer
from priml.train.train_step import TrainStep

if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepOutput


class SpeedrunDiTTrainStep(TrainStep):
    class Config(Makes["SpeedrunDiTTrainStep"], TrainStep.Config, kw_only=True):
        model: Makeable[nn.Module] = field(default_factory=SpeedrunDiT.Config)
        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(default_factory=lambda: CompositeOptimizer.Config(optimizers=[PartialConfig(torch.optim.AdamW, lr=1e-4, weight_decay=0.0)]))
        loss: Makeable[Callable[..., SpeedrunDiTLoss]] = field(default_factory=SpeedrunDiTLoss.Config)
        total_train_steps: int = 400_000

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: SpeedrunDiTTrainStep.Config = config
        self.objective = config.loss.make()

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        media = batch["media"]
        label = batch["label"]
        cls_token = batch["cls_token"]
        features = batch.get("features")
        assert isinstance(media, Tensor) and isinstance(label, Tensor) and isinstance(cls_token, Tensor)
        assert features is None or isinstance(features, list)
        self.model.train()
        with self._autocast():
            terms = self.objective(self.model, media=media, label=label, cls_token=cls_token, features=features)
            loss = terms["loss"].mean()
        loss.backward()
        if torch.isfinite(loss) and torch.isfinite(torch.stack([p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None])).all():
            if self.config.gradient_clip_norm < float("inf"):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            with self.timer_step:
                self.optimizer.step()
        self.model.zero_grad(set_to_none=True)
        return {"loss": terms["loss"].detach(), "model": terms["denoising"].detach(), "metrics": {k: v.detach() for k, v in terms.items() if k != "loss"}}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        media, label, cls_token = batch["media"], batch["label"], batch["cls_token"]
        assert isinstance(media, Tensor) and isinstance(label, Tensor) and isinstance(cls_token, Tensor)
        with torch.no_grad():
            terms = self.objective(self.model, media=media, label=label, cls_token=cls_token, features=batch.get("features"))
        return {"loss": terms["loss"].detach(), "model": terms["denoising"].detach()}
