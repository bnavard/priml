"""Lightweight metrics for training and generated-sample checks."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor


class MeanMetric:
    class Config(Fig["MeanMetric"]):
        name: str = "loss"

    def __init__(self, config: Config) -> None:
        self.name = config.name
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: Tensor, **_: object) -> None:
        self.total += float(value.detach().mean())
        self.count += 1

    def compute(self) -> dict[str, float]:
        return {self.name: self.total / self.count if self.count else 0.0}

