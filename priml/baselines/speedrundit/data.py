"""Prepared latent data and deterministic synthetic fixtures."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Iterator, Self, TypedDict, override

import torch
from configgle import Fig
from torch import Tensor

from priml.paths import resolve_working_dir
from priml.runtime import get_device


class SpeedrunDiTData:
    class Config(Fig["SpeedrunDiTData"]):
        base_dir: Path | str | None = None
        working_dir: Path | str = "/datasets/imagenet256"
        batch_size: int = 8
        eval_batch_size: int = 8
        image_size: int = 16
        latent_channels: int = 32
        cls_token_dim: int = 768
        num_classes: int = 1000
        num_samples: int = 16
        synthetic: bool = False
        device: str = "auto"
        dtype: torch.dtype = torch.float32

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.batch_size <= 0 or self.eval_batch_size <= 0:
                raise ValueError("batch sizes must be positive")
            return super().finalize()

    class Batch(TypedDict):
        media: Tensor
        label: Tensor
        cls_token: Tensor
        features: list[Tensor]

    def __init__(self, config: Config) -> None:
        self.config = config
        device = get_device(config.device)
        if config.synthetic:
            generator = torch.Generator().manual_seed(0)
            n = config.num_samples
            self.media = torch.randn(n, config.latent_channels, config.image_size, config.image_size, generator=generator, dtype=config.dtype, device=device)
            self.label = torch.arange(n, device=device) % config.num_classes
            self.cls = torch.randn(n, config.cls_token_dim, generator=generator, dtype=config.dtype, device=device)
            self.features = [self.cls[:, None, :]]
        else:
            path = Path(config.working_dir) / "train.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Prepared SpeedrunDiT data not found at {path}.")
            payload = torch.load(path, map_location=device, weights_only=True)
            for key in ("media", "label", "cls_token"):
                if key not in payload:
                    raise ValueError(f"{path} is missing required key {key!r}.")
            self.media = payload["media"].to(device=device, dtype=config.dtype)
            self.label = payload["label"].to(device=device, dtype=torch.long)
            self.cls = payload["cls_token"].to(device=device, dtype=config.dtype)
            self.features = [v.to(device=device, dtype=config.dtype) for v in payload.get("features", [])]
        if self.media.ndim != 4 or self.media.shape[1:] != (config.latent_channels, config.image_size, config.image_size):
            raise ValueError("latent tensor has incompatible SpeedrunDiT geometry")

    def train_dataloader(self) -> Iterator[Batch]:
        order = torch.randperm(len(self.media), device=self.media.device)
        return self._batches(order, self.config.batch_size)

    def eval_dataloader(self) -> Iterator[Batch]:
        return self._batches(torch.arange(len(self.media), device=self.media.device), self.config.eval_batch_size)

    def _batches(self, order: Tensor, size: int) -> Iterator[Batch]:
        for ids in order.split(size):
            yield {"media": self.media[ids], "label": self.label[ids], "cls_token": self.cls[ids], "features": [f[ids] for f in self.features]}

    def __len__(self) -> int:
        return len(self.media)
