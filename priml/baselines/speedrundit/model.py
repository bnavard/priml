"""Native, configurable SiT/SR-DiT model."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

import math

from configgle import Fig
from torch import Tensor, nn

import torch
from torch.nn import functional as F


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class SpeedrunDiT(nn.Module):
    """SiT backbone with optional SPRINT-style sparse residual fusion."""

    class Config(Fig["SpeedrunDiT"]):
        """Model geometry and SR-DiT mechanisms."""

        channels_in: int = 32
        channels_out: int = 32
        image_size: int = 16
        patch_size: int = 2
        channels_hidden: int = 768
        depth: int = 12
        heads: int = 12
        mlp_ratio: float = 4.0
        num_classes: int = 1000
        class_dropout: float = 0.1
        projector_dims: tuple[int, ...] = (768,)
        projector_hidden: int = 2048
        cls_token_dim: int = 768
        _: KW_ONLY
        qk_norm: bool = False
        rope: bool = False
        sprint: bool = False
        sprint_drop_ratio: float = 0.75
        path_drop_prob: float = 0.05
        dtype: torch.dtype | None = None

        @override
        def finalize(self) -> Self:
            if self.image_size % self.patch_size:
                raise ValueError("image_size must be divisible by patch_size.")
            if self.channels_hidden % self.heads:
                raise ValueError("channels_hidden must be divisible by heads.")
            if self.depth < 1:
                raise ValueError("depth must be positive.")
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        config.finalize()
        self.config = config
        d = config.channels_hidden
        self.patch = nn.Conv2d(config.channels_in, d, config.patch_size, config.patch_size)
        self.grid = config.image_size // config.patch_size
        self.position = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, d))
        self.time = _Timestep(d)
        self.label = nn.Embedding(config.num_classes + 1, d)
        self.cls_in = nn.Linear(config.cls_token_dim, d)
        self.blocks = nn.ModuleList([_Block(d, config.heads, config.mlp_ratio) for _ in range(config.depth)])
        self.final = _Final(d, config.patch_size, config.channels_out, config.cls_token_dim)
        self.projectors = nn.ModuleList([_Projector(d, config.projector_hidden, z) for z in config.projector_dims])
        self.sprint = config.sprint
        self.drop_ratio = config.sprint_drop_ratio
        self._reset()

    def _reset(self) -> None:
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.label.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.out.weight)
        nn.init.zeros_(self.final.out.bias)
        nn.init.zeros_(self.final.cls_out.weight)
        nn.init.zeros_(self.final.cls_out.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor, *, cls_token: Tensor, return_projections: bool = True) -> tuple[Tensor, list[Tensor], Tensor]:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_in(cls_token)[:, None], tokens), dim=1) + self.position
        labels = y.clamp_min(0).clamp_max(self.config.num_classes)
        if self.training and self.config.class_dropout > 0:
            dropped = torch.rand(labels.shape, device=labels.device) < self.config.class_dropout
            labels = torch.where(dropped, torch.full_like(labels, self.config.num_classes), labels)
        condition = self.time(t) + self.label(labels)
        projections: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, condition)
            if return_projections and i < len(self.projectors):
                projections.append(self.projectors[i](tokens))
            if self.sprint and self.training and i == len(self.blocks) // 2:
                keep = max(1, int(tokens.shape[1] * (1 - self.drop_ratio)))
                ids = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device).topk(keep, dim=1).indices
                sparse = tokens.gather(1, ids[..., None].expand(-1, -1, tokens.shape[-1]))
                tokens = tokens + sparse.mean(dim=1, keepdim=True)
        patch, cls_out = self.final(tokens, condition)
        p = self.config.patch_size
        patch = patch.reshape(x.shape[0], self.grid, self.grid, p, p, self.config.channels_out)
        velocity = torch.einsum("nhwpqc->nchpwq", patch).reshape_as(x)
        return velocity, projections, cls_out


class _Timestep(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = torch.cat((torch.cos(t[:, None] * freq), torch.sin(t[:, None] * freq)), dim=-1)
        if emb.shape[-1] != self.dim:
            emb = F.pad(emb, (0, 1))
        return self.net(emb.to(t.dtype))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * ratio)), nn.GELU(), nn.Linear(int(dim * ratio), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, -1)
        h = _modulate(self.norm1(x), shift_a, scale_a)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a[:, None] * h
        return x + gate_m[:, None] * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))


class _Final(nn.Module):
    def __init__(self, dim: int, patch: int, channels: int, cls_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.out = nn.Linear(dim, patch * patch * channels)
        self.cls_out = nn.Linear(dim, cls_dim)

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale = self.ada(c).chunk(2, -1)
        x = _modulate(self.norm(x), shift, scale)
        return self.out(x[:, 1:]), self.cls_out(x[:, 0])


class _Projector(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
