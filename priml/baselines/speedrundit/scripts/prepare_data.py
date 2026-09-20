"""Validate or create a prepared SpeedrunDiT latent bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--cls-dim", type=int, default=768)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    payload = {
        "media": torch.randn(args.samples, args.channels, args.resolution, args.resolution, generator=generator),
        "label": torch.arange(args.samples) % 1000,
        "cls_token": torch.randn(args.samples, args.cls_dim, generator=generator),
        "features": [],
    }
    target = args.output / "train.pt"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}.")
    torch.save(payload, target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

