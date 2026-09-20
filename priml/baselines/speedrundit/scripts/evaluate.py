"""Small evaluator for generated latent batches."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    args = parser.parse_args()
    samples = torch.load(args.samples, map_location="cpu", weights_only=True)
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
        raise ValueError("samples must be a rank-4 latent tensor")
    print({"count": samples.shape[0], "mean": float(samples.mean()), "std": float(samples.std())})


if __name__ == "__main__":
    main()

