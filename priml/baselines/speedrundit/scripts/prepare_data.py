#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Stage the SR-DiT latent corpus, or synthesize a small one for smoke runs.

Encoding ImageNet through INVAE and DINOv2 needs both checkpoints and a GPU,
so the real path VERIFIES and publishes an already-encoded tree rather than
producing one; the encode itself belongs to the reference's own preprocessing
and is not reimplemented here. The synthetic path writes the same layout with
random tensors, which is what makes the smoke experiment and the loader tests
run with no ImageNet, no tokenizer, and no network.

Publishing is atomic: the tree is built in a hidden sibling directory on the
same filesystem and renamed into place, so an interrupted run leaves either
the previous corpus or nothing, never half of one.

Examples:
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --help  # noqa: E501
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --synthetic --samples 64  # noqa: E501
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --verify --source /data/imagenet256  # noqa: E501

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import json
import shutil
import tempfile

import numpy as np

from priml.baselines.speedrundit.data import (
    SpeedrunDiTData,
    read_labels,
    relative_names,
)
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from collections.abc import Sequence


SHARD_WIDTH: Final = 5
"""Digits in the shard subdirectory the reference's preprocessing writes."""

INDEX_WIDTH: Final = 8
"""Digits in a sample's zero-padded index."""


class _Flags(Protocol):
    """Parsed command line."""

    directory: Path | None
    source: Path | None
    synthetic: bool
    verify: bool
    samples: int
    image_size: int
    latent_size: int
    latent_channels: int
    num_classes: int
    encoder_width: int
    seed: int


def default_directory() -> Path:
    """Resolve the corpus path a default run would read.

    Derived by finalizing the dataset config beneath the loop's own
    ``base_dir`` rather than by naming a path, so the preparer and the trainer
    cannot drift apart.

    Returns:
      directory: Where the corpus belongs.

    """
    config = SpeedrunDiTData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def verify(directory: Path) -> int:
    """Check a prepared corpus against the layout the loader expects.

    Args:
      directory: Corpus root.

    Returns:
      count: Samples the corpus holds.

    Raises:
      FileNotFoundError: If a required tree or the manifest is missing.
      ValueError: If the two trees disagree or a label is unaccounted for.

    """
    latents_dir = directory / "vae-in"
    manifest = latents_dir / "dataset.json"
    if not latents_dir.is_dir():
        raise FileNotFoundError(f"No vae-in/ tree under {directory}.")
    if not manifest.is_file():
        raise FileNotFoundError(f"No label manifest at {manifest}.")
    latent_names = relative_names(latents_dir)
    images_dir = directory / "images"
    if images_dir.is_dir():
        image_names = relative_names(images_dir)
        if len(image_names) != len(latent_names):
            raise ValueError(
                f"{len(image_names)} images against {len(latent_names)} "
                "latents; the trees are paired by position.",
            )
    labels = read_labels(manifest)
    missing = [name for name in latent_names if name not in labels]
    if missing:
        raise ValueError(
            f"{len(missing)} latents have no label; first is {missing[0]}.",
        )
    return len(latent_names)


def synthesize(
    directory: Path,
    *,
    samples: int,
    image_size: int,
    latent_size: int,
    latent_channels: int,
    num_classes: int,
    encoder_width: int,
    seed: int,
) -> None:
    """Write a synthetic corpus in the reference's layout.

    Args:
      directory: Destination, created fresh.
      samples: Samples to write.
      image_size: Side length of each stored image.
      latent_size: Side length of each stored latent.
      latent_channels: Channels per latent.
      num_classes: Classes cycled through.
      encoder_width: Width of the synthetic alignment features.
      seed: Seed for every draw.

    Raises:
      FileExistsError: If the destination already holds a corpus.

    """
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty {directory}.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}-"),
    )
    try:
        _write_synthetic(
            staging,
            samples=samples,
            image_size=image_size,
            latent_size=latent_size,
            latent_channels=latent_channels,
            num_classes=num_classes,
            encoder_width=encoder_width,
            seed=seed,
        )
        if directory.exists():
            directory.rmdir()
        staging.replace(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_synthetic(
    root: Path,
    *,
    samples: int,
    image_size: int,
    latent_size: int,
    latent_channels: int,
    num_classes: int,
    encoder_width: int,
    seed: int,
) -> None:
    """Fill a staging directory with a synthetic corpus.

    Args:
      root: Staging directory.
      samples: Samples to write.
      image_size: Side length of each stored image.
      latent_size: Side length of each stored latent.
      latent_channels: Channels per latent.
      num_classes: Classes cycled through.
      encoder_width: Width of the synthetic alignment features.
      seed: Seed for every draw.

    """
    rng = np.random.default_rng(seed)
    images_dir = root / "images"
    latents_dir = root / "vae-in"
    labels: list[Sequence[object]] = []
    tokens = 1 + latent_size * latent_size
    for index in range(samples):
        shard = f"{index:0{INDEX_WIDTH}d}"[:SHARD_WIDTH]
        (images_dir / shard).mkdir(parents=True, exist_ok=True)
        (latents_dir / shard).mkdir(parents=True, exist_ok=True)
        stem = f"{index:0{INDEX_WIDTH}d}"
        np.save(
            images_dir / shard / f"img{stem}.npy",
            rng.integers(0, 256, (3, image_size, image_size), dtype=np.uint8),
        )
        latent = rng.standard_normal(
            (1, latent_channels, latent_size, latent_size),
        ).astype(np.float32)
        np.save(latents_dir / shard / f"img-latents-{stem}.npy", latent)
        labels.append([f"{shard}/img-latents-{stem}.npy", index % num_classes])
    (latents_dir / "dataset.json").write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )
    # The alignment targets a real run precomputes with DINOv2. Stored beside
    # the corpus rather than inside vae-in/, which the loader pairs by
    # position and would otherwise mistake for latents.
    np.save(
        root / "cls_token.npy",
        rng.standard_normal((samples, encoder_width)).astype(np.float32),
    )
    np.save(
        root / "features.npy",
        rng.standard_normal((samples, tokens, encoder_width)).astype(np.float32),
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the command line.

    Args:
      parser: Parser to populate.

    """
    parser.add_argument("--directory", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)


def main() -> int:
    """Prepare or verify the corpus.

    Returns:
      status: Zero on success.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    directory = flags.directory or default_directory()

    if flags.synthetic:
        synthesize(
            directory,
            samples=flags.samples,
            image_size=flags.image_size,
            latent_size=flags.latent_size,
            latent_channels=flags.latent_channels,
            num_classes=flags.num_classes,
            encoder_width=flags.encoder_width,
            seed=flags.seed,
        )
        print(f"wrote {flags.samples} synthetic samples to {directory}")
        return 0

    if flags.source is not None:
        count = verify(flags.source)
        print(f"{flags.source} holds {count} verified samples")
        if flags.source.resolve() != directory.resolve():
            shutil.copytree(flags.source, directory, dirs_exist_ok=False)
            print(f"published to {directory}")
        return 0

    count = verify(directory)
    print(f"{directory} holds {count} verified samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
