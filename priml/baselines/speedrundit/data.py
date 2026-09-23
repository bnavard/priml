"""Prepared ImageNet latents, images, and class labels for SR-DiT.

On-disk contract, as written by ``scripts/prepare_data.py`` and by the
reference's own preprocessing::

    <working_dir>/
      images/
        00000/img00000000.png      # uint8 RGB, the REPA encoder's input
        ...
      vae-in/
        00000/img-latents-00000000.npy   # float32 [1, C, H, W] INVAE latent
        ...
        dataset.json                     # {"labels": [[name, class], ...]}
      cls_token.npy                      # float32 [N, width], required
      features.npy                       # float32 [N, 1 + H * W, width], optional

The last two are the frozen encoder's class and patch features, in latent
order. They are not part of the reference's layout -- it encodes them from
the images every step -- so a real corpus needs them precomputed.

Two facts about that layout are load-bearing and easy to lose. The two trees
are enumerated and sorted INDEPENDENTLY and then paired by position, which
works only because both naming schemes are index-monotonic; and the labels come
from ``vae-in/dataset.json`` alone, keyed by the latent's relative path with
separators normalized to forward slashes. A loader that walks one tree and
derives the other's names, or that reads ``images/dataset.json``, will look
correct and pair the wrong label to the wrong latent.

This module only READS. Staging lives in ``scripts/prepare_data.py``, so
building a config touches neither the network nor the disk.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired, Self, TypedDict, cast, override

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.lib.custom_json import DictCodec, IntCodec, ListCodec, StrCodec, loads
from priml.math.seed import salt
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from numpy.typing import NDArray


__all__ = ["SpeedrunDiTBatch", "SpeedrunDiTData", "read_labels", "relative_names"]


LATENT_RANK: Final = 4
"""Rank of a stored latent once its singleton encode axis is dropped."""


class SpeedrunDiTBatch(TypedDict):
    """One training batch.

    ``media`` and ``label`` carry priml's blessed meanings. The rest is named
    rather than smuggled: ``cls_token`` is the frozen encoder's class feature,
    which the model diffuses alongside the latent, and ``features`` are the
    per-token alignment targets. ``raw_image`` is present only when a run
    computes its own features and is absent otherwise, which is why it is
    ``NotRequired`` rather than an empty tensor.
    """

    media: Tensor
    """Scaled INVAE latents, ``[batch, channels, size, size]``."""

    label: Tensor
    """ImageNet class indices, ``[batch]``."""

    cls_token: Tensor
    """Frozen encoder class features, ``[batch, channels_cls]``."""

    features: list[Tensor]
    """Per-target token features, each ``[batch, tokens, width]``."""

    valid_count: int
    """Rows of the batch that are real rather than padding."""

    raw_image: NotRequired[Tensor]
    """Source images, ``[batch, 3, size, size]`` uint8, when kept."""


class SpeedrunDiTData:
    """Reads a prepared latent corpus and serves shuffled batches.

    Example:
      cfg = SpeedrunDiTData.Config(batch_size=256)
      data = cfg.make()
      batch = next(iter(data.train_dataloader()))

    Raises:
      FileNotFoundError: From the dataloaders, when the prepared corpus is
        absent. Construction never reads the disk, so the failure surfaces at
        first use rather than at config time.

    """

    class Config(Fig["SpeedrunDiTData"]):
        """Configuration for SpeedrunDiTData."""

        base_dir: Path | str | None = None
        """Resource root; the loop fills it when left unset."""

        working_dir: Path | str = "/datasets/imagenet256-invae"
        """Logical corpus location, resolved beneath ``base_dir``."""

        _: KW_ONLY

        batch_size: int = 256
        """Samples per training step."""

        eval_batch_size: int | None = None
        """Samples per evaluation step; ``None`` reuses ``batch_size``."""

        latent_scale: float = 0.3099
        """Multiplier putting INVAE latents at unit-ish variance.

        A property of the tokenizer's training, not a tunable: changing it
        without re-fitting the tokenizer rescales every target and makes a
        score incomparable with any other run."""

        drop_last: bool = True
        """Discard a short final batch rather than padding it."""

        keep_images: bool = False
        """Hold the source images in memory for on-the-fly REPA features.

        Off by default: the images are ~50x the latents and a run using
        precomputed features never reads them."""

        num_samples: int | None = None
        """Prefix of the corpus to use; ``None`` takes all of it."""

        device: str = "auto"
        """Where the corpus is resident; ``auto`` resolves per the runtime."""

        dtype: torch.dtype = torch.float32
        """Latent dtype once loaded."""

        seed: int = 0
        """Base seed for the shuffle stream.

        Set rather than left to the loop's ``seed`` because the order samples
        are visited is a reproducibility knob rather than a variance one, and
        a resumed run has to replay the pass it was interrupted in."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be positive; got {self.batch_size}.")
            if self.eval_batch_size is not None and self.eval_batch_size <= 0:
                raise ValueError(
                    f"eval_batch_size must be positive; got {self.eval_batch_size}.",
                )
            if self.num_samples is not None and self.num_samples <= 0:
                raise ValueError(
                    f"num_samples must be positive; got {self.num_samples}.",
                )
            return super().finalize()

    class StateDict(TypedDict):
        """Checkpoint payload; ``active_pass`` is None between passes."""

        passes: int
        active_pass: int | None
        next_batch: int
        timer_epoch: Mapping[str, object]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.directory = Path(config.working_dir).expanduser()
        self.device = get_device(config.device)
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        self.timer_epoch = CheckpointableStepTimer()
        self._corpus: _Corpus | None = None
        self._passes = 0
        self._active_pass: int | None = None
        self._next_batch = 0

    @property
    def corpus(self) -> _Corpus:
        """Load the corpus on first use.

        Returns:
          corpus: The resident tensors.

        """
        if self._corpus is None:
            self._corpus = _load_corpus(
                self.directory,
                device=self.device,
                dtype=self.config.dtype,
                latent_scale=self.config.latent_scale,
                keep_images=self.config.keep_images,
                limit=self.config.num_samples,
            )
        return self._corpus

    def train_dataloader(self) -> _TrainPasses:
        """Serve shuffled passes over the corpus, a fresh one per ``iter``.

        Re-iterable rather than an iterator: the loop asks for this once and
        calls ``iter`` on it at every epoch boundary, so an exhausted generator
        here would end the run at the first boundary.

        Returns:
          passes: An iterable whose every iteration walks one pass; a short
          final batch is dropped unless ``drop_last`` is off.

        """
        return _TrainPasses(self._train_pass, self.__len__)

    def eval_dataloader(self) -> Iterator[SpeedrunDiTBatch]:
        """Iterate the corpus once, in stored order.

        Returns:
          batches: Unshuffled batches, padding the last one.

        """
        corpus = self.corpus
        order = torch.arange(corpus.count, device=corpus.media.device)
        size = self.eval_batch_size
        for rows in order.split(size):
            yield corpus.batch(rows, width=size, valid=int(rows.numel()))

    def state_dict(self) -> StateDict:
        """Capture the loader position.

        Returns:
          state: Passes started, the pass in progress, the next batch within
          it, and the epoch timer.

        """
        return {
            "passes": self._passes,
            "active_pass": self._active_pass,
            "next_batch": self._next_batch,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the loader position.

        Takes effect when the next pass begins iterating, because the loop
        restores before it asks for an iterator; an interrupted pass resumes
        on its own permutation at the batch the checkpoint had not yet seen.

        Args:
          state_dict: A payload from :meth:`state_dict`.

        """
        state = cast(SpeedrunDiTData.StateDict, state_dict)
        self._passes = state["passes"]
        self._active_pass = state["active_pass"]
        self._next_batch = state["next_batch"]
        self.timer_epoch.load_state_dict(state["timer_epoch"])

    def __len__(self) -> int:
        """Batches in one training pass.

        Returns:
          count: Batch count under the configured ``drop_last``.

        """
        count, size = self.corpus.count, self.config.batch_size
        if self.config.drop_last:
            return count // size
        return (count + size - 1) // size

    def _permutation(self, count: int, pass_index: int) -> Tensor:
        """Draw the visiting order for one pass.

        A named, salted stream rather than the global one: the order has to be
        replayable from ``(seed, pass_index)`` alone so a resumed run rebuilds
        the pass it was interrupted in without having saved the permutation.

        Args:
          count: Samples in the corpus.
          pass_index: Zero-based pass number.

        Returns:
          order: A permutation of ``[0, count)``.

        """
        generator = torch.Generator()
        generator.manual_seed(salt("speedrundit_shuffle", self.config.seed, pass_index))
        order = torch.randperm(count, generator=generator)
        return order.to(self.corpus.media.device)

    def _train_pass(self) -> Iterator[SpeedrunDiTBatch]:
        """Walk the pass in progress, or begin the next one.

        Yields:
          batch: One training batch.

        """
        corpus = self.corpus
        if self._active_pass is None:
            self._active_pass = self._passes
            self._passes += 1
            self._next_batch = 0
        order = self._permutation(corpus.count, self._active_pass)
        size = self.config.batch_size
        for index, rows in enumerate(order.split(size)):
            valid = int(rows.numel())
            if (self.config.drop_last and valid != size) or index < self._next_batch:
                continue
            # Recorded BEFORE the yield: a checkpoint taken after the consumer
            # steps on this batch saves the index of the one it has not seen.
            self._next_batch = index + 1
            yield corpus.batch(rows, width=size, valid=valid)
        self._active_pass = None
        self._next_batch = 0


class _TrainPasses:
    """The training stream: each ``iter`` walks one pass over the corpus."""

    def __init__(
        self,
        walk: Callable[[], Iterator[SpeedrunDiTBatch]],
        length: Callable[[], int],
    ) -> None:
        self._walk = walk
        self._length = length

    def __iter__(self) -> Iterator[SpeedrunDiTBatch]:
        return self._walk()

    def __len__(self) -> int:
        return self._length()


class _Corpus:
    """The resident tensors of a prepared corpus."""

    def __init__(
        self,
        *,
        media: Tensor,
        label: Tensor,
        cls_token: Tensor,
        features: list[Tensor],
        images: Tensor | None,
    ) -> None:
        self.media = media
        self.label = label
        self.cls_token = cls_token
        self.features = features
        self.images = images
        self.count = int(media.shape[0])

    def batch(self, rows: Tensor, *, width: int, valid: int) -> SpeedrunDiTBatch:
        """Gather one batch, padding to a constant width.

        Shapes stay constant across a pass so a compiled step never retraces
        and the padding is REPORTED rather than hidden; ``valid_count`` is what
        a metric truncates by.

        Args:
          rows: Sample indices.
          width: Rows the batch must present.
          valid: Rows of ``width`` that are real.

        Returns:
          batch: One batch.

        """
        pad = width - valid
        batch: SpeedrunDiTBatch = {
            "media": _pad(self.media[rows], pad),
            "label": _pad(self.label[rows], pad),
            "cls_token": _pad(self.cls_token[rows], pad),
            "features": [_pad(feature[rows], pad) for feature in self.features],
            "valid_count": valid,
        }
        if self.images is not None:
            batch["raw_image"] = _pad(self.images[rows], pad)
        return batch


def _pad(x: Tensor, pad: int) -> Tensor:
    """Extend a batch to a constant width with zero rows."""
    if pad <= 0:
        return x
    filler = x.new_zeros((pad, *x.shape[1:]))
    return torch.cat([x, filler], dim=0)


def read_labels(manifest: Path) -> dict[str, int]:
    """Read the label manifest, keyed by slash-separated latent name.

    One parser for the loader and the preparer's verifier: two would agree
    only until one of them stopped normalizing separators.

    Args:
      manifest: ``vae-in/dataset.json``.

    Returns:
      labels: Class index per relative latent name.

    """
    payload = DictCodec.coerce(loads(manifest.read_text(encoding="utf-8")))
    entries = [ListCodec.coerce(entry) for entry in ListCodec.coerce(payload["labels"])]
    # Keys are normalized to forward slashes: a corpus prepared on Windows
    # writes backslashes into the manifest but is read on either platform.
    return {
        StrCodec.coerce(entry[0]).replace("\\", "/"): IntCodec.coerce(
            entry[1],
            default=None,
        )
        for entry in entries
    }


def relative_names(root: Path) -> list[str]:
    """Enumerate a tree's files as sorted, slash-separated relative names.

    The sort is the pairing key between the two trees, so it is written once
    here and used for both. Separators are normalized so a corpus prepared on
    one platform reads identically on another.

    Args:
      root: Directory to walk.

    Returns:
      names: Sorted relative paths, excluding the label manifest.

    """
    found = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".json"
    }
    return sorted(found)


def _load_corpus(
    directory: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    latent_scale: float,
    keep_images: bool,
    limit: int | None,
) -> _Corpus:
    """Read a prepared corpus into memory.

    Args:
      directory: Corpus root holding ``images/`` and ``vae-in/``.
      device: Where the tensors land.
      dtype: Latent dtype.
      latent_scale: Multiplier applied to every latent.
      keep_images: Whether to retain the source images.
      limit: Prefix to read, or ``None`` for all.

    Returns:
      corpus: The resident tensors.

    Raises:
      FileNotFoundError: If the corpus or its manifest is absent.
      ValueError: If the two trees disagree in length or a latent is misshaped.

    """
    latents_dir = directory / "vae-in"
    images_dir = directory / "images"
    manifest = latents_dir / "dataset.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No prepared SR-DiT corpus at {directory}. Build one with: "
            "uv --quiet run --frozen python -m "
            "priml.baselines.speedrundit.scripts.prepare_data",
        )
    latent_names = relative_names(latents_dir)
    image_names = relative_names(images_dir) if images_dir.is_dir() else []
    if image_names and len(image_names) != len(latent_names):
        raise ValueError(
            f"{len(image_names)} images against {len(latent_names)} latents in "
            f"{directory}; the two trees are paired by position and must match.",
        )
    if limit is not None:
        latent_names = latent_names[:limit]
        image_names = image_names[:limit]

    table = read_labels(manifest)
    labels = [table[name] for name in latent_names]

    latents = [np.load(latents_dir / name) for name in latent_names]
    stacked = torch.from_numpy(np.stack(latents))
    if stacked.ndim == LATENT_RANK + 1 and stacked.shape[1] == 1:
        # The encoder wrote one sample at a time, so each file carries a
        # singleton batch axis the corpus does not want.
        stacked = stacked.squeeze(1)
    if stacked.ndim != LATENT_RANK:
        raise ValueError(
            f"latents must be rank {LATENT_RANK} per sample; "
            f"got {tuple(stacked.shape)}.",
        )
    # Scaled once, here, so the model and the sampler both see the same
    # space; decoding inverts it by dividing.
    media = stacked.to(device=device, dtype=dtype) * latent_scale

    images = None
    if keep_images and image_names:
        images = torch.from_numpy(
            np.stack([_read_image(images_dir / name) for name in image_names]),
        ).to(device=device)

    count = media.shape[0]
    # The class token is an INPUT the model diffuses alongside the latent, so
    # a corpus without one cannot train at all; the per-token alignment
    # targets are optional, and without them the alignment term is zero.
    cls_path = directory / "cls_token.npy"
    if not cls_path.is_file():
        raise FileNotFoundError(
            f"No class-token targets at {cls_path}. They are the frozen "
            "encoder's class features, one row per latent; the synthetic "
            "corpus writes them, and a real corpus needs them precomputed.",
        )
    return _Corpus(
        media=media,
        label=torch.tensor(labels, device=device, dtype=torch.long),
        cls_token=_load_rows(cls_path, count, device, dtype),
        features=(
            [_load_rows(directory / "features.npy", count, device, dtype)]
            if (directory / "features.npy").is_file()
            else []
        ),
        images=images,
    )


def _read_image(path: Path) -> NDArray[np.uint8]:
    """Decode one stored image to ``[channels, height, width]`` uint8."""
    if path.suffix.lower() == ".npy":
        array = cast("NDArray[np.uint8]", np.load(path))
        shape = cast("tuple[int, ...]", array.shape)
        return array.reshape(-1, *shape[-2:])
    # Deferred: a run using precomputed features never loads Pillow.
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return array.transpose(2, 0, 1)


def _load_rows(
    path: Path,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Read the first ``count`` rows of a per-sample array."""
    return torch.from_numpy(np.load(path)[:count]).to(device=device, dtype=dtype)
