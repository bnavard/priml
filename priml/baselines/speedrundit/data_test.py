"""The loader: on-disk contract, pairing, shuffling, and resume.

Every test builds a corpus in ``tmp_path`` in the exact layout the reference
writes, so the enumeration, the sort, and the label join are exercised on real
files rather than asserted from the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json

import numpy as np
import pytest
import torch

from priml.baselines.speedrundit.data import SpeedrunDiTData, relative_names


CHANNELS: Final = 8
SIZE: Final = 4
WIDTH: Final = 16


def write_corpus(
    root: Path,
    *,
    count: int = 8,
    features: bool = True,
) -> None:
    """Write a corpus in the reference's layout.

    Each latent is a distinct constant, so a row is identifiable by value and
    an ordering bug shows up as a wrong number rather than a wrong shape.

    Args:
      root: Destination directory.
      count: Samples to write.
      features: Whether to write alignment targets beside the corpus.

    """
    images = root / "images" / "00000"
    latents = root / "vae-in" / "00000"
    images.mkdir(parents=True)
    latents.mkdir(parents=True)
    labels: list[list[object]] = []
    for index in range(count):
        stem = f"{index:08d}"
        np.save(
            images / f"img{stem}.npy",
            np.full((3, SIZE, SIZE), index, dtype=np.uint8),
        )
        np.save(
            latents / f"img-latents-{stem}.npy",
            np.full((1, CHANNELS, SIZE, SIZE), index, dtype=np.float32),
        )
        labels.append([f"00000/img-latents-{stem}.npy", index % 5])
    (root / "vae-in" / "dataset.json").write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )
    if features:
        tokens = 1 + SIZE * SIZE
        np.save(root / "cls_token.npy", np.zeros((count, WIDTH), np.float32))
        np.save(root / "features.npy", np.zeros((count, tokens, WIDTH), np.float32))


def make(root: Path, **overrides: object) -> SpeedrunDiTData:
    """Build a loader over a corpus in ``root``.

    ``base_dir`` is ``/`` so the logical path composes to ``root`` unchanged;
    that is the standard way to defeat ``/opt/scratch`` in a test.

    Args:
      root: Corpus directory.
      **overrides: Config fields to set.

    Returns:
      data: A built loader.

    """
    config = SpeedrunDiTData.Config()
    config.base_dir = "/"
    config.working_dir = str(root)
    config.batch_size = 2
    config.eval_batch_size = 2
    config.device = "cpu"
    config.latent_scale = 1.0
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_batch_carries_the_named_contract(tmp_path: Path) -> None:
    """Blessed keys plus the auxiliaries this model needs, all named."""
    write_corpus(tmp_path)
    batch = next(iter(make(tmp_path).eval_dataloader()))
    assert batch["media"].shape == (2, CHANNELS, SIZE, SIZE)
    assert batch["label"].shape == (2,)
    assert batch["cls_token"].shape == (2, WIDTH)
    assert batch["valid_count"] == 2
    assert len(batch["features"]) == 1


def test_the_singleton_encode_axis_is_dropped(tmp_path: Path) -> None:
    """The encoder wrote one sample per file, so each carries a batch axis of
    one that the corpus must not keep."""
    write_corpus(tmp_path)
    assert make(tmp_path).corpus.media.ndim == 4


def test_labels_pair_with_their_own_latents(tmp_path: Path) -> None:
    """The join is by name, and the two trees are ordered independently.

    Each latent is the constant of its index, so a mispaired label is visible
    as a value that disagrees with the row it sits beside.
    """
    write_corpus(tmp_path)
    data = make(tmp_path, eval_batch_size=8)
    batch = next(iter(data.eval_dataloader()))
    for row in range(8):
        assert batch["media"][row].flatten()[0].item() == float(row)
        assert batch["label"][row].item() == row % 5


def test_eval_order_is_the_stored_order(tmp_path: Path) -> None:
    """Evaluation must not shuffle, or two evals disagree."""
    write_corpus(tmp_path)
    data = make(tmp_path, eval_batch_size=8)
    first = next(iter(data.eval_dataloader()))["media"]
    second = next(iter(data.eval_dataloader()))["media"]
    assert torch.equal(first, second)


def test_the_latent_scale_is_applied_once(tmp_path: Path) -> None:
    """The tokenizer constant multiplies the stored latent, exactly once."""
    write_corpus(tmp_path)
    data = make(tmp_path, latent_scale=0.5, eval_batch_size=8)
    batch = next(iter(data.eval_dataloader()))
    assert batch["media"][3].flatten()[0].item() == pytest.approx(1.5)


def test_training_shuffles_and_the_stream_is_replayable(tmp_path: Path) -> None:
    """Two loaders on one seed walk the corpus identically.

    The order has to be rebuildable from ``(seed, pass_index)`` alone, because
    that is what lets a resumed run continue the pass it was interrupted in
    without having saved the permutation.
    """
    write_corpus(tmp_path)
    first = [b["media"].clone() for b in make(tmp_path).train_dataloader()]
    second = [b["media"].clone() for b in make(tmp_path).train_dataloader()]
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_consecutive_passes_differ(tmp_path: Path) -> None:
    """A new pass reshuffles; repeating one order would defeat shuffling."""
    write_corpus(tmp_path)
    data = make(tmp_path)
    first = torch.cat([b["media"] for b in data.train_dataloader()])
    second = torch.cat([b["media"] for b in data.train_dataloader()])
    assert not torch.equal(first, second)


def test_a_short_final_batch_is_dropped_by_default(tmp_path: Path) -> None:
    """Training keeps a constant shape so a compiled step never retraces."""
    write_corpus(tmp_path, count=7)
    data = make(tmp_path, batch_size=2)
    batches = list(data.train_dataloader())
    assert len(batches) == 3
    assert all(b["valid_count"] == 2 for b in batches)


def test_eval_pads_and_reports_the_valid_count(tmp_path: Path) -> None:
    """Evaluation keeps every row, padding to a constant width and SAYING so.

    A metric that averaged the padding in would report a score that depends on
    how short the final batch happened to be.
    """
    write_corpus(tmp_path, count=7)
    batches = list(make(tmp_path, eval_batch_size=2).eval_dataloader())
    assert len(batches) == 4
    assert batches[-1]["valid_count"] == 1
    assert batches[-1]["media"].shape[0] == 2
    assert torch.all(batches[-1]["media"][1] == 0)


def test_resume_continues_the_interrupted_pass(tmp_path: Path) -> None:
    """A restored loader yields exactly the batches the first one had left."""
    write_corpus(tmp_path)
    reference = make(tmp_path)
    expected = [b["media"].clone() for b in reference.train_dataloader()]

    interrupted = make(tmp_path)
    stream = interrupted.train_dataloader()
    taken = [next(stream)["media"].clone()]
    state = interrupted.state_dict()

    resumed = make(tmp_path)
    resumed.load_state_dict(state)
    rest = [b["media"].clone() for b in resumed.train_dataloader()]

    for left, right in zip(expected, taken + rest, strict=True):
        assert torch.equal(left, right)


def test_state_round_trips(tmp_path: Path) -> None:
    """The loader's position survives a save and load."""
    write_corpus(tmp_path)
    data = make(tmp_path)
    _ = list(data.train_dataloader())
    state = data.state_dict()
    restored = make(tmp_path)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_relative_names_sorts_and_skips_the_manifest(tmp_path: Path) -> None:
    """The sort is the pairing key between the two trees.

    Separators are normalized so a corpus prepared on one platform reads the
    same on another, and the manifest is excluded so it cannot be mistaken for
    a sample.
    """
    write_corpus(tmp_path, count=3)
    names = relative_names(tmp_path / "vae-in")
    assert names == sorted(names)
    assert all(name.endswith(".npy") for name in names)
    assert all("\\" not in name for name in names)
    assert len(names) == 3


def test_num_samples_takes_a_prefix(tmp_path: Path) -> None:
    """Capping the corpus keeps the first rows, not a random subset."""
    write_corpus(tmp_path)
    data = make(tmp_path, num_samples=3, eval_batch_size=3)
    assert data.corpus.count == 3
    batch = next(iter(data.eval_dataloader()))
    assert [v.item() for v in batch["media"][:, 0, 0, 0]] == [0.0, 1.0, 2.0]


def test_a_corpus_without_features_still_loads(tmp_path: Path) -> None:
    """Alignment targets are a preparation product, not a requirement."""
    write_corpus(tmp_path, features=False)
    batch = next(iter(make(tmp_path).eval_dataloader()))
    assert batch["features"] == []


def test_images_are_absent_unless_asked_for(tmp_path: Path) -> None:
    """Images are ~50x the latents and a run using precomputed features never
    reads them."""
    write_corpus(tmp_path)
    assert "raw_image" not in next(iter(make(tmp_path).eval_dataloader()))
    kept = make(tmp_path, keep_images=True)
    assert next(iter(kept.eval_dataloader()))["raw_image"].shape == (2, 3, SIZE, SIZE)


def test_a_missing_corpus_names_the_preparer(tmp_path: Path) -> None:
    """The error has to say what to run, not just what is absent."""
    data = make(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        _ = next(iter(data.eval_dataloader()))


def test_mismatched_trees_are_refused(tmp_path: Path) -> None:
    """The trees are paired by position, so unequal lengths are unrecoverable."""
    write_corpus(tmp_path)
    (tmp_path / "images" / "00000" / "img00000007.npy").unlink()
    with pytest.raises(ValueError, match="paired by position"):
        _ = next(iter(make(tmp_path).eval_dataloader()))


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("eval_batch_size", -1), ("num_samples", 0)],
)
def test_nonpositive_sizes_are_refused(field: str, value: int) -> None:
    """A zero batch is a configuration error, not an empty pass."""
    config = SpeedrunDiTData.Config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        _ = config.copy_tree().finalize()


def test_length_matches_what_the_loader_yields(tmp_path: Path) -> None:
    """``len`` is what a progress bar and an epoch budget read."""
    write_corpus(tmp_path, count=7)
    data = make(tmp_path, batch_size=2)
    assert len(data) == len(list(data.train_dataloader()))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
