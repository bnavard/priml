"""Corpus staging: the layout it writes, and the ones it refuses.

Hermetic by parameter rather than by marker: every call passes an explicit
``--directory``, so nothing reaches ``/opt/scratch`` or the network.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from priml.baselines.speedrundit.data import SpeedrunDiTData, read_labels
from priml.baselines.speedrundit.scripts import prepare_data
from priml.lib.custom_json import DictCodec, ListCodec, StrCodec, loads


def test_default_directory_matches_the_loader(tmp_path: Path) -> None:
    """The preparer and the trainer resolve the corpus independently.

    If they drift, a successful preparation is followed by a run that cannot
    find what it just wrote.
    """
    del tmp_path
    config = SpeedrunDiTData.Config()
    config.base_dir = "/opt/scratch"
    expected = Path(config.copy_tree().finalize().working_dir)
    assert prepare_data.default_directory() == expected


def test_synthetic_writes_the_layout_the_loader_reads(tmp_path: Path) -> None:
    """The synthetic path is what makes the smoke run need no ImageNet."""
    target = tmp_path / "corpus"
    prepare_data.synthesize(
        target,
        samples=4,
        image_size=8,
        latent_size=4,
        latent_channels=8,
        num_classes=10,
        encoder_width=16,
        seed=0,
    )
    manifest = target / "vae-in" / "dataset.json"
    assert manifest.is_file()
    assert len(read_labels(manifest)) == 4
    assert (target / "cls_token.npy").is_file()
    assert (target / "features.npy").is_file()

    config = SpeedrunDiTData.Config()
    config.base_dir = "/"
    config.working_dir = str(target)
    config.batch_size = 2
    config.eval_batch_size = 2
    config.device = "cpu"
    batch = next(iter(config.make().eval_dataloader()))
    assert batch["media"].shape == (2, 8, 4, 4)
    assert batch["cls_token"].shape == (2, 16)


def test_synthetic_is_reproducible(tmp_path: Path) -> None:
    """One seed writes one corpus, byte for byte."""
    kwargs = {
        "samples": 3,
        "image_size": 8,
        "latent_size": 4,
        "latent_channels": 8,
        "num_classes": 10,
        "encoder_width": 16,
        "seed": 7,
    }
    first, second = tmp_path / "a", tmp_path / "b"
    prepare_data.synthesize(first, **kwargs)
    prepare_data.synthesize(second, **kwargs)
    left = (first / "vae-in" / "00000" / "img-latents-00000000.npy").read_bytes()
    right = (second / "vae-in" / "00000" / "img-latents-00000000.npy").read_bytes()
    assert left == right


def test_an_occupied_destination_is_untouched(tmp_path: Path) -> None:
    """Refuse rather than overwrite, and leave the existing tree intact."""
    target = tmp_path / "corpus"
    target.mkdir()
    (target / "keep.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_data.synthesize(
            target,
            samples=1,
            image_size=8,
            latent_size=4,
            latent_channels=8,
            num_classes=10,
            encoder_width=16,
            seed=0,
        )
    assert (target / "keep.txt").read_text(encoding="utf-8") == "hello"


def test_a_failed_build_publishes_nothing(tmp_path: Path) -> None:
    """Staging is atomic: an interrupted build leaves no partial corpus."""
    target = tmp_path / "corpus"
    with pytest.raises((ValueError, OSError)):
        prepare_data.synthesize(
            target,
            samples=-1,
            image_size=8,
            latent_size=4,
            latent_channels=-8,
            num_classes=10,
            encoder_width=16,
            seed=0,
        )
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_verify_counts_a_good_corpus(tmp_path: Path) -> None:
    """Verification reports what the loader would find."""
    target = tmp_path / "corpus"
    prepare_data.synthesize(
        target,
        samples=5,
        image_size=8,
        latent_size=4,
        latent_channels=8,
        num_classes=10,
        encoder_width=16,
        seed=0,
    )
    assert prepare_data.verify(target) == 5


def test_verify_accepts_what_the_loader_reads(tmp_path: Path) -> None:
    """A manifest written on Windows carries backslashes, which the loader
    normalizes; a verifier that did not would refuse a readable corpus.
    """
    target = tmp_path / "corpus"
    prepare_data.synthesize(
        target,
        samples=2,
        image_size=8,
        latent_size=4,
        latent_channels=8,
        num_classes=10,
        encoder_width=16,
        seed=0,
    )
    manifest = target / "vae-in" / "dataset.json"
    payload = DictCodec.coerce(loads(manifest.read_text(encoding="utf-8")))
    payload["labels"] = [
        [StrCodec.coerce(entry[0]).replace("/", "\\"), entry[1]]
        for entry in map(ListCodec.coerce, ListCodec.coerce(payload["labels"]))
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert prepare_data.verify(target) == 2


def test_verify_names_a_missing_manifest(tmp_path: Path) -> None:
    """Labels live in ``vae-in/dataset.json`` and nowhere else."""
    target = tmp_path / "corpus"
    (target / "vae-in").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="manifest"):
        _ = prepare_data.verify(target)


def test_verify_catches_an_unlabelled_latent(tmp_path: Path) -> None:
    """A latent with no label would pair against the wrong row."""
    target = tmp_path / "corpus"
    prepare_data.synthesize(
        target,
        samples=3,
        image_size=8,
        latent_size=4,
        latent_channels=8,
        num_classes=10,
        encoder_width=16,
        seed=0,
    )
    manifest = target / "vae-in" / "dataset.json"
    payload = DictCodec.coerce(loads(manifest.read_text(encoding="utf-8")))
    payload["labels"] = ListCodec.coerce(payload["labels"])[:2]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no label"):
        _ = prepare_data.verify(target)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
