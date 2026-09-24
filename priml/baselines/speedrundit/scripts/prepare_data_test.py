"""Corpus staging: the layout it writes, and the ones it refuses.

Hermetic by parameter rather than by marker: every call passes an explicit
``--directory``, so nothing reaches ``/opt/scratch`` or the network.
"""

from __future__ import annotations

from pathlib import Path

import json

from PIL import Image

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


def write_imagenet(root: Path, *, broken: bool = False) -> None:
    """Write a two-synset extracted ImageNet ``train/`` tree.

    Args:
      root: Destination; ``train/`` is created beneath it.
      broken: Whether the second file of the first synset is undecodable.

    """
    for synset, shade in (("n01440764", 40), ("n01443537", 200)):
        directory = root / "train" / synset
        directory.mkdir(parents=True)
        for name, size in (("10", (48, 32)), ("2", (32, 48))):
            image = Image.new("RGB", size, (shade, 255 - shade, 7))
            image.save(directory / f"{synset}_{name}.JPEG", format="JPEG")
    if broken:
        (root / "train" / "n01440764" / "n01440764_2.JPEG").write_bytes(b"not a jpeg")


def test_convert_writes_the_reference_image_tree(tmp_path: Path) -> None:
    """Sorted-walk numbering, canonical labels, and 256px RGB PNGs."""
    write_imagenet(tmp_path / "imagenet")
    count = prepare_data.convert(tmp_path / "imagenet", tmp_path / "corpus")
    images = tmp_path / "corpus" / "images"
    assert count == 4
    payload = DictCodec.coerce(loads((images / "dataset.json").read_text("utf-8")))
    # ``_10`` sorts before ``_2``: the numbering is the walk's string order.
    assert payload["labels"] == [
        ["00000/img00000000.png", 0],
        ["00000/img00000001.png", 0],
        ["00000/img00000002.png", 1],
        ["00000/img00000003.png", 1],
    ]
    with Image.open(images / "00000" / "img00000000.png") as first:
        assert (first.size, first.mode) == ((256, 256), "RGB")


def test_convert_leaves_the_gap_of_an_undecodable_image(tmp_path: Path) -> None:
    """The reference numbers by position in the walk, so a failure is a gap,
    not a shift that would pair every later image with the wrong index.
    """
    write_imagenet(tmp_path / "imagenet", broken=True)
    count = prepare_data.convert(tmp_path / "imagenet", tmp_path / "corpus")
    images = tmp_path / "corpus" / "images" / "00000"
    assert count == 3
    assert sorted(p.name for p in images.iterdir()) == [
        "img00000000.png",
        "img00000002.png",
        "img00000003.png",
    ]


def test_convert_refuses_a_directory_that_is_not_a_synset(tmp_path: Path) -> None:
    """Labels come from the canonical list; an unknown class has none."""
    write_imagenet(tmp_path / "imagenet")
    cats = tmp_path / "imagenet" / "train" / "cats"
    cats.mkdir()
    Image.new("RGB", (8, 8)).save(cats / "cats_1.JPEG", format="JPEG")
    with pytest.raises(ValueError, match="not synsets"):
        _ = prepare_data.convert(tmp_path / "imagenet", tmp_path / "corpus")
    assert not (tmp_path / "corpus" / "images").exists()


def test_convert_leaves_an_occupied_tree_alone(tmp_path: Path) -> None:
    """Refuse rather than mix two conversions in one tree."""
    write_imagenet(tmp_path / "imagenet")
    images = tmp_path / "corpus" / "images"
    images.mkdir(parents=True)
    (images / "keep.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _ = prepare_data.convert(tmp_path / "imagenet", tmp_path / "corpus")
    assert sorted(p.name for p in images.iterdir()) == ["keep.txt"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
