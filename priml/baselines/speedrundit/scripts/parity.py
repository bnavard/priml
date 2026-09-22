#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Measure this port against the pinned upstream SpeedrunDiT, bit for bit.

Clones the reference at a pinned commit, imports it UNMODIFIED, and compares
four checkpoints with exact tensor equality inside Priml's host-agnostic
numeric context: the loader, initialization, one forward, the objective, and
five optimizer steps.

Nothing here is a unit test. It needs a network, a git clone, and several
minutes, and it establishes the port ONCE against a moving upstream; the
bit-for-bit goldens under ``testdata/`` are what keep it frozen afterwards. So
this module is never imported by the library, and the library never imports it.

The comparison is deliberately one-substitution: both sides run the same
geometry, the same seed, and the same input tensors, and neither is adjusted
to make the other agree. Where a difference is structural rather than
numerical -- the port names value-residual parameters differently, and Priml's
EMA declines to average frozen tensors -- it is REPORTED under its own heading
rather than normalized away.

Examples:
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.parity  # noqa: E501
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.parity --upstream /path/to/SpeedrunDiT  # noqa: E501
'''
# fmt: on

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import json
import subprocess
import sys
import tempfile

import numpy as np

from torch import Tensor, nn

import torch

from priml.baselines.speedrundit.data import SpeedrunDiTData
from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


SOURCE_URL: Final = "https://github.com/SwayStar123/SpeedrunDiT.git"
"""Reference repository."""

SOURCE_REVISION: Final = "c24c2ff25699cce63174ca56c2afcfeeb225e367"
"""Pinned reference commit; every golden in testdata/ was measured against it."""

STEPS: Final = 5
"""Optimizer steps compared."""

# Small enough to run in seconds, wide enough that no axis is degenerate: two
# heads would hide a head-axis transpose, and a depth under five would drop
# SPRINT's sparse stage entirely.
GEOMETRY: Final = {
    "input_size": 4,
    "patch_size": 1,
    "in_channels": 8,
    "hidden_size": 64,
    "decoder_hidden_size": 64,
    "depth": 6,
    "num_heads": 4,
    "encoder_depth": 2,
    "num_classes": 10,
    "class_dropout_prob": 0.1,
    "use_cfg": True,
    "z_dims": [16],
    "projector_dim": 32,
    "cls_token_dim": 16,
}


class _Flags(Protocol):
    """Parsed command line."""

    upstream: Path | None
    keep: bool


def native_config() -> SpeedrunDiT.Config:
    """Build the port at the comparison geometry.

    Returns:
      cfg: A config matching :data:`GEOMETRY`.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = GEOMETRY["in_channels"]
    cfg.channels_hidden = GEOMETRY["hidden_size"]
    cfg.image_size = GEOMETRY["input_size"]
    cfg.patch_size = GEOMETRY["patch_size"]
    cfg.num_layers = GEOMETRY["depth"]
    cfg.heads = GEOMETRY["num_heads"]
    cfg.num_classes = GEOMETRY["num_classes"]
    cfg.class_dropout = GEOMETRY["class_dropout_prob"]
    cfg.projector_dims = (GEOMETRY["z_dims"][0],)
    cfg.projector_hidden = GEOMETRY["projector_dim"]
    return cfg


@contextmanager
def pinned_source(explicit: Path | None) -> Generator[Path]:
    """Yield a checkout of the reference at :data:`SOURCE_REVISION`.

    An explicit path is used as given and NOT verified, so a local working
    copy can be measured; a clone is pinned by SHA and the SHA is read back,
    which catches a moved ref rather than trusting the name.

    Args:
      explicit: An existing checkout, or ``None`` to clone one.

    Yields:
      root: Directory holding the reference.

    Raises:
      RuntimeError: If a fresh clone does not land on the pinned commit.

    """
    if explicit is not None:
        yield explicit.expanduser().resolve()
        return
    with tempfile.TemporaryDirectory(prefix="speedrundit-parity-") as name:
        root = Path(name) / "srdit"
        subprocess.run(  # noqa: S603 -- Fixed executable and a constant URL.
            ["git", "clone", "--quiet", "--no-checkout", SOURCE_URL, str(root)],  # noqa: S607
            check=True,
        )
        subprocess.run(  # noqa: S603 -- Fixed executable, constant revision.
            ["git", "-C", str(root), "checkout", "--quiet", SOURCE_REVISION],  # noqa: S607
            check=True,
        )
        head = subprocess.run(  # noqa: S603 -- Fixed executable.
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != SOURCE_REVISION:
            raise RuntimeError(
                f"clone landed on {head}, expected {SOURCE_REVISION}.",
            )
        yield root


def report(label: str, ok: bool, detail: str = "") -> bool:
    """Print one comparison line.

    Args:
      label: What was compared.
      ok: Whether it matched.
      detail: Extra text shown on failure.

    Returns:
      ok: The value passed in, so callers can accumulate.

    """
    mark = "OK  " if ok else "FAIL"
    suffix = f"  {detail}" if detail and not ok else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def tensors_equal(left: Tensor, right: Tensor) -> tuple[bool, str]:
    """Compare two tensors exactly, reporting ULPs when they differ.

    Args:
      left: Reference tensor.
      right: Candidate tensor.

    Returns:
      ok: Whether every bit matches.
      detail: A description of the largest disagreement.

    """
    if left.shape != right.shape:
        return False, f"shape {tuple(left.shape)} != {tuple(right.shape)}"
    if left.dtype != right.dtype:
        return False, f"dtype {left.dtype} != {right.dtype}"
    if torch.equal(left, right):
        return True, ""
    diff = (left.double() - right.double()).abs().max().item()
    return False, f"maxabs={diff:.6e}"


def compare_named(
    left: Sequence[tuple[str, Tensor]],
    right: Sequence[tuple[str, Tensor]],
    label: str,
) -> bool:
    """Compare two parameter lists positionally.

    Positional rather than by name: the port names value-residual parameters
    ``attn.value_residual.weight`` where the reference names them
    ``attn.v1_lambda``, and the construction order is the thing under test, so
    position is the honest key.

    Args:
      left: Reference ``(name, tensor)`` pairs.
      right: Candidate pairs.
      label: Heading for the report line.

    Returns:
      ok: Whether every tensor matched.

    """
    if len(left) != len(right):
        return report(
            label,
            ok=False,
            detail=f"{len(left)} tensors against {len(right)}",
        )
    bad: list[str] = []
    for (left_name, left_value), (right_name, right_value) in zip(
        left,
        right,
        strict=True,
    ):
        ok, detail = tensors_equal(left_value, right_value)
        if not ok:
            bad.append(f"{left_name} vs {right_name}: {detail}")
    if bad:
        return report(label, ok=False, detail=f"{len(bad)} differ; first: {bad[0]}")
    return report(label, ok=True)


def synthetic_corpus(root: Path, *, count: int) -> None:
    """Write a corpus in the reference's on-disk layout.

    Args:
      root: Destination directory.
      count: Samples to write.

    """
    images = root / "images" / "00000"
    latents = root / "vae-in" / "00000"
    images.mkdir(parents=True)
    latents.mkdir(parents=True)
    labels: list[list[object]] = []
    rng = np.random.default_rng(0)
    for index in range(count):
        stem = f"{index:08d}"
        np.save(images / f"img{stem}.npy", rng.integers(0, 255, (3, 4, 4), np.uint8))
        latent = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
        np.save(latents / f"img-latents-{stem}.npy", latent)
        labels.append([f"00000/img-latents-{stem}.npy", index % 10])
    manifest = {"labels": labels}
    (root / "vae-in" / "dataset.json").write_text(json.dumps(manifest), "utf-8")


def compare_loader(upstream: Path) -> bool:
    """Compare the loader against the reference's dataset.

    Runs both over one synthetic tree in the reference's own layout, so the
    enumeration, the sort, the label join, and the latent scaling are all
    exercised on real files rather than asserted from the source.

    Args:
      upstream: Reference checkout.

    Returns:
      ok: Whether the loaders agree.

    """
    print("loader")
    from torch.utils.data import DataLoader  # noqa: PLC0415 -- Script-local.

    count = 6
    with tempfile.TemporaryDirectory(prefix="speedrundit-corpus-") as name:
        root = Path(name)
        synthetic_corpus(root, count=count)
        sys.path.insert(0, str(upstream))
        dataset = __import__("dataset").CustomDataset(str(root))
        loader = DataLoader(dataset, batch_size=count, shuffle=False)
        raw_image, latent, label = next(iter(loader))

        config = SpeedrunDiTData.Config()
        config.base_dir = "/"
        config.working_dir = str(root)
        config.batch_size = count
        config.eval_batch_size = count
        config.device = "cpu"
        config.keep_images = True
        # Compare the loader's own arithmetic, not the tokenizer constant.
        config.latent_scale = 1.0
        data = config.make()
        batch = next(iter(data.eval_dataloader()))

        ok = report("order and labels", *_swap(tensors_equal(label, batch["label"])))
        ok &= report(
            "latents",
            *_swap(tensors_equal(latent.squeeze(1), batch["media"])),
        )
        ok &= report(
            "images",
            *_swap(tensors_equal(raw_image, batch["raw_image"])),
        )
        return ok


def _swap(result: tuple[bool, str]) -> tuple[bool, str]:
    """Adapt ``tensors_equal`` to :func:`report`'s argument order.

    Args:
      result: ``(ok, detail)``.

    Returns:
      arguments: The same pair, for splatting into :func:`report`.

    """
    return result


def build_batches(count: int) -> list[dict[str, Tensor]]:
    """Draw the inputs both sides consume.

    Args:
      count: Batches to build.

    Returns:
      batches: Fixed inputs, identical for both implementations.

    """
    generator = torch.Generator().manual_seed(99)
    batches: list[dict[str, Tensor]] = []
    for _ in range(count):
        batches.append(
            {
                "media": torch.randn(2, 8, 4, 4, generator=generator),
                "label": torch.randint(10, (2,), generator=generator),
                "cls_token": torch.randn(2, 16, generator=generator),
                "features": torch.randn(2, 17, 16, generator=generator),
            },
        )
    return batches


def run_upstream(model: nn.Module, batches: Sequence[dict[str, Tensor]]) -> list[dict]:
    """Drive the reference for :data:`STEPS` optimizer steps.

    Args:
      model: The reference model.
      batches: Fixed inputs.

    Returns:
      trace: Per-step losses, gradients, and post-step weights.

    """
    from loss import SILoss  # noqa: PLC0415 -- Script-local, from the clone.

    objective = SILoss(
        path_type="linear",
        weighting="uniform",
        cfm_weighting="uniform",
        apply_time_shift=True,
        shift_base=4096,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )
    model.train()
    trace: list[dict] = []
    for batch in batches:
        denoise, proj, time, noise, cls, cfm, _cfm_cls = objective(
            model,
            batch["media"],
            {"y": batch["label"]},
            zs=[batch["features"]],
            cls_token=batch["cls_token"],
        )
        loss = (
            denoise.mean()
            + 0.5 * proj.mean()
            + 0.03 * cls.mean()
            + 0.05 * cfm.mean()
        )
        loss.backward()
        grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        _ = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trace.append(
            {
                "loss": loss.detach().clone(),
                "time": time.detach().clone(),
                "noise": noise.detach().clone(),
                "grads": grads,
                "weights": [
                    (n, p.detach().clone()) for n, p in model.named_parameters()
                ],
            },
        )
    return trace


def run_native(model: SpeedrunDiT, batches: Sequence[dict[str, Tensor]]) -> list[dict]:
    """Drive the port for :data:`STEPS` optimizer steps.

    Args:
      model: The ported model.
      batches: Fixed inputs.

    Returns:
      trace: Per-step losses, gradients, and post-step weights.

    """
    objective = SpeedrunDiTLoss.Config().make()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )
    model.train()
    trace: list[dict] = []
    for batch in batches:
        result = objective(
            model,
            media=batch["media"],
            label=batch["label"],
            cls_token=batch["cls_token"],
            features=[batch["features"]],
        )
        result.loss.backward()
        grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        _ = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trace.append(
            {
                "loss": result.loss.detach().clone(),
                "time": result.time.detach().clone(),
                "noise": result.noise.detach().clone(),
                "grads": grads,
                "weights": [
                    (n, p.detach().clone()) for n, p in model.named_parameters()
                ],
            },
        )
    return trace


def main() -> int:
    """Run every comparison and report.

    Returns:
      status: Zero when every checkpoint matched.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help="Existing reference checkout; omit to clone the pinned commit.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Print every differing tensor rather than the first.",
    )
    flags = cast(_Flags, parser.parse_args())

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    with pinned_source(flags.upstream) as upstream:
        sys.path.insert(0, str(upstream))
        ok = compare_loader(upstream)

        from models.sit import SiT  # noqa: PLC0415 -- Script-local, from the clone.

        print("initialization")
        torch.manual_seed(1234)
        reference = SiT(qk_norm=True, fused_attn=True, **GEOMETRY)
        torch.manual_seed(1234)
        candidate = native_config().make()
        ok &= compare_named(
            list(reference.named_parameters()),
            list(candidate.named_parameters()),
            "parameters after init",
        )

        batches = build_batches(STEPS)
        print(f"{STEPS} training steps")
        torch.manual_seed(777)
        with host_agnostic_numerics():
            left = run_upstream(reference, batches)
        torch.manual_seed(777)
        with host_agnostic_numerics():
            right = run_native(candidate, batches)

        for index, (a, b) in enumerate(zip(left, right, strict=True), start=1):
            for key in ("time", "noise", "loss"):
                same = tensors_equal(a[key], b[key])
                ok &= report(f"step {index} {key}", *_swap(same))
            ok &= compare_named(a["grads"], b["grads"], f"step {index} gradients")
            ok &= compare_named(a["weights"], b["weights"], f"step {index} weights")

    print()
    print("known structural differences, not normalized away:")
    print("  - value-residual parameters are named attn.value_residual.weight")
    print("    here and attn.v1_lambda upstream; compared positionally.")
    print("  - Priml's EMA declines to average frozen tensors, so the shadow")
    print("    of pos_embed does not drift as the reference's does. The model")
    print("    weights above are unaffected; the EMA shadow is not compared.")
    print()
    print("PARITY HOLDS" if ok else "PARITY BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
