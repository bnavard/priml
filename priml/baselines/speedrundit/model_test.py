"""Structure, routing, and initialization contracts of the SR-DiT model."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from torch import Tensor

from priml.baselines.speedrundit.model import (
    SpeedrunDiT,
    SprintRouting,
    VisionRoPE,
    sincos_position_table,
)


# One named constant per axis; the grid is what drives sequence length, and
# sequence length is quadratic in attention.
GRID: Final = 4
CHANNELS: Final = 8
HIDDEN: Final = 64
HEADS: Final = 4
LAYERS: Final = 6
CLASSES: Final = 10
TARGET: Final = 16


def tiny() -> SpeedrunDiT.Config:
    """Build the model at the smallest size that covers every path.

    Returns:
      cfg: A shrunk config; the recipe is untouched.

    """
    cfg = SpeedrunDiT.Config()
    cfg.channels_in = CHANNELS
    cfg.channels_hidden = HIDDEN
    cfg.image_size = GRID
    cfg.patch_size = 1
    cfg.num_layers = LAYERS
    cfg.heads = HEADS
    cfg.num_classes = CLASSES
    cfg.projector_dims = (TARGET,)
    cfg.projector_hidden = 32
    return cfg


def batch(size: int = 2) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build one input batch.

    Args:
      size: Samples in the batch.

    Returns:
      media: Latents.
      time: Flow times.
      label: Class indices.
      cls_token: Class features.

    """
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(size, CHANNELS, GRID, GRID, generator=generator),
        torch.rand(size, generator=generator),
        torch.randint(CLASSES, (size,), generator=generator),
        torch.randn(size, TARGET, generator=generator),
    )


def test_forward_shapes_follow_the_latent_geometry() -> None:
    """The velocity comes back in the space the latent came from."""
    model = tiny().make()
    media, time, label, cls_token = batch()
    out = model(media, time, label, cls_token)
    assert out.velocity.shape == media.shape
    assert out.cls_velocity.shape == cls_token.shape
    assert len(out.projections) == 1
    assert out.projections[0].shape == (2, 1 + GRID * GRID, TARGET)


def test_the_first_block_has_no_value_residual() -> None:
    """Block zero has no earlier layer to blend toward.

    A residual there would either blend a layer with itself or read an
    uninitialized tensor, and both train without ever raising.
    """
    model = tiny().make()
    assert model.blocks[0].attn.value_residual is None
    assert all(block.attn.value_residual is not None for block in model.blocks[1:])


def test_initialization_is_reproducible_under_a_seed() -> None:
    """Two builds under one seed agree bit for bit.

    This is what the bit-for-bit goldens rest on: construction order fixes the
    RNG draw order, so a reordered constructor fails here first.
    """
    torch.manual_seed(7)
    first = tiny().make()
    torch.manual_seed(7)
    second = tiny().make()
    for (name, left), (_, right) in zip(
        first.named_parameters(),
        second.named_parameters(),
        strict=True,
    ):
        assert torch.equal(left, right), name


def test_gated_sublayers_start_as_identity() -> None:
    """Zeroed modulation makes every block a no-op at step zero.

    The readout is zeroed too, so the very first velocity is exactly zero and
    the model starts from the data rather than from noise it invented.
    """
    model = tiny().make()
    for block in model.blocks:
        assert torch.all(block.modulation.modulation[-1].weight == 0)
        assert torch.all(block.modulation.modulation[-1].bias == 0)
    assert torch.all(model.final_layer.linear.weight == 0)
    media, time, label, cls_token = batch()
    out = model(media, time, label, cls_token)
    assert torch.all(out.velocity == 0)


def test_position_table_leaves_the_class_row_zero() -> None:
    """The class token carries no spatial position.

    Giving it one would place it at patch (0, 0) and make the corner patch and
    the class token indistinguishable to the first block.
    """
    table = sincos_position_table(HIDDEN, GRID, lead=1)
    assert table.shape == (1 + GRID * GRID, HIDDEN)
    assert torch.all(table[0] == 0)
    assert torch.any(table[1] != 0)


def test_rope_rotates_the_whole_head_and_spares_the_lead() -> None:
    """Leading tokens pass through unrotated; the rest do not."""
    cfg = VisionRoPE.Config(channels_head=HIDDEN // HEADS, grid=GRID)
    rope = cfg.make()
    tokens = 1 + GRID * GRID
    x = torch.randn(2, HEADS, tokens, HIDDEN // HEADS)
    positions = torch.arange(GRID * GRID).view(1, -1).expand(2, -1)
    rotated = rope(x, positions)
    assert torch.equal(rotated[:, :, 0, :], x[:, :, 0, :])
    # Position zero is the identity rotation, so the first PATCH is untouched
    # too; the second must move.
    assert not torch.equal(rotated[:, :, 2, :], x[:, :, 2, :])


def test_sprint_keeps_a_quarter_of_the_tokens_while_training() -> None:
    """Routing drops tokens in training and no tokens in eval."""
    cfg = SprintRouting.Config(channels_in=HIDDEN)
    routing = cfg.make()
    x = torch.randn(3, 20, HIDDEN)
    routing.train()
    plan = routing.plan(x)
    assert plan.keep is not None
    assert plan.tokens.shape == (3, 5, HIDDEN)
    routing.eval()
    assert routing.plan(x).keep is None


def test_sprint_scatter_restores_position() -> None:
    """A token put back lands where it was taken from.

    Scattering to the wrong slot would leave the fusion reading a patch's
    representation at another patch's position, which trains and degrades.
    """
    cfg = SprintRouting.Config(channels_in=4)
    routing = cfg.make().train()
    x = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    plan = routing.plan(x)
    assert plan.keep is not None
    restored = routing.scatter(plan.tokens, plan.keep, 6)
    for row in range(2):
        for slot, index in enumerate(plan.keep[row].tolist()):
            assert torch.equal(restored[row, index], plan.tokens[row, slot])


def test_eval_draws_no_randomness() -> None:
    """An eval forward must not move the RNG.

    Label dropout, token drop and path drop are all training-only; if any
    leaked, two evals of one checkpoint would disagree.
    """
    model = tiny().make().eval()
    media, time, label, cls_token = batch()
    before = torch.get_rng_state()
    first = model(media, time, label, cls_token)
    middle = torch.get_rng_state()
    second = model(media, time, label, cls_token)
    assert torch.equal(before, middle)
    assert torch.equal(torch.get_rng_state(), middle)
    assert torch.equal(first.velocity, second.velocity)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("image_size", 3, "divisible"),
        ("heads", 7, "divisible"),
        ("projector_dims", (), "at least one"),
        ("channels_decoder", 128, "channels_decoder"),
    ],
)
def test_invalid_geometry_is_refused(field: str, value: object, match: str) -> None:
    """A misshaped config fails at finalize, naming the field.

    Args:
      field: Config field to corrupt.
      value: The offending value.
      match: Substring the message must carry.

    """
    cfg = tiny()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        _ = cfg.copy_tree().finalize()


def test_sprint_cannot_reserve_more_layers_than_the_trunk_holds() -> None:
    """A split deeper than the trunk is a configuration error, not a clamp."""
    cfg = tiny()
    cfg.num_layers = 3
    assert cfg.sprint is not None
    cfg.sprint.num_encoder_layers = 2
    cfg.sprint.num_decoder_layers = 2
    with pytest.raises(ValueError, match="reserves"):
        _ = cfg.copy_tree().finalize()


def test_disabling_sprint_runs_every_block_densely() -> None:
    """Without routing the trunk is a plain stack, and still produces a
    velocity of the right shape."""
    cfg = tiny()
    cfg.sprint = None
    model = cfg.make()
    assert model.sprint is None
    media, time, label, cls_token = batch()
    assert model(media, time, label, cls_token).velocity.shape == media.shape


def test_parameter_count_is_stable_across_a_rebuild() -> None:
    """The same config builds the same parameter inventory twice."""
    first = tiny().make()
    second = tiny().make()
    left = [(n, tuple(p.shape)) for n, p in first.named_parameters()]
    right = [(n, tuple(p.shape)) for n, p in second.named_parameters()]
    assert left == right


def test_the_frozen_position_table_is_not_trained() -> None:
    """``pos_embed`` is a parameter so it rides in the checkpoint, but it must
    never receive a gradient."""
    model = tiny().make()
    assert not model.pos_embed.requires_grad
    media, time, label, cls_token = batch()
    out = model(media, time, label, cls_token)
    (out.velocity.sum() + out.cls_velocity.sum()).backward()
    assert model.pos_embed.grad is None


def test_every_trainable_parameter_receives_a_gradient() -> None:
    """A parameter with no gradient is dead weight the optimizer still carries.

    The mask token is the one that hides: it only receives a gradient through
    the padded slots, which exist only when routing actually drops something.
    """
    torch.manual_seed(0)
    model = tiny().make().train()
    media, time, label, cls_token = batch()
    out = model(media, time, label, cls_token)
    loss = out.velocity.square().mean() + out.cls_velocity.square().mean()
    loss = loss + sum(p.square().mean() for p in out.projections)
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
