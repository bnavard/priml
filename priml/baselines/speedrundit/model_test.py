import torch

from priml.baselines.speedrundit.model import SpeedrunDiT


def _config() -> SpeedrunDiT.Config:
    return SpeedrunDiT.Config(channels_in=4, channels_out=4, image_size=4, patch_size=2, channels_hidden=16, depth=1, heads=2, cls_token_dim=8, projector_dims=(8,), projector_hidden=16)


def test_forward_shapes() -> None:
    model = _config().make()
    velocity, projections, cls = model(torch.randn(2, 4, 4, 4), torch.rand(2), torch.zeros(2, dtype=torch.long), cls_token=torch.randn(2, 8))
    assert velocity.shape == (2, 4, 4, 4)
    assert projections[0].shape == (2, 5, 8)
    assert cls.shape == (2, 8)

