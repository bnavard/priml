import torch

from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model_test import _config


def test_linear_flow_loss_is_finite() -> None:
    model = _config().make()
    loss = SpeedrunDiTLoss.Config(projection_coeff=0).make()
    batch = dict(media=torch.randn(2, 4, 4, 4), label=torch.zeros(2, dtype=torch.long), cls_token=torch.randn(2, 8), features=[])
    result = loss(model, **batch, time=torch.tensor([0.25, 0.75]), noise=torch.zeros_like(batch["media"]))
    assert result["loss"].shape == (2,)
    assert torch.isfinite(result["loss"]).all()

