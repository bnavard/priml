import torch

from priml.baselines.speedrundit.data import SpeedrunDiTData
from priml.baselines.speedrundit.experiments import exp_smoke


def test_synthetic_train_step() -> None:
    cfg = exp_smoke()
    step = cfg.step.make()
    data = cfg.dataset.make()
    step.train_step(**next(data.train_dataloader()))
    assert torch.isfinite(step.train_step(**next(data.train_dataloader()))["loss"]).all()

