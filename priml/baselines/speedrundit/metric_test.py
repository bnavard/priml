import torch

from priml.baselines.speedrundit.metric import MeanMetric


def test_mean_metric() -> None:
    metric = MeanMetric.Config(name="denoising").make()
    metric.update(torch.tensor([1.0, 3.0]))
    assert metric.compute() == {"denoising": 2.0}

