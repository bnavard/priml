import torch

from priml.baselines.speedrundit.model_test import _config
from priml.baselines.speedrundit.sampler import EulerMaruyamaSampler


def test_sampler_preserves_latent_shape() -> None:
    model = _config().make()
    sampler = EulerMaruyamaSampler.Config(num_steps=2).make()
    x = torch.randn(2, 4, 4, 4)
    result = sampler(model, x, torch.zeros(2, dtype=torch.long), torch.randn(2, 8))
    assert result.shape == x.shape

