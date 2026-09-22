"""The held-out velocity metric: weighting, padding, and round-tripping."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.speedrundit.metric import VelocityError


def metric(**overrides: object) -> VelocityError:
    """Build a metric.

    Args:
      **overrides: Config fields to set.

    Returns:
      metric: A built metric.

    """
    config = VelocityError.Config()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_the_mean_is_over_samples_not_batches() -> None:
    """Two batches of unequal size must not weight equally.

    Averaging batch means would let a short final batch count as much as a
    full one, which moves the reported score with the batch size.
    """
    accumulator = metric()
    accumulator.update(torch.tensor([1.0, 1.0, 1.0]), valid_count=3)
    accumulator.update(torch.tensor([5.0]), valid_count=1)
    assert accumulator.compute()["mse"] == pytest.approx(2.0)


def test_padding_rows_are_excluded() -> None:
    """Rows squaring off a short batch must not enter the denominator."""
    accumulator = metric()
    accumulator.update(torch.tensor([4.0, 0.0, 0.0]), valid_count=1)
    assert accumulator.compute()["mse"] == pytest.approx(4.0)


def test_a_missing_valid_count_takes_every_row() -> None:
    """A producer that does not report padding is taken at its word."""
    accumulator = metric()
    accumulator.update(torch.tensor([2.0, 4.0]))
    assert accumulator.compute()["mse"] == pytest.approx(3.0)


def test_a_valid_count_beyond_the_batch_is_clamped() -> None:
    """A count larger than the tensor would index off the end."""
    accumulator = metric()
    accumulator.update(torch.tensor([3.0]), valid_count=8)
    assert accumulator.compute()["mse"] == pytest.approx(3.0)


def test_an_empty_metric_reports_zero() -> None:
    """An eval that saw nothing must not divide by zero."""
    assert metric().compute()["mse"] == 0.0


def test_reset_clears_accumulation() -> None:
    """Each eval starts from nothing, or scores drift across evals."""
    accumulator = metric()
    accumulator.update(torch.tensor([9.0]), valid_count=1)
    accumulator.reset()
    assert accumulator.compute()["mse"] == 0.0


def test_compute_is_idempotent() -> None:
    """Reading the score must not consume it."""
    accumulator = metric()
    accumulator.update(torch.tensor([1.0, 3.0]), valid_count=2)
    assert accumulator.compute() == accumulator.compute()


def test_state_round_trips() -> None:
    """A resumed eval continues the accumulation it was interrupted in."""
    accumulator = metric()
    accumulator.update(torch.tensor([1.0, 3.0]), valid_count=2)
    state = accumulator.state_dict()
    restored = metric()
    restored.load_state_dict(state)
    assert restored.compute() == accumulator.compute()


def test_the_key_is_configurable() -> None:
    """The published name is a value, so two metrics can coexist."""
    accumulator = metric(key="velocity_mse")
    accumulator.update(torch.tensor([1.0]), valid_count=1)
    assert "velocity_mse" in accumulator.compute()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
