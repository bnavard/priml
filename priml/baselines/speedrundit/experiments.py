"""Experiment ladder for the native SpeedrunDiT baseline."""

from __future__ import annotations

from dataclasses import field

from configgle import Makes

from priml.baselines.speedrundit.data import SpeedrunDiTData
from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.train_step import SpeedrunDiTTrainStep
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


class SpeedrunDiTLoop(Makes["TrainLoop"], TrainLoop.Config[SpeedrunDiTTrainStep.Config, SpeedrunDiTData.Config]):
    step: SpeedrunDiTTrainStep.Config = field(default_factory=SpeedrunDiTTrainStep.Config)
    dataset: SpeedrunDiTData.Config = field(default_factory=SpeedrunDiTData.Config)


def exp000() -> SpeedrunDiTLoop:
    """Linear-flow SiT control with the upstream AdamW recipe.

    Hypothesis:
      A plain latent SiT with class-token conditioning establishes the
      comparable control before SR-DiT's alignment and routing additions.

    References:
      https://github.com/SwayStar123/SpeedrunDiT
      https://arxiv.org/abs/2512.12386

    Results:
      TBD.
    """
    cfg = SpeedrunDiTLoop()
    cfg.study_name = "speedrundit"
    cfg.experiment_name = "exp000"
    cfg.step.model = SpeedrunDiT.Config()
    cfg.step.loss = SpeedrunDiTLoss.Config()
    cfg.step.loss.projection_coeff = 0.0
    cfg.dataset.batch_size = 8
    cfg.max_steps = cfg.step.total_train_steps = 400_000
    cfg.runtime = SingleProcess.Config()
    return cfg


def exp001() -> SpeedrunDiTLoop:
    """exp000 + representation alignment from REG.

    Hypothesis:
      Aligning intermediate token projections to frozen visual features gives
      the velocity field semantic structure earlier in training.

    References:
      https://arxiv.org/abs/2304.03277

    Results:
      TBD.
    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.loss.projection_coeff = 0.5
    return cfg


def exp002() -> SpeedrunDiTLoop:
    """exp001 + SPRINT sparse token routing."""
    cfg = exp001()
    cfg.experiment_name = "exp002"
    cfg.step.model.sprint = True
    return cfg


def exp003() -> SpeedrunDiTLoop:
    """exp002 + QK normalization configuration seam."""
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.step.model.qk_norm = True
    return cfg


def exp_smoke() -> SpeedrunDiTLoop:
    """Small synthetic one-update run for installation and wiring checks."""
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.max_steps = cfg.step.total_train_steps = 1
    cfg.dataset.synthetic = True
    cfg.dataset.num_samples = 2
    cfg.dataset.batch_size = 2
    cfg.dataset.eval_batch_size = 2
    cfg.dataset.image_size = 4
    cfg.dataset.latent_channels = 4
    cfg.dataset.cls_token_dim = 8
    cfg.step.model = SpeedrunDiT.Config(channels_in=4, channels_out=4, image_size=4, patch_size=2, channels_hidden=16, depth=1, heads=2, cls_token_dim=8, projector_dims=(), projector_hidden=16)
    return cfg
