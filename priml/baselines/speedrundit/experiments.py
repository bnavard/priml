r"""SR-DiT experiment ladder.

``exp000`` REPRODUCES the pinned reference rather than stating a naive recipe
of our own, and is never edited. That is a departure from the usual meaning of
``exp000`` and it is deliberate: the baseline exists to establish that Priml's
components compute the published model exactly, so the control has to be the
published model, SPRINT routing and value residuals included. The bit-for-bit
goldens beside it are the primary artifact, not the FID.

Run::

  uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp000  # noqa: E501

Prepare the corpus first::

  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data  # noqa: E501
"""

from __future__ import annotations

from dataclasses import field
from typing import Final

from configgle import Makes

from priml.baselines.speedrundit.data import SpeedrunDiTData
from priml.baselines.speedrundit.metric import VelocityError
from priml.baselines.speedrundit.train_step import SpeedrunDiTTrainStep
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


__all__ = ["SpeedrunDiTLoop", "exp000", "exp_smoke"]


LATENT_SIZE: Final = 16
"""Side length of an INVAE latent for a 256px image; the tokenizer is f16."""

LATENT_CHANNELS: Final = 32
"""Channels an INVAE latent carries."""

NUM_CLASSES: Final = 1000
"""ImageNet-1k classes."""

ENCODER_WIDTH: Final = 768
"""Feature width of DINOv2 ViT-B/14, the alignment target."""


class SpeedrunDiTLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[SpeedrunDiTTrainStep.Config, SpeedrunDiTData.Config],
):
    """Narrows the loop's two slots to this baseline's concrete configs.

    Narrowed twice over: the generic parameters let a factory read
    ``cfg.step.model`` with no ``isinstance``, and the field redeclarations
    make the runtime defaults the narrow ones. Drop either and the slots
    silently revert to the library's defaults.
    """

    step: SpeedrunDiTTrainStep.Config = field(
        default_factory=SpeedrunDiTTrainStep.Config,
    )
    """What one optimizer update does."""

    dataset: SpeedrunDiTData.Config = field(default_factory=SpeedrunDiTData.Config)
    """Supplies the train and eval loaders."""


def exp000() -> SpeedrunDiTLoop:
    """SR-DiT-B/1 on ImageNet-256 INVAE latents, reproducing the reference.

    Hypothesis:
      Priml's building blocks compute the published SR-DiT exactly. The claim
      under test is numerical identity with the pinned reference across
      initialization, one forward, the objective, and five optimizer steps --
      not a score. A score only becomes meaningful once identity holds.

    References:
      https://github.com/SwayStar123/SpeedrunDiT
        Pinned at c24c2ff25699cce63174ca56c2afcfeeb225e367.
      https://arxiv.org/abs/2512.12386
        Bhanded 2025, "Speedrunning ImageNet Diffusion."
      https://arxiv.org/abs/2410.06940
        Yu et al. 2024, "Representation Alignment for Generation."
      https://arxiv.org/abs/2506.05350
        Stoica et al. 2025, "Contrastive Flow Matching."

    Returns:
      cfg: Training loop for the reference recipe.

    Results:
      TBD.

    """
    cfg = SpeedrunDiTLoop()
    cfg.study_name = "speedrundit"
    cfg.experiment_name = "exp000"

    model = cfg.step.model
    model.channels_in = LATENT_CHANNELS
    model.channels_hidden = 768
    model.image_size = LATENT_SIZE
    model.patch_size = 1
    model.num_layers = 12
    model.heads = 12
    model.num_classes = NUM_CLASSES
    model.class_dropout = 0.1
    model.projector_dims = (ENCODER_WIDTH,)
    model.projector_hidden = 2048

    cfg.dataset.batch_size = 256
    cfg.dataset.latent_scale = 0.3099

    cfg.metrics_eval["velocity"] = VelocityError.Config()

    # A constant rate for the whole run: the reference builds no scheduler, and
    # a warmup here would be a second change riding along with the port.
    cfg.max_steps = cfg.step.train_budget_steps = 400_000
    cfg.num_steps_eval = 10_000
    cfg.runtime = SingleProcess.Config()
    return cfg


def exp_smoke() -> SpeedrunDiTLoop:
    """exp000 at the smallest size that still exercises every path.

    Cuts width, depth, heads, the latent grid, the class count, the batch, and
    the step budget. The SPRINT split is kept at two encoder and two decoder
    layers around one sparse layer, because a trunk short enough to drop the
    sparse stage entirely would stop covering the routing, the mask token, and
    the fusion projection -- which is most of what makes this model unusual.

    Not a result: nothing measured here is comparable with exp000.

    Returns:
      cfg: A tiny loop that runs on CPU without a prepared corpus.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"

    model = cfg.step.model
    model.channels_in = 8
    model.channels_hidden = 64
    model.image_size = 4
    model.num_layers = 5
    model.heads = 4
    model.num_classes = 10
    model.projector_dims = (16,)
    model.projector_hidden = 32

    cfg.dataset.batch_size = 2
    cfg.dataset.eval_batch_size = 2
    cfg.dataset.num_samples = 4

    cfg.step.dtype_autocast = None
    cfg.step.compile = None
    cfg.max_steps = cfg.step.train_budget_steps = 4
    cfg.num_steps_eval = 2
    cfg.checkpointer = None
    return cfg
