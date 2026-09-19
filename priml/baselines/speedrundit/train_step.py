"""Priml training step for latent SpeedrunDiT training.

Planned responsibilities:

- define ``SpeedrunDiTTrainStep`` and its typed configuration;
- consume the dataset sample contract without performing data preparation;
- call the model and loss modules;
- combine configured objective terms;
- apply mixed precision, gradient scaling, clipping, and optimizer updates
  through Priml's training abstractions;
- maintain and checkpoint the EMA model state;
- expose scalar loss-term and gradient metrics; and
- support a tiny deterministic update suitable for numerical golden tests.

The train step must not own command-line parsing, distributed process setup,
ImageNet discovery, or external evaluator execution.  Its public output will
follow Priml's ``TrainStepOutput`` contract: at minimum a ``loss`` tensor and
the model state needed by the loop.  Optimizer, schedule, loss, precision,
parallelism, and checkpointable EMA behavior will be injectable Config slots;
the step will not branch on a closed set of optimizer or architecture strings.
"""
