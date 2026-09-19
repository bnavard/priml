"""Configurable latent transformer model for SpeedrunDiT.

Planned responsibilities:

- define the SiT-B/1-style latent patchification and unpatchification;
- embed flow time and class labels;
- implement the transformer backbone and output head;
- expose configuration slots for RMSNorm, RoPE, QK normalization, value
  residual learning, and SPRINT-style token routing;
- expose child modules as Config slots rather than flattening child internals
  into scalar parent fields;
- propagate boundary shape nouns such as ``channels_in`` and ``depth`` through
  checked Priml protocols;
- return the velocity prediction, representation projections, and class-token
  prediction required by the training objective; and
- make tensor shapes and conditioning semantics explicit in the public model
  contract.

The implementation should reuse Priml transformer, normalization, rotary
embedding, initialization, and attention components wherever their semantics
match the upstream reference.  Any deliberate numerical divergence must be
covered by a focused test and documented in the experiment README.

The model will remain stateful model-layer code.  Pure patch, time, and tensor
transformations will either use existing ``priml/math`` functions or stay as
small implementation details until a general second caller justifies
promotion.
"""
