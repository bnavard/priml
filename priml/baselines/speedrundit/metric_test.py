"""Tests for SpeedrunDiT metrics and evaluation adapters.

Planned coverage:

- metric configuration and state construction;
- sample/reference shape validation;
- deterministic toy-feature metric behavior;
- correct aggregation across batches; and
- explicit skipping or marking of external FID-family evaluation when its
  evaluator, reference batch, or GPU requirements are unavailable.
"""
