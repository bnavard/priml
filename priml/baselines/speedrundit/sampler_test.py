"""Tests for SpeedrunDiT latent sampling.

Planned coverage:

- deterministic sampling with a fixed seed;
- sampler step count and endpoint behavior;
- conditioning and guidance shape contracts;
- latent-to-image decode shape and range; and
- evaluator batch serialization without requiring the full ImageNet dataset.
"""
