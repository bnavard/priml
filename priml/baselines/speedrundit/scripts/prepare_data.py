"""Preparation plan for the SpeedrunDiT ImageNet-256 dataset.

Planned responsibilities:

- convert the supported ImageNet source into the reference resolution and
  crop convention;
- generate or validate INVAE latent files and their scaling metadata;
- optionally precompute DINOv2 representation features for alignment loss;
- write a manifest binding images, latents, labels, and features;
- verify counts, names, dimensions, dtypes, and split identity;
- support idempotent stages rather than silently overwriting prepared data;
  and
- print a resolved preparation configuration suitable for reproducing a run.

The first implementation should make precomputed representation features the
default native Priml seam.  Online encoder extraction may be added later if an
exact throughput reproduction requires it.
"""
