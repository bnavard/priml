"""Data configuration and loading seam for the SpeedrunDiT baseline.

Planned responsibilities:

- define a ``SpeedrunDiTData`` dataset and its typed ``Config``;
- expose explicit ``base_dir`` and logical ``working_dir`` fields, resolving
  paths during lightweight finalization;
- read the prepared ImageNet-256 directory layout without changing sample
  ordering or label association;
- load image tensors, INVAE latents, labels, and optional precomputed
  representation features;
- validate resolution, latent channels, spatial size, dtype, and manifest
  consistency;
- provide deterministic synthetic fixtures for smoke tests; and
- keep dataset preparation concerns separate from experiment construction.

The Config will own every data-affecting choice that must be reproducible:
split identity, resolution, latent layout, feature layout, batch size, loader
workers, shuffling policy, and validation behavior.  Dataset geometry will be
propagated from the model where it is derived rather than duplicated as
independent tunables.

The module must not download ImageNet, load DINOv2 or INVAE encoders, or scan
the filesystem while an experiment configuration is being built.  Disk and
network access belong only to the constructed dataset or preparation script.
"""
