"""Latent sampling and image decoding for SpeedrunDiT.

Planned responsibilities:

- implement the supported Euler-Maruyama or equivalent upstream sampler;
- expose the number and schedule of sampling steps as configuration;
- apply class conditioning and optional classifier-free guidance according to
  the reference behavior;
- decode generated INVAE latents;
- save reproducible image grids and evaluator-compatible ``.npz`` batches; and
- provide small deterministic sampling tests independent of ImageNet quality.

Sampling is intentionally separate from the training loss and train step so
that evaluation can evolve without changing the training recipe.
"""
