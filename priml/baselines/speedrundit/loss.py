"""SpeedrunDiT objective and flow-path mathematics.

Planned responsibilities:

- sample flow times and Gaussian noise;
- apply the linear interpolation path between clean latent and noise;
- construct the upstream velocity targets;
- implement optional time shifting;
- compute the latent denoising loss;
- compute representation-alignment loss against frozen reference features;
- compute the class-token objective;
- compute contrastive flow-matching loss; and
- return named per-sample terms so the train step can log and weight them.

The module will compare its flow-path and target conventions against Priml's
existing diffusion utilities.  Shared math should be reused when it is exactly
equivalent; SpeedrunDiT-specific behavior should remain visible here rather
than hidden in the training loop.  Objective choices, coefficients, reductions,
and weighting functions will be Config or callable slots so they appear in
``pprint()``, can be overridden, and can be changed by a fork.  A bare function
with hidden keyword arguments is not an acceptable experiment boundary.
"""
