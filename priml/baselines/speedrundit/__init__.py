"""Native Priml port of SpeedrunDiT.

This package will provide a reproducible ImageNet-256 latent image-generation
baseline based on SpeedrunDiT.  The implementation is intentionally planned as
Priml-native components rather than as a vendored copy of the upstream
repository.

The package will eventually expose an ``exp000`` control recipe and subsequent
experiment factories that add the SpeedrunDiT mechanisms one at a time.  The
first implementation milestone is a small ``exp_smoke`` path that can execute
one synthetic training update without ImageNet, INVAE, DINOv2, or a GPU.

This module is documentation-only until the design and upstream audit are
complete.
"""
