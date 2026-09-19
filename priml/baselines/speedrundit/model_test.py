"""Tests for the configurable SpeedrunDiT model.

Planned coverage:

- patchify/unpatchify shape and ordering contracts;
- timestep and class-conditioning shapes;
- output shapes for velocity, projection, and class-token predictions;
- each optional architectural feature in isolation;
- combinations used by the experiment ladder;
- parameter-count and geometry sanity checks; and
- focused upstream parity tests on fixed synthetic tensors where exact
  equality is intended.

Model initialization and forward behavior will have separate bit-for-bit
goldens where the computation is intended to be host-independent.  Tests will
exercise gradients as well as values for numerically delicate operations.
"""
