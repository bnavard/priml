"""Tests for the SpeedrunDiT prepared-data contract.

Planned coverage:

- synthetic samples have the documented image, latent, label, and feature
  shapes;
- manifest labels and feature files remain aligned;
- malformed or mismatched samples fail with actionable errors;
- resolution and latent scaling metadata are validated;
- dataset iteration is deterministic under a fixed seed; and
- the dataset can be exercised entirely from temporary test fixtures.

Tests will shrink only size and resource fields.  They will not substitute a
different optimizer, loss, schedule, or data semantics merely to make the
test pass quickly.
"""
