"""Tests for the SpeedrunDiT flow and objective implementation.

Planned coverage:

- linear path interpolation and time-shift formulas;
- velocity target construction;
- per-sample reduction behavior;
- each optional loss term and its coefficient;
- contrastive batch-roll behavior;
- gradient flow through the model outputs; and
- numerical parity with the upstream loss on fixed inputs.

Closed-form tests will cover the pure formulas.  The complete configured loss
will have a short bit-for-bit fixture when its arithmetic is stable enough to
serve as a portable reference.
"""
