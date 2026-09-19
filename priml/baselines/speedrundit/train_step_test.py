"""Tests for the SpeedrunDiT training step.

Planned coverage:

- one synthetic forward/backward/update cycle on CPU;
- optimizer and gradient clipping behavior;
- EMA initialization and update semantics;
- checkpoint save and restore of model, optimizer, and EMA state;
- logging of the named loss terms; and
- a short deterministic numerical golden for the complete update.

GPU, mixed-precision, distributed, and fused-kernel tests should be separate
from the portable CPU correctness tests and explicitly marked.

The portable golden will be minted and replayed through Priml's
``host_agnostic_numerics()`` harness, returning float32 outputs and comparing
bits exactly rather than using tolerance-based comparisons.
"""
