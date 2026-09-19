"""Tests for the SpeedrunDiT experiment ladder.

Planned coverage:

- every factory finalizes without data, network, encoder weights, or a GPU;
- each factory's ``experiment_name`` matches its function name;
- ``exp000`` matches a Configgle readable finalized configuration golden;
- forks change only their documented fields;
- model, latent, class, and evaluator geometry stay consistent;
- schedules and training budgets end at the same horizon;
- ``exp_smoke`` reduces every expensive axis while preserving the recipe; and
- the smallest experiment trains one complete update end to end.

These tests should protect the experiment ladder as a reproducibility contract,
not merely as a collection of constructor tests.  Portable tests must remain
under 100 ms; slow distributed, compiler, GPU, and evaluator tests belong in
explicitly marked tests outside the default tier.
"""
