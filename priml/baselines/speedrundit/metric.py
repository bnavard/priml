"""Metrics and evaluation adapters for SpeedrunDiT.

Planned responsibilities:

- define lightweight training-time metrics such as objective terms and sample
  counts;
- define a configuration object for expensive generative evaluation;
- preserve the upstream reference-batch and sample-batch conventions;
- report FID, sFID, IS, precision, and recall when the evaluator dependencies
  and reference data are available; and
- clearly distinguish smoke metrics from benchmark-quality metrics.

Expensive external evaluation must be explicitly marked and must not run in
the default CPU unit-test tier.
"""
