"""Experiment factories for the native SpeedrunDiT baseline.

Planned responsibilities:

- define a typed ``SpeedrunDiTLoop`` based on Priml's ``TrainLoop``;
- pre-narrow the concrete ``step`` and ``dataset`` Config slots once on the
  loop, so factories do not need repeated ``isinstance`` checks;
- define the frozen ``exp000`` strong naive control recipe;
- define one-change experiment forks for the SpeedrunDiT mechanisms;
- define the reduced ``exp_smoke`` recipe;
- bind model geometry to the dataset and evaluator configuration through
  lightweight finalization only;
- keep all experiment construction free of filesystem, network, and GPU I/O;
  and
- document each experiment with a hypothesis, references, parent, change,
  budget, and measured results.

The module is the only place where a dataset name, model recipe, or experiment
budget should be assembled.  Reusable behavior belongs in data, model, loss,
train-step, sampler, or metric modules instead.

Factories will construct a config, mutate its full dotted paths, and return
it.  They will not use module-level tunable constants, environment variables,
filesystem reads, network calls, or side effects in ``finalize()``.  Optimizer
configuration will use Priml's ``CompositeOptimizer`` adapter even for one
optimizer, and every schedule horizon will match the loop's bound budget.

Experiment tests will compare the parent's and child's rendered config deltas;
they will not infer the intended change from the implementation after the
fact.
"""
