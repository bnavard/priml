# SpeedrunDiT

Native Priml port of [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT),
an ImageNet latent diffusion/flow-matching baseline built around a SiT-B/1
DiT-style model.

This package is currently a documentation-only skeleton.  No implementation is
present yet.  The purpose of this directory is to make the intended shape of
the baseline concrete before porting numerical behavior.

The design follows Priml's four constraints: the complete run is one
hierarchical config tree, changing behavior happens through injectable slots,
configuration construction is hermetic, and the finalized tree is printable.
Anything that affects a result must therefore appear in the config and in the
experiment's rendered configuration.  Dataset staging, encoder loading, and
evaluation side effects must not happen during `finalize()`.

## Intended scope

The first complete version will target ImageNet-256 and will support:

1. prepared ImageNet images, INVAE latents, class labels, and representation
   features;
2. a configurable SiT-B/1-style latent transformer;
3. velocity prediction on a linear flow path;
4. representation-alignment, denoising, class, and contrastive flow-matching
   loss terms;
5. EMA training and checkpointing;
6. latent sampling and INVAE decoding;
7. reference-compatible FID-family evaluation; and
8. a small synthetic smoke experiment that runs without external data.

The implementation will use Priml's configuration, train-loop, distributed
runtime, checkpointing, model, loss, and diffusion abstractions.  The upstream
repository is the behavioral and numerical reference, not a dependency to be
imported wholesale.

## Priml boundaries

The package will keep behavior local until a second caller or a clearly
general primitive justifies promotion into `priml/`:

- `data.py` owns disk-backed input and preparation metadata;
- `model.py` owns stateful neural modules and their Configs;
- `loss.py` owns configgleable objective composition, while broadly reusable
  pure diffusion math belongs in `priml/math/diffusion`;
- `train_step.py` owns run-time optimization, RNG, and train-step state;
- `sampler.py` owns inference-time generation and decoding;
- `metric.py` owns stateful metric accumulation;
- `experiments.py` owns only dataset-specific recipe assembly; and
- tests and `testdata/` remain test-only and are never imported by production.

The implementation will use Priml's blessed names such as `channels_in`,
`channels_out`, `channels_hidden`, `heads`, `norm`, `norm_qk`, `norm_out`,
`device`, `dtype`, `working_dir`, `max_steps`, and `seed`.  It will prefer
Config or callable slots over mode strings, enum-like flags, hidden sentinels,
module globals, and environment reads.

Training batches will use Priml's standard `media` and `label` keys.  Auxiliary
inputs such as latents and representation features must be explicitly named
and documented in the batch type rather than smuggled through global state.

## Planned experiment ladder

The exact ladder will be finalized after auditing the upstream repository and
its REG ablation history.  The current working proposal is:

| Experiment | Planned purpose |
|---|---|
| `exp000` | Strong naive SiT-B/1 + INVAE latent-space flow-matching control |
| `exp001` | Add representation alignment / REG-style loss |
| `exp002` | Add SPRINT token routing |
| `exp003` | Add RMSNorm, RoPE, and QK normalization |
| `exp004` | Add value residual learning |
| `exp005` | Add contrastive flow matching |
| `exp006` | Add time shifting |
| `exp007` | Full composed SpeedrunDiT recipe |
| `exp_smoke` | Tiny synthetic end-to-end validation recipe |

These names are provisional.  `exp000` is not a toy or a reduced smoke model:
it is the strongest straightforward recipe that uses no SpeedrunDiT-specific
exotica.  Each published experiment must have a clear named parent, one
attributable change, a hypothesis, references, and measured results.  An
optimizer and the schedule it prescribes may count as one inseparable change;
otherwise the fork must change one config path.  `exp000` will be frozen once
its recipe is established.  `seed` remains at its default unless an experiment
studies seed variance.

## Data contract

The prepared dataset will eventually expose a stable sample contract containing
at least:

- the preprocessed image or image metadata needed for evaluation;
- an INVAE latent with fixed channel and spatial dimensions;
- an ImageNet class label; and
- representation-alignment features, initially planned to be precomputed.

The upstream dataset uses `images/` and `vae-in/` trees plus a label manifest.
The Priml preparation layer will document and validate this format rather than
silently accepting mismatched files.

Experiment construction must never download data, inspect the dataset, or load
model weights.  Preparation and heavyweight encoder work belong in the script
or dataset runtime, not in `experiments.py`.

## Dependency policy

The port will not copy the upstream `requirements.txt`.  Priml currently has
its own supported Python and PyTorch versions, and the baseline should reuse
existing Priml components wherever possible.  Optional GPU-only kernels,
external encoders, and expensive evaluators must be isolated behind explicit
runtime or integration paths.

## Planned commands

Once implemented, the intended user-facing commands are:

```bash
uv run python -m priml.baselines.speedrundit.scripts.prepare_data
uv run python -m priml priml.baselines.speedrundit.experiments.exp000
uv run pytest priml/baselines/speedrundit
```

The smoke path should be runnable with the repository's ordinary CPU test
dependencies and synthetic fixtures.

## Test policy

Portable tests must run on CPU and stay below 100 ms by shrinking depth, width,
batch size, and step count without changing the recipe.  They must not replace
the real optimizer, schedule, loss, initialization, or model behavior merely
to make a test fast.

The experiment suite will use Configgle's public
`configgle.testing.assert_pprint_golden` for finalized configuration goldens.
Model and optimizer behavior will use Priml's bit-for-bit golden harness inside
`host_agnostic_numerics()`, with `.float()` outputs and exact bit comparison.
Goldens must be deliberately perturbed once to verify that they fail before
being trusted.

## Upstream reference

The port is based on the public SpeedrunDiT repository and must preserve its
attribution, license obligations, model semantics, latent scaling, loss
definitions, sampling conventions, and evaluation protocol.
