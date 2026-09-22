# SpeedrunDiT

A Priml port of [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT), a
latent flow-matching transformer that reaches FID 3.49 on ImageNet-256 in
about 15 H100-hours. `exp000` reproduces the reference exactly; the goldens
beside it are what keep it reproduced.

Pinned reference: `c24c2ff25699cce63174ca56c2afcfeeb225e367`.

## Running

```bash
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --synthetic --samples 64
uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp_smoke
uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp000
uv --quiet run --frozen pytest priml/baselines/speedrundit
```

`exp000` expects a corpus prepared by the reference's own encoding pipeline:
ImageNet-256 through INVAE (f16, 32 channels) for the latents, and DINOv2
ViT-B/14 for the alignment targets. Neither encoder is reimplemented here.
`prepare_data --verify` checks a tree against the layout the loader reads;
`prepare_data --synthetic` writes that same layout with random tensors, which
is what lets the smoke experiment and every loader test run with no ImageNet.

## Experiments

| Experiment | Purpose |
|---|---|
| `exp000` | SR-DiT-B/1 reproducing the pinned reference |
| `exp_smoke` | The same recipe at a size that runs on a laptop CPU |

`exp000` is a parity port, not the "best naive recipe" the usual `exp000`
contract asks for. That is a deliberate, argued departure: the baseline exists
to show Priml's components compute the published model exactly, so the control
has to be the published model — SPRINT routing, value residuals, rotary
positions and all. Once parity holds, a later fork can strip one of them and
measure what it earned.

## Bit-for-bit parity

Four things are pinned to the reference, and `scripts/parity.py` measures each
against a fresh clone at the commit above:

| Checkpoint | How it is compared |
|---|---|
| Data loading | Both loaders over one synthetic tree in the reference's layout; order, labels, latents and images compared with `torch.equal` |
| Initialization | Same seed, same geometry; every parameter compared positionally |
| Forward and objective | Fixed inputs; every named loss term compared |
| Five optimizer steps | Loss, drawn times, drawn noise, every gradient, and every post-step weight |

```bash
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.parity
```

The script needs a network and a git clone, so it is not a unit test. It
establishes the port once against a moving upstream; the goldens under
`testdata/` freeze it afterwards and run on CPU in the ordinary suite.

Two structural differences are **reported rather than normalized away**:

- Value-residual parameters are `attn.value_residual.weight` here and
  `attn.v1_lambda` upstream, so parameters are compared by position, which is
  what construction order makes the honest key anyway.
- Priml's `EMA` declines to average tensors with `requires_grad=False`
  (`priml/train/ema.py:502`), while the reference's `update_ema` walks every
  `named_parameter` and therefore lerps the frozen `pos_embed` — and
  `p * 0.9999 + p * 0.0001` is not `p` in float32, so its shadow drifts by
  rounding. Model weights are unaffected; the EMA shadow is not compared.

### Why some of `priml.math.diffusion` is not on the training path

The schedule family there is parameterized on log-SNR:
`log_snr_from_log_time_per_logit` maps `t` through a logit and
`log_sigma_from_log_snr_per_rectified_flow` maps back through a sigmoid, so
recovering `sigma` costs a round trip. The reference writes `alpha = 1 - t`
and `sigma = t` directly. The two agree far below a float32 ULP and are not
bit-identical, and this baseline's contract is exact parity, so the
interpolant stays in time space in `loss.py`. `target_rectified_flow`'s own
target, `eps - x`, *is* reproduced exactly by the linear path's `-1 * x + 1 * eps`.

`priml.math.diffusion.sample` is not used by `sampler.py` for a structural
reason rather than a numerical one: it threads a single state tensor, and this
model diffuses a latent and a class token jointly on a shared time grid.

## Layout

| File | Owns |
|---|---|
| `data.py` | The prepared corpus, the shuffle stream, and resume |
| `model.py` | Every module and its Config |
| `loss.py` | The four-term objective and the path functions it injects |
| `train_step.py` | One optimizer update |
| `sampler.py` | Euler–Maruyama generation |
| `metric.py` | Held-out velocity error |
| `experiments.py` | Recipe assembly only |
| `scripts/prepare_data.py` | Staging; never imported by a config |
| `scripts/parity.py` | The upstream comparison; never imported by the library |

## Testing

Portable tests run on CPU and shrink depth, width, batch and step count
without touching the recipe. Two things are deliberately **not** shrunk:

- `heads` stays at four. Two heads make the head axis and the batch axis the
  same length at the test batch size, which would hide a transpose.
- The SPRINT split keeps two encoder and two decoder layers around at least
  one sparse layer. A shorter trunk drops the sparse stage entirely and stops
  covering the routing, the mask token, and the fusion projection.

Regenerating goldens:

```bash
CONFIGGLE_REGENERATE_GOLDEN=1 uv --quiet run --frozen pytest priml/baselines/speedrundit/experiments_test.py
BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/baselines/speedrundit/bfb_test.py
```

A missing bit-for-bit golden is minted **and still fails**, which is what
forces someone to read it before the next run accepts it. Regenerating one
means the recipe changed.

## Attribution

SpeedrunDiT is MIT-licensed. The model derives from SiT and DiT, the rotary
implementation from EVA-02, and the position table from MAE; the alignment
term is REPA and the contrastive term is Contrastive Flow Matching. See the
module docstrings for citations.
