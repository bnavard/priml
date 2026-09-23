# SpeedrunDiT

A Priml port of [SpeedrunDiT](https://github.com/SwayStar123/SpeedrunDiT), a
latent flow-matching transformer that reaches FID 3.49 on ImageNet-256 in
about 15 H100-hours. `exp000` reproduces the reference exactly; the goldens
beside it are what keep it reproduced.

Pinned reference: `c24c2ff25699cce63174ca56c2afcfeeb225e367`.

## Running

```bash
uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --synthetic --samples 4 --image-size 8 --latent-size 4 --latent-channels 8 --num-classes 10 --encoder-width 16
uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp_smoke
uv --quiet run --frozen python -m priml priml.baselines.speedrundit.experiments.exp000
uv --quiet run --frozen pytest priml/baselines/speedrundit
```

The synthetic corpus has to match the geometry of the experiment reading it:
`prepare_data`'s own defaults are exp000's (32 channels, a 16x16 latent,
768-wide targets), so the flags above are `exp_smoke`'s, and
`experiments_test.py` keeps them in step with its docstring.

`exp000` expects a corpus prepared by the reference's own encoding pipeline:
ImageNet-256 through INVAE (f16, 32 channels) for the latents, plus the frozen
encoder's targets beside them. `cls_token.npy` is required -- the class token is
an input the model diffuses -- and `features.npy` is optional, the alignment
term being zero without it. The reference computes both with DINOv2 ViT-B/14
from the images at every step; this port reads them precomputed and does not
reimplement either encoder. `prepare_data --verify` checks a tree against the
layout the loader reads, through the loader's own manifest parser.

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

`scripts/parity.py` measures the port against a fresh clone at the commit
above, every comparison exact (`torch.equal`) inside
`host_agnostic_numerics()`:

| Checkpoint | How it is compared |
|---|---|
| Data loading | Both loaders over one synthetic tree in the reference's layout; order, labels, latents and images |
| Initialization | Same seed; every parameter, buffer and the frozen table, plus the RNG state construction leaves behind |
| Parameter order | The order of every parameter that receives a gradient, which is the order the clip reduces its norm in |
| Five optimizer steps | Drawn times, drawn noise, the loss, every gradient, and every post-step weight |
| Eval forward | Perturbed weights, with the sparse path kept and dropped (`uncond`) |
| Sampling | Guided, on an interval, against the reference's FID sampler `euler_maruyama_sampler_path_drop` |

```bash
uv --quiet run --frozen --with timm python -m priml.baselines.speedrundit.scripts.parity
```

The reference imports `timm`, which Priml does not depend on, so the command
supplies it for its own duration. The script needs a network and a git clone,
so it is not a unit test. It establishes the port once against a moving
upstream; the goldens under `testdata/` freeze it afterwards and run on CPU in
the ordinary suite. Its geometry is the goldens' geometry, so what they freeze
is what it compared.

Two structural differences are **reported rather than normalized away**:

- State names differ (`parity.RENAMES` maps them), and the reference's root
  owns `mask_token` beside `pos_embed` where the port keeps the mask token with
  the routing that uses it. `named_parameters` yields a module's own
  parameters first, so the frozen table enumerates on the other side of the
  mask token; tensors are therefore compared by name, and the order of the
  trainable ones -- the only order any computation reads -- separately.
- Priml's `EMA` declines to average tensors with `requires_grad=False`
  (`priml/train/ema.py:502`), while the reference's `update_ema` walks every
  `named_parameter` and therefore lerps the frozen `pos_embed` — and
  `p * 0.9999 + p * 0.0001` is not `p` in float32, so its shadow drifts by
  rounding. Model weights are unaffected; the EMA shadow is not compared.

### What is shared, and what is not

| Piece | From |
|---|---|
| Train step, optimizer build, step timer, schedule scaling | `priml.train.train_step.TrainStep` |
| Weight averaging | `priml.train.ema.EMA` |
| Global gradient-norm clip | `priml.train.grad_clip.clip_grad_norm_` |
| Attention kernel | `priml.model.attention.kernel.SdpaFused`, through the `AttentionKernel` slot every shared attention uses |
| Every normalization | `priml.model.norm.RMSNorm` |
| `rectified_flow_path` | `priml.math.diffusion`'s log-SNR schedule and `compute_log_alpha` |
| `logit_normal_time` | `priml.math.probability.random_logit_normal` |
| The replayable shuffle stream | `priml.math.seed.salt` |

Local, each for a measured or structural reason:

- **The sampler.** `math.diffusion.ddpm_ddim` is the DDPM posterior step. Under
  the straight path it discretizes the reverse SDE with diffusion
  `2t / (1 - t)`, where the reference's Euler--Maruyama sampler uses `2t`; the
  two agree only in the continuous limit, and `math.diffusion` has no
  Euler--Maruyama step nor a score recovered from a velocity under an arbitrary
  path. The sampler therefore reuses the objective's own `interpolant` and
  `time_transform` slots instead, and `parity.py` checks it bit for bit.
- **`VisionRoPE`.** `priml.model.attention.rope.RoPE` computes the same
  rotation bit for bit in float32, but rotates in float32 whatever the
  activation dtype, while the reference rotates in it: under exp000's bf16
  autocast the two differ in 17% of elements. It also right-pads identity
  rotations where the class token here is a *leading* one.
- **`AdaLNModulation`.** `AdaLNZero` orders its chunks (scale, shift, gate)
  where the reference's weights are (shift, scale, gate), and its zero init
  draws nothing where `nn.Linear` draws, shifting every later initialization.
- **The leaves.** `nn.Linear`, `nn.Conv2d` and `nn.Embedding` rather than
  Priml's wrappers, whose initializers draw different counts.
- **`gelu_tanh`.** Priml costs no GELU.

`linear_path` writes `alpha = 1 - t`, `sigma = t` directly and is the default
over `rectified_flow_path` **only** because the reference writes the straight
form: the logit/sigmoid round trip rounds twice where `1 - t` rounds once.
`loss_test.py` asserts both halves -- they agree to `1e-6`, and they are not
bitwise equal. `resolution_time_shift` keeps `math.sqrt`, which the style
guide bans in favour of `** 0.5`, for the same reason: the two disagree for
1,557 of the element counts below `2**20`.

## Layout

| File | Owns |
|---|---|
| `data.py` | The prepared corpus, the shuffle stream, and resume |
| `model.py` | Every module and its Config |
| `loss.py` | The four-term objective and the path functions it injects |
| `train_step.py` | One optimizer update |
| `sampler.py` | The reference's Euler–Maruyama generation, with path-drop guidance |
| `metric.py` | Held-out velocity error |
| `experiments.py` | Recipe assembly only |
| `scripts/prepare_data.py` | Staging; never imported by a config |
| `scripts/parity.py` | The upstream comparison; never imported by the library |

## Testing

Portable tests run on CPU and shrink depth, width, batch and step count
without touching the recipe. Two things are deliberately **not** shrunk:

- No two axes share a length where a transpose between them could pass: the
  goldens run batch 3 against 2 heads, and 3 latent channels against a 4-wide
  grid.
- The SPRINT split keeps two encoder and two decoder layers around at least
  one sparse layer. A shorter trunk drops the sparse stage entirely and stops
  covering the routing, the mask token, and the fusion projection.

Regenerating goldens:

```bash
uv --quiet run --frozen --with timm python -m priml.baselines.speedrundit.scripts.parity --mint
CONFIGGLE_REGENERATE_GOLDEN=1 uv --quiet run --frozen pytest priml/baselines/speedrundit/experiments_test.py
BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/baselines/speedrundit/bfb_test.py
```

The initialization golden is minted by the parity script from the REFERENCE,
never from the port, and only once every comparison holds. The other two go
through the bit-for-bit harness, which randomizes and then loads parameters on
replay -- which is why initialization cannot be one of them. A missing harness
golden is minted **and still fails**, which is what forces someone to read it
before the next run accepts it. Regenerating one means the recipe changed.

## Attribution

SpeedrunDiT is MIT-licensed. The model derives from SiT and DiT, the rotary
implementation from EVA-02, and the position table from MAE; the alignment
term is REPA and the contrastive term is Contrastive Flow Matching. See the
module docstrings for citations.
