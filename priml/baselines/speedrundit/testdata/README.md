# SpeedrunDiT testdata

Small reviewed fixtures only. ImageNet, INVAE and DINOv2 weights, checkpoints,
and FID reference batches do not belong here.

| Artifact | Minted by | Freezes |
|---|---|---|
| `exp000.txt` | `experiments_test.py` via `configgle.testing.assert_pprint_golden` | The finalized configuration tree of `exp000`, with defaults shown |
| `speedrundit_init.pt` | `bfb_test.py::test_initialization_bfb` | Construction order and every initialization rule |
| `speedrundit_forward.pt` | `bfb_test.py::test_forward_bfb` | The op order of one eval forward |
| `speedrundit_five_steps.pt` | `bfb_test.py::test_five_steps_bfb` | The recipe: objective, draws, gradients, clip, AdamW moments |

The three `.pt` goldens are minted at a miniature geometry — 64 channels, six
layers, four heads, a 4x4 latent grid — which shrinks size and nothing else.
The routing ratios, the path-drop probability, the value residual, the rotary
positions, the qk norms and every initialization rule are the ones `exp000`
uses.

## Regenerating

```bash
CONFIGGLE_REGENERATE_GOLDEN=1 uv --quiet run --frozen pytest priml/baselines/speedrundit/experiments_test.py
BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/baselines/speedrundit/bfb_test.py
```

A missing bit-for-bit golden is minted **and still fails**, which forces
someone to read it before the next run accepts it. Regenerating one means the
recipe changed — say which change, in the commit.

These goldens are only meaningful against upstream commit
`c24c2ff25699cce63174ca56c2afcfeeb225e367`. They were established by
`scripts/parity.py`; re-run it before regenerating any of them, or the new
golden freezes a drift instead of a decision.

## Not committed here

The parity comparison writes nothing. Its fixture corpus is built in a
temporary directory and its measurements go to stdout, because a golden of
"we matched upstream once" would go stale the moment upstream moved, and the
three goldens above already carry the numbers forward.
