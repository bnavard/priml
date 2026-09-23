# SpeedrunDiT testdata

Small reviewed fixtures only. ImageNet, INVAE and DINOv2 weights, checkpoints,
and FID reference batches do not belong here. The baseline README covers how
to regenerate these and what regenerating one means; this file says what each
one is.

| Artifact | Minted by | Freezes |
|---|---|---|
| `exp000.txt` | `experiments_test.py`, via `configgle.testing.assert_pprint_golden` | The finalized configuration tree of `exp000`, with defaults shown |
| `speedrundit_init.pt` | `bfb_test.py::test_initialization_bfb` | Construction order and every initialization rule |
| `speedrundit_forward.pt` | `bfb_test.py::test_forward_bfb` | The op order of one eval forward |
| `speedrundit_five_steps.pt` | `bfb_test.py::test_five_steps_bfb` | The recipe, driven through `SpeedrunDiTTrainStep`: objective, draws, clip, EMA, AdamW moments |

The three `.pt` goldens are minted at a miniature geometry — 64 channels, six
layers, four heads, a 4x4 latent grid — which shrinks size and nothing else.
The routing ratios, the path-drop probability, the value residual, the rotary
positions, the qk norms and every initialization rule are the ones `exp000`
uses.

They are only meaningful against upstream commit
`c24c2ff25699cce63174ca56c2afcfeeb225e367`, and were established by
`scripts/parity.py`. Re-run it before regenerating any of them, or the new
golden freezes a drift instead of a decision.

The parity run itself writes nothing here. Its fixture corpus is built in a
temporary directory and its measurements go to stdout, because a golden of "we
matched upstream once" goes stale the moment upstream moves, and the three
above already carry the numbers forward.
