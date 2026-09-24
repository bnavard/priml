# SpeedrunDiT testdata

Small reviewed fixtures only. ImageNet, INVAE and DINOv2 weights, checkpoints,
and FID reference batches do not belong here. The baseline README covers how
to regenerate these and what regenerating one means; this file says what each
one is.

| Artifact | Minted by | Freezes |
|---|---|---|
| `exp000.txt` | `experiments_test.py`, via `configgle.testing.assert_pprint_golden` | The finalized configuration tree of `exp000`, with defaults shown |
| `source_init.pt` | `scripts/parity.py --mint`, from the pinned reference itself | Every initialized tensor and the RNG state construction leaves, under the port's state names |
| `speedrundit_forward.pt` | `bfb_test.py::test_forward_bfb` | The op order of one eval forward |
| `speedrundit_five_steps.pt` | `bfb_test.py::test_five_steps_bfb` | The recipe, driven through `SpeedrunDiTTrainStep`: objective, draws, clip, AdamW moments, and the EMA shadow's averaging (its construction-time seed is compared by `scripts/parity.py`) |

The three `.pt` goldens share the parity script's geometry -- width 16, five
layers, two heads, a 4x4 grid of three latent channels, batch three -- which
shrinks size and nothing else. The routing ratios, the path-drop probability,
the value residual, the rotary positions, the qk norms and every
initialization rule are the ones `exp000` uses.

They are only meaningful against upstream commit
`c24c2ff25699cce63174ca56c2afcfeeb225e367`, and were established by
`scripts/parity.py`, whose every comparison held when they were minted. Re-run
it before regenerating any of them, or the new golden freezes a drift instead
of a decision.

The parity run writes nothing here unless asked to with `--mint`. Its fixture
corpus is built in a temporary directory and its measurements go to stdout,
because a golden of "we matched upstream once" goes stale the moment upstream
moves, and the goldens above already carry the numbers forward.
