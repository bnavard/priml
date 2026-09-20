# SpeedrunDiT testdata plan

This directory will contain reviewed, small fixtures rather than ImageNet.

The repository keeps only small reviewed artifacts here. The portable tests
exercise synthetic tensors directly; full ImageNet assets do not belong here.

Artifacts:

- `exp000.txt`: Configgle finalized configuration golden;
- `speedrundit_model_forward.pt`: fixed tiny-model forward golden;
- `speedrundit_train_step.pt`: fixed synthetic one-update golden; and
- optional upstream parity fixtures containing inputs and expected named loss
  terms.

Full ImageNet samples, encoder weights, checkpoints, and FID reference batches
do not belong in the repository testdata directory.

Configuration goldens will be regenerated only through the Configgle harness.
Numerical goldens will be generated through the Priml BFB harness, reviewed,
perturbed to verify that they fail, and then replayed without regeneration.
