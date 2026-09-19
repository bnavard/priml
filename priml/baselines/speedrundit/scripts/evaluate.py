"""Planned command-line entry point for generative evaluation.

This script will eventually:

- load a Priml checkpoint or generated sample directory;
- generate a fixed number of samples with the EMA model when requested;
- serialize evaluator-compatible sample batches;
- locate or validate the matching reference batch;
- invoke the native or explicitly wrapped FID-family evaluator; and
- write metrics and the resolved evaluation configuration to the run output.

It must remain separate from the fast training and unit-test paths.
"""
