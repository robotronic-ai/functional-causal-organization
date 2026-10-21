# Qwen3-8B Lambda-op Refinement 02

## Purpose

Validation 01 refuted the three-feature exploratory descriptor on a fresh 128-system panel. The seven collision fibers differ only in SUBSTITUTE outcomes. Refinement 02 therefore uses a target-informed, explicitly exploratory substitute-first search.

## Inputs

The search combines two already executed panels:

- the 96-system current holdout;
- the 128-system validation-01 panel.

No new model execution is required for candidate refinement.

## Search order

The candidate search evaluates three feature pools in this order:

1. SUBSTITUTE-only internal features;
2. SUBSTITUTE plus ID baseline features;
3. all available internal features.

For each pool, multiple quantization resolutions are evaluated. Greedy feature selection minimizes functional collision pairs first and descriptor complexity second.

## Candidate status

Any candidate produced by this search is target-informed and exploratory. A candidate is not accepted as transported closure until it passes Validation 02 without parameter changes.

## Validation 02

Validation 02 freezes the selected features and cutpoints, seals 128 fresh systems with no exact row reuse from the two search panels, runs Qwen3-8B, and scores exact fiber closure.
