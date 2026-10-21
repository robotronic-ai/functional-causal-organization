# Exploratory Candidate Result

## Status

`LAMBDA_OP_EXPLORE_GREEDY_QUANTIZED` is a target-informed exploratory candidate selected from the executed 96-system Qwen3-8B source holdout.

The candidate achieves zero functional collision fibers on the source holdout with three selected scalar features. The resulting partition has 89 fibers, of which 7 are nontrivial and contain 14 systems total.

## Frozen feature set

1. `SUBSTITUTE:H3:DLOG:reference_k_rms`
2. `SUBSTITUTE:H1:DLOG:reference_k_rms`
3. `KEEP:H1:DLOG:corridor_ratio`

Exact cutpoints are stored in `qwen3_8b_results/EXPLORATORY_BEST_CANDIDATE.json`.

## Interpretation

The source-holdout result is not confirmatory because behavioral targets were used to choose the descriptor. It is also only weakly compressive: 89 descriptors are observed across 96 systems. The correct next step is fresh validation with the candidate frozen exactly as selected.
