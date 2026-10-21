# Content-Typed Kernel 12 Result

## Status

`USEFUL_EXPLORATORY_CANDIDATE`

## Executed panel

- systems: 32
- technical skips: 0
- panel SHA-256: `9b2b9ec96836d39383852ac666ddc2c4aa362473ebed7e9b52d24270e6a8d90a`
- full feature dimension: 24
- operation family selected: `SUBSTITUTE_ONLY`

## Candidate

`CONTENT_TYPED_SUBSTITUTE_ONLY_2F`

Selected quantized features:

1. `SUBSTITUTE:H3:HEAD_ROTATE:NONE_MINUS_ORIGINAL`, 6 bins.
2. `SUBSTITUTE:H1:ZERO_SUFFIX:DONOR_MINUS_ORIGINAL`, 6 bins.

The quantile edges were fit without behavioral targets. Greedy feature selection was exploratory and used the behavioral closure target. Baseline absolute log-probabilities and baseline outcomes were excluded from descriptor construction.

## Construction metrics

- fibers: 22
- nontrivial fibers: 9
- systems in nontrivial fibers: 19
- nontrivial coverage: 59.375%
- collision pairs: 0
- within-fiber rich distortion: 24.5471%
- rich variance explained by fibers: 75.4529%

The candidate satisfies the declared exploratory gate but lies close to the 25% distortion ceiling. It therefore requires unchanged evaluation on a fresh validation panel before any promotion.

## Interpretation

The selected features are content-selective causal effects within `SUBSTITUTE`. They retain whether internal interventions alter the relative causal support for DONOR, NONE, and ORIGINAL content rather than only the magnitude of hidden-state propagation. This is an enrichment of the operation-indexed Lambda proxy, not an additional primitive component and not a validation of complete `C_O=(Gamma,Delta,R,Lambda)`.

## Fresh Validation 12

The candidate must be frozen exactly as constructed. No feature reselection, bin refit, standardizer refit, or threshold adjustment is permitted before scoring the fresh 64-system panel.
