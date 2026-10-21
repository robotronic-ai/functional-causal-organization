# Fidelity-First 10: Target-Blind Representation Gate Before Closure

## Motivation

Metrology 09 found exact repeatability in the tested CUDA/bfloat16 stack while also showing substantial information loss in the handcrafted summaries used to compress the current cross-target Lambda/active-kernel proxy.

The next phase therefore separates two questions that had been mixed in several exploratory screens:

1. Does a compressed representation preserve the rich internal-response object with adequate fidelity?
2. Only after that representation is frozen, are behavioral targets constant on its fibers?

No behavioral target may be used to select the representation dimension or quotient granularity.

## Source object

The source is the sealed 24-system Cross-Target 07 panel.

Each system is represented by the target-blind 288-dimensional log-response vector:

`operation x source_address x target_address x horizon x internal_metric`

with four operations, three source addresses, three target addresses, two horizons, and four internal metrics.

This is a Lambda/active-kernel proxy. It is not the full four-component `C_O` profile and does not contain certified Gamma, Delta, or R.

## Stage A: target-blind dimensional fidelity

A standardized PCA family is evaluated at dimensions:

`1, 2, 4, 6, 8, 12, 16, 20, 22`

The PCA basis is fit without behavioral outcomes. For every dimension the audit reports:

- pairwise-distance Spearman rank preservation;
- 3-nearest-neighbor overlap;
- leave-one-out PCA reconstruction R2 of the standardized 288D vector;
- construction variance retained.

A PCA representation qualifies only if all engineering gates are met:

- pairwise-distance Spearman >= 0.95;
- kNN-3 overlap >= 0.75;
- leave-one-out reconstruction R2 >= 0.60.

The smallest qualifying dimension is selected. These thresholds are explicit engineering gates, not theoretical identities.

## Stage B: target-blind quotient fidelity

Only after a PCA representation qualifies, deterministic k-means partitions are fit on its PCA scores for:

`K = 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20`

No behavioral target is used for fitting or selecting K.

For each partition, fidelity is evaluated back in the standardized rich 288D space. A quotient qualifies only if:

- within-fiber rich distortion <= 0.20;
- nontrivial fiber coverage >= 0.20.

Among qualifying quotients, the smallest realized K is selected.

## Stage C: closure inspection after freeze

Only after the PCA dimension and quotient K are frozen is the 12-cell behavioral target opened.

Possible statuses are:

- `NO_FIDELITY_QUALIFIED_REPRESENTATION`
- `NO_FIDELITY_QUALIFIED_QUOTIENT`
- `FIDELITY_QUALIFIED_BUT_CLOSURE_REFUTED`
- `FIDELITY_QUALIFIED_EXPLORATORY_CANDIDATE`

A fresh validation is allowed only for the final status.

## Fresh Validation 10

If a candidate is eligible, Validation 10 runs 64 fresh systems. The PCA basis, standardizer, and quotient centers are copied unchanged from construction.

A validation candidate is promoted only if all three conditions hold:

- zero behavioral collision pairs;
- nontrivial coverage >= 0.20;
- within-fiber rich distortion <= 0.20 on the fresh panel.

Possible validation outcomes are:

- `USEFUL_VALIDATED`
- `REFUTED`
- `NONPROMOTED_LOW_COVERAGE`
- `NONPROMOTED_FIDELITY_FAILURE`

## Precision limitation

Metrology 09 established repeatability in the current bfloat16 CUDA stack but did not estimate systematic bfloat16 bias relative to a float32 execution. Fidelity-First 10 therefore tests internal consistency and compression fidelity relative to the current rich proxy, not absolute numerical truth against a higher-precision reference.
