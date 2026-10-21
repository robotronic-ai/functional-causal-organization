# Protocol CF++ CROSS-TYPE ASYMMETRY HOLDOUT v0.6.1 — FROZEN

**Date:** 2026-09-14  
**Status:** candidate-fixed holdout frozen before holdout fixture generation.

Candidate under test:

`joint_typed_propagation_candidate_v0_6.py`

Candidate hash is frozen in the manifest.

## Purpose

Test the intended rule:

> Cross-type role permutations are not treated as free coordinate
> transformations.

The prior I2 fixture was exactly symmetric under the proposed swap and was
therefore non-discriminating.

## Frozen asymmetric channel

Source:

\[
X\sim Bernoulli(1/2).
\]

Typed roles:

- `ACCESS-L`;
- `ACCESS-R`.

Reference channel:

\[
Y_L=X,
\qquad
Y_R=0.
\]

Forbidden cross-type swap while keeping the type contract fixed:

\[
Y_L=0,
\qquad
Y_R=X.
\]

## Obligations

H0. The candidate hash is unchanged from v0.6.

H1. The canonical typed joint channel of the reference is not equal to that of
the forbidden swap.

H2. Coherent output-label complement within `ACCESS-L` preserves the reference
equivalence class.

H3. Coherent source-label complement preserves the reference equivalence class.

## Verdict

If H0–H3 pass:

\[
\boxed{\texttt{CFPLUSPLUS-CROSS-TYPE-HOLDOUT-PASS}}.
\]
