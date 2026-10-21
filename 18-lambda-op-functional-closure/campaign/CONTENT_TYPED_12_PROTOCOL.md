# Content-Typed Kernel 12 Protocol

## Purpose

Test whether the closure failures of the RMS-only active proxy are caused by loss of typed content information rather than a missing primitive component.

## Fixed system algebra

The system operation algebra remains exactly:

`{ID, KEEP, REVOKE, SUBSTITUTE}`

No experimental intervention is added to this algebra.

## Experimental interventions

- `ZERO_SUFFIX`
- `HEAD_ROTATE`

Both act only on the control-instruction KV suffix in decoder layers 21-23.

## Measured content-effect channel

For `REVOKE` and `SUBSTITUTE` at each of `H1`, `H2`, and `H3`, candidate sequence log-probabilities are measured under the normal control state and under each internal intervention. Descriptor construction keeps only centered causal effects:

- delta(DONOR - ORIGINAL)
- delta(NONE - ORIGINAL)

for each intervention. This yields 24 continuous coordinates.

Absolute baseline log-probabilities and the baseline winning label are excluded from descriptor construction. The behavioral target remains the ordered 12-cell `operation x address` outcome vector.

## Exploratory screen

Each continuous coordinate is standardized. Candidate bin edges are target-blind empirical quantiles. Greedy feature selection is explicitly target-informed because the campaign is in exploratory candidate-finding mode. Search families include SUBSTITUTE-only, REVOKE+SUBSTITUTE, ZERO_SUFFIX-only, and SUBSTITUTE+ZERO_SUFFIX subsets.

A construction candidate is useful only if all hold:

- zero functional collision pairs;
- nontrivial coverage at least 20%;
- within-fiber distortion at most 25%.

## Validation

A selected candidate is copied unchanged into Fresh Validation 12. Standardization, selected coordinates, and quantile edges are not refit. Validation uses 64 fresh systems and the same promotion rule.

## Scope

This protocol enriches the typed Lambda proxy. It does not directly measure certified Gamma, Delta, or R, and does not establish a scalar complete `C_O`.
