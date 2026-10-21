# Qwen3-8B Formula Measurement 17 Protocol

## Status

Prospective development measurement after Formula Contract 16 was frozen. This phase keeps the system-operation level fixed at OP and does not introduce ROLE, composed operations, or a target-informed symmetry quotient.

## Frozen prerequisite

The local Formula Contract 16 freeze must exist and have SHA256:

`c17c401ff51751b16c33426a65dda95d9659b15e4c34c907de565de7ef8d0d04`

The opened Content-Typed Validation 13 panel is used only as the source of 12 development systems and, after raw measurement is complete, as the source of the independent behavioral target. Its required SHA256 is:

`92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4`

The 12 systems are selected by lexicographically sorting a SHA256 of the binding-plus-donor tuple. Selection is behavior-blind.

## Fixed design axes

- intervention algebra `A`: OP with exactly `ID`, `KEEP`, `REVOKE`, `SUBSTITUTE`;
- quotient/symmetry axis `Q`: unchanged in this phase;
- macro grain: `H1/H2/H3`;
- numeric partition resolution `rho`: fixed at `1/16` for normalized continuous coordinates;
- categorical signatures: retained exactly;
- raw measurement never reads `A_dagger`.

Experimental interventions are not system operations.

## Finite support

Each role receives a deterministic three-state numeric support constructed without the behavioral target:

1. original value;
2. declared donor value;
3. first unique wrapped offset from a fixed offset list.

All values remain inside the campaign support `30..89`. The finite perturbation support therefore has `|P*|=3`.

All information quantities are exact for the declared finite batteries. They are not asserted to be intrinsic model-wide constants.

## Gamma

For each operation and each singleton-vs-pair role cut, measure both directions of the finite-battery causal channel.

For the singleton-to-pair direction, vary the singleton through its three declared states under two fixed opposite-side contexts. The opposite-side readout is a tagged mixture of the two role-value distributions.

For the pair-to-singleton direction, vary the pair jointly through the three aligned support states under two fixed singleton contexts. The singleton readout is its finite role-value distribution.

For each direction:

`J = average_context I(P ; Y)`.

Normalize both directional values by `log2(3)`, report directional balance, and use the arithmetic mean as the cut aggregator. Save the complete cut profile before the per-operation minimum.

This is a finite-battery operational realization of the directional information contract. It does not claim exhaustive enumeration of all latent Qwen cuts.

## Delta

For each operation and source address, reuse the native opposite-side Gamma context and the three source perturbations. The readout is the tagged mixture over all three role-value distributions.

The preregistered perturbation distribution is:

`q(P) = (0.5, 0.3, 0.2)`.

Report:

- `H(P)`;
- `I(P;Y)`;
- `kappa_q = H(P)/log2(3)`;
- `delta_q = I(P;Y)/H(P)`;
- `Delta_cap = I(P;Y)/log2(3) = kappa_q * delta_q`.

The pair `(kappa_q, delta_q)` is primary; `Delta_cap` is the scalar synthesis.

## R

For each operation and source address, use three internal intervention states:

- `CONTROL`;
- `ZERO_LOCAL`;
- `HEAD_ROTATE_LOCAL`.

The interventions act only on the named source clause in decoder layers 21--23 and are not added to the system-operation algebra.

Build the source quotient from two frozen auxiliary contexts. For each intervention, keep the ordered top-2 source-role response labels in both auxiliary contexts. Identical signatures define one source class `C`. The auxiliary contexts are never used as return contexts.

Measure the return channel on two disjoint frozen return contexts and report:

- `K_eff = H(C)`;
- `B_reentry = mean_U I(C ; Y_ret | U=u)`;
- `R = B_reentry / K_eff` when `K_eff>0`;
- source-class partition;
- label-duplication invariance.

The Qwen causal negative control is adapted to the cache intervention: all cached positions causally prior to the localized source clause must remain unchanged to `1e-12` in the intervened cache. This storage-level control is not claimed to be identical to the earlier Pythia pre-logit control, so the measurement is labeled `ADAPTED_QWEN_REENTRY_MEASUREMENT`.

## Lambda

For every operation, source address, and internal intervention state, probe all three roles on the same intervened cache. For each role keep the ordered top-2 finite content labels, and retain the ordered `H1/H2/H3` tuple as the primary signature.

No marginal pooling is used in the primary Lambda signature. Marginal probability vectors are saved only as supporting data.

This is an explicitly declared finite joint typed signature. It is not claimed to recover an unrestricted full latent joint distribution.

## Development evaluation

Only after the raw measurement panel is complete is the frozen behavioral profile opened. Exactly 11 candidate families are evaluated:

`GAMMA`, `DELTA`, `R`, `LAMBDA`, all six pairs, and `FULL_CO_OP`.

The representation rule for each component is fixed before target opening. Continuous normalized coordinates are rounded at the fixed `1/16` resolution; categorical Lambda and R source-partition signatures are exact.

A development family is admissible only if all conditions hold:

- zero heterogeneous collision pairs;
- nontrivial coverage at least `0.20`;
- fiber fraction at most `0.80`;
- normalized sufficiency loss equal to zero;
- AMI exceeds the 95th percentile of 1000 target permutations with descriptor fiber structure fixed.

Among admissible preregistered families, select the coarsest partition: fewer fibers first, then fewer components, then lexical tie-break.

AMI is evaluation-only. It does not define measurements, roles, interventions, symmetries, or resolution.

## Promotion rule

A development candidate is not a closure validation. If one candidate is found, it must be frozen without refit and opened exactly once on a fresh holdout. If no candidate is admissible, do not automatically enrich the operation algebra; first inspect component and quotient-axis failure modes at the fixed OP level.
