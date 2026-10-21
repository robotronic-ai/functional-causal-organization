# Qwen3-8B Quotient / Resolution Replication 20 Protocol

## Purpose

Replication 20 tests whether the Phase 18 Lambda quotient that passed prospective Holdout 19 reproduces on a second, larger fresh panel. This is a replication step, not a selection or refinement step.

## Frozen candidate

- Candidate: `Q18_RHO2_L1_R1__LAMBDA`
- Candidate SHA-256: `ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc`
- Required Holdout 19 result SHA-256: `4fdeaf9e0145ecb5e99f0bb37a132382f07eb10f6f766e5caf41f973f69832e6`
- Generalization level: `OP`
- Operation algebra: exactly `ID / KEEP / REVOKE / SUBSTITUTE`
- Frozen Lambda rank depth: ordered top-1
- No candidate refit
- No quotient/resolution grid reopening
- No operation-algebra change

## Replication sample

- Systems: 64
- Seed: `202609292001`
- Value support: integers 30 through 89
- Binding/donor tuples are fresh relative to the earlier declared campaign panels and the Holdout 19 sealed schedule.

The replication uses the same finite battery and the same Lambda measurement branch as Holdout 19. This isolates reproducibility from cross-battery transport.

## Measurement

Only the component needed by the frozen candidate is measured:

- joint typed Lambda signature over H1/H2/H3;
- raw ordered top-2 signature retained in the panel;
- descriptor deterministically truncates each role ranking to ordered top-1;
- behavioral target is the separately co-measured ordered 12-cell operation x source-address outcome vector.

Behavior is not used to construct the descriptor.

## Replication criteria

A replication PASS requires all of the following on the 64 fresh systems:

1. zero heterogeneous collision pairs;
2. nontrivial coverage at least 0.20;
3. fiber fraction at most 0.80;
4. normalized sufficiency loss at most numerical zero (`1e-12` tolerance in the scorer);
5. adjusted mutual information above the 95th percentile of 5000 target permutations for this single fixed candidate.

No development-grid multiplicity correction is needed because no grid is reopened.

## Interpretation

`PROSPECTIVE_REPLICATION_PASS` supports reproducibility of the frozen quotient on a second independent panel sampled from the same declared finite battery. It still does not establish universal closure of full `C_O` or transport to different prompt families, supports, macro-grains, or models.
