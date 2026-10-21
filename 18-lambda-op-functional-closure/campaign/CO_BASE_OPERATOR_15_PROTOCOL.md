# Qwen3-8B Base-Operator C_O Component Audit 15

## Status

Prospective exploratory component audit. This phase is prepared but not yet executed on Qwen.

## Methodological correction

The campaign now separates three axes that must not be changed simultaneously:

1. component definition: Gamma, Delta, R, Lambda;
2. intervention algebra `A`;
3. quotient/symmetry contract `Q`;
4. resolution and macro-grain contract `rho`;
5. numerical compression.

Phase 15 freezes `(A,Q,rho)` at the current OP-level measurement contract and does not construct a global scalar `C_O`. Algebra richness and quotient strength are explicitly treated as different axes.

The system operation algebra remains exactly:

`{ID, KEEP, REVOKE, SUBSTITUTE}`

Experimental interventions are not system operations.

## Source systems

The pilot reuses 16 systems from the already opened Content-Typed Validation 13 panel. Selection is behavior-blind: the 16 rows with the lexicographically smallest SHA-256 hashes of their binding+donor tuples are selected.

Expected source panel SHA-256:

`92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4`

The existing typed content-effect Lambda proxy and the behavioral closure target are reused from that opened panel. New GPU work measures the missing routing information needed to construct candidate operational estimators for Gamma, Delta, and R.

## Experimental interventions

For each operation and named source address, the exact token span of that source clause inside `CONTROL_MAP` is identified without changing the operation instruction. Two source-localized cache interventions are applied at decoder layers 21-23:

- `ZERO_LOCAL`
- `HEAD_ROTATE_LOCAL`

For every source intervention, all three target addresses are probed over horizons 1, 2, 3, and 4. Final-hidden, primary-block, post-primary-block, and reference-block relative RMS responses are retained.

## Gamma candidate estimator

`GAMMA_QWEN_ROLE_CUT_RESPONSE_V1`

For each operation, a source-to-target causal response matrix is constructed from horizon-1 localized responses. For each non-trivial role cut, forward and backward role-normalized capacities are retained. The scalar minimum is reported only as a secondary summary; the cut profile is primary.

This is a candidate operational estimator. It is not asserted to be identical to the manuscript Gamma without construct validation.

## Delta candidate estimator

`DELTA_QWEN_INTERVENTION_RICHNESS_X_ROLE_PROPAGATION_V1`

For each operation:

- `Delta_cap` is estimated from the normalized effective rank of the source/intervention response family;
- `Pi_role` is estimated from off-diagonal role-propagation mass;
- the primary composition is `Delta_cap * Pi_role`.

The minimum and normalized mean compositions are reported as alternatives. They are not silently substituted for the primary product.

## R candidate estimator

`R_QWEN_ROLE_CYCLE_PROFILE_V1`

The localized horizon-1 routing matrix is row-normalized into a role transition matrix. Cycle-return values are reported for k=1..4. Direct same-role persistence across horizons 1..4 is reported separately as a control.

The cycle profile is a candidate algebraic re-entry estimator. It is not asserted to equal the manuscript `R(k)` until temporal construct validity is established.

## Lambda

Phase 15 reuses the existing 24D typed content-effect Lambda proxy from the opened Validation 13 panel. It does not refit or rescreen Lambda before the component audit.

## Diagnostic attribution and partition evaluation

Every component family is screened separately, followed by pairwise combinations and the full profile. Every compact partition reports heterogeneous collisions, sufficiency loss `H(A_dagger|C)/H(A_dagger)`, AMI, a fixed-fiber permutation control, fiber count, partition entropy, and nontrivial coverage. AMI is evaluation-only and cannot define raw component coordinates, roles, symmetries, or resolution.

All behaviorally heterogeneous system pairs are retained in the result with the compact component families that separate them. This makes attribution auditable pair by pair.

The diagnostic vocabulary is frozen to:

- `LOCALIZED`
- `INTERACTIVE`
- `DIFFUSE`
- `UNRESOLVED`

A diagnosis is exploratory and applies only to this pilot. If no tested representation supports a compact closed quotient, the result must remain `UNRESOLVED`.

## Scalarization checks

The audit explicitly compares:

- Gamma cut profile vs per-operation scalar minimum;
- Delta product vs minimum vs normalized mean;
- R k=1 only vs k=1..4 cycle profile;
- typed Lambda vs scalar norm.

No global scalar `C_O` is constructed.

## Next search phase

Only after Phase 15 identifies a stable component definition should the campaign compare preregistered points in the search space `(A,Q,rho)`. The component formulas must remain fixed during that comparison.

Enriching `A` (for example OP to PARAM_OP or ALGEBRA) generally adds distinctions. Strengthening `Q` (for example ROLE or orbit-based symmetry) removes declared irrelevant distinctions. Changing `rho` changes measurement grain. These directions must not be conflated.

Candidate selection on Qwen development data uses sufficiency loss, AMI with fixed-fiber permutation control, and explicit partition cost, then freezes one candidate for a single fresh holdout.

## Commands

```bash
/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python qwen3_8b_co_baseop15_prepare.py
/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python qwen3_8b_co_baseop15_run.py
/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python qwen3_8b_co_baseop15_score.py
```

## Implementation correction in v27

`operation_instruction()` renders the addressed CONTROL_MAP clause first. Therefore textual clause order depends on the source address. Clause localization now sorts address markers by their actual rendered position rather than assuming H1,H2,H3 order. The instruction text itself is unchanged.

The scorer now refuses empty or partial routing panels with explicit skip causes instead of allowing an empty NumPy stack.
