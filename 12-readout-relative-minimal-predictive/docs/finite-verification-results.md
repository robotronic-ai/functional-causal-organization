# Finite verification results

Source material: `RESULTS.md`. Five exhaustive finite-universe checks, run by
`scripts/run_all.py`; every one below is status `PASS` with **zero** failures.

## 1. Exhaustive arbitrary-readout orbit corollary

**Universe.** Four finite states; all `4^4 = 256` deterministic maps; all 15
non-empty source subsets; 3,840 map/source cases; all scalar functions as admissible
readouts. **Script:** `scripts/verify_all_readouts_orbit_corollary.py`.

**Result.** Orbit-machine / behavioral-quotient failures: `0`. Minimal predictive
dimension vs. reachable-state-count failures: `0`. Reachable state count distribution:
1 state (256 cases), 2 states (768), 3 states (1,440), 4 states (1,376).

Status: `EXHAUSTIVE-FINITE-CHECK / PASS`.

## 2. Exhaustive restricted-readout nonlinear finite check

**Universe.** `X = F2^2`; all 256 deterministic maps on the four states; all 15
non-empty source subsets; declared readout basis `{x1, x2}`; 3,840 map/source cases.
**Script:** `scripts/verify_finite_predictive_theorem.py`.

**Verified with zero failures:** (1) Koopman closure invariance; (2) behavioral
quotient invariance under the transition; (3) separation of distinct behavioral
classes by an original readout at some future delay; (4) well-defined linear
transition on predictive evaluation states; (5) generalized Hankel rank equals
predictive realization dimension.

Predictive dimension distribution: 0 (64 cases), 1 (336), 2 (1,046), 3 (1,650),
4 (744).

Status: `EXHAUSTIVE-FINITE-CHECK / PASS`.

## 3. Exhaustive U0-style linear corollary

**Universe.** Field `F2`; state dimension 3; all 512 binary `3x3` dynamics; source
interface injects `e1, e2`; declared readouts are coordinate functionals `x1, x2`.
**Script:** `scripts/verify_u0_linear_corollary.py`.

**Checked identity:** Hankel rank = reachable dimension − reachable unobservable
dimension. Failures: `0 / 512`. Minimal predictive dimension distribution:
dimension 2 (224 systems), dimension 3 (288 systems).

Status: `EXHAUSTIVE-FINITE-CHECK / PASS`.

## Interpretation (checks 1-3)

The finite checks support the definitions and the proof architecture; they do not
replace the general derivation. The single most informative observation: the
deterministic behavioral quotient and the minimal linear predictive realization are
genuinely different objects. In the restricted-readout nonlinear universe (check 2),
the behavioral quotient may retain every reached state while the linear predictive
realization has strictly lower dimension, because predictive state vectors can be
linearly dependent (compare the dimension-0/1/2/3/4 histogram above against the
reachable-state-count histogram in check 1 — the same 3,840 cases give visibly
different distributions).

## 4. Exhaustive controlled finite-state check

**Universe.** Three abstract states; two deterministic control generators; all
`27^2 = 729` ordered transition-map pairs; all 7 non-empty source subsets; one
declared binary readout `1[state == 1]`; 5,103 generator-pair/source cases.
**Script:** `scripts/verify_controlled_finite_theorem.py`.

**Verified with zero failures:** (1) multi-generator Koopman closure; (2) controlled
behavioral congruence; (3) separation of distinct predictive classes by a declared
readout under some control word; (4) well-defined predictive linear dynamics under
each generator; (5) controlled generalized Hankel rank equals predictive dimension.

Predictive dimension distribution: 0 (522 cases), 1 (711), 2 (1,436), 3 (2,434).

Status: `EXHAUSTIVE-FINITE-CHECK / PASS`.

**Interpretation.** This result supports the observable-semigroup formulation
(`docs/semigroup-generalization.md`) and directly bridges a single autonomous
transition to intervention/control families. As with checks 1-3, this is validation of
the implementation and definitions, not a substitute for the algebraic proof.

## 5. Exact finite Markov expectation corollary

**Universe.** Two states; two control kernels; each row uses `P(next=1)` in
`{0, 1/2, 1}`; 9 kernels and 81 ordered kernel pairs; source sets `{0}`, `{1}`,
`{0,1}`; 243 kernel-pair/source cases; exact rational arithmetic (`fractions.Fraction`).
**Script:** `scripts/verify_markov_expectation_corollary.py`.

**Verified with zero failures:** (1) Markov Koopman closure; (2) invariance of the
predictive dual state space; (3) generalized Hankel rank equals predictive
realization dimension.

Predictive dimension distribution: 0 (9 cases), 1 (26), 2 (208).

Status: `EXHAUSTIVE-FINITE-CHECK / PASS`.

This result supports the expectation-level Markov corollary. It does **not**
establish full path-law equivalence or probabilistic bisimulation — see
`docs/stochastic-extension-and-open-questions.md`.

## Summary

| Check | Cases | Failures | Status |
|---|---:|---:|---|
| Arbitrary-readout orbit corollary | 3,840 | 0 | PASS |
| Restricted-readout nonlinear theorem | 3,840 | 0 | PASS |
| U0-style linear corollary | 512 | 0 | PASS |
| Controlled finite-state theorem | 5,103 | 0 | PASS |
| Markov expectation corollary | 243 | 0 | PASS |
| **Total** | **13,538** | **0** | **PASS** |
