# Functional Access ↔ C_O Necessity Bridge (Breadth / Temporal / Unity)

**Status: executed on frozen synthetic fixtures.** Harness/result:
`scripts/af_access_consequence_candidate_v0_1.py`,
`results/ACCESS_UNITY_GAMMA_BRIDGE_RESULT_v0.2.json`,
`results/FUNCTIONAL_ACCESS_CONSTRUCT_CF_BRIDGE_RESULT_v0.1.json`.

**Notation note:** as elsewhere in this folder, `Δ_prop → Δ` in this
manuscript's notation; `c_F` below is the geometric scalar summary discussed
in §18/Annexe A of the manuscript, not `C_O` itself.

## 1. Purpose

`H_CO⁺` (and its predecessor formulations) needs some argument for *why*
`Γ`, `Δ`, and `R(k)` are the right three organizational quantities to test,
rather than an arbitrary engineering choice. This document gives an
independent, non-circular argument: define functional access directly (not
via C_O), then derive necessity of each component from stated access
axioms.

## 2. Independent target construct

Functional access is represented independently of C_O by a typed,
task-conditional access process `A(e) = [K_A]_semantic`. The manuscript's
behavioral battery `A†_I` is a finite consequence sample from this
construct — it is not identical to it.

## 3. Breadth theorem

If the same content `P` causally informs at least two distinct certified
consumer roles (`I(P;Y^(j)) > 0` and `I(P;Y^(ℓ)) > 0`), then both
corresponding Shapley contributions are positive, giving `N_eff > 1` and
`Π_role > 0`. Since total discriminable information is then positive:

```
A_breadth > 0  ⟹  Δ > 0
```

This is exact under the frozen role definitions.

## 4. Finite bounded-depth temporal theorem

For a fixed implementation with `n` endogenous functional role types and no
unbounded external append-only store: if content introduced at `t₀` remains
causally available through endogenous state at a horizon requiring a
directed role path of length at least `n`, the role-transition graph
contains a directed cycle (an acyclic fixed-role implementation can sustain
endogenous content only up to `H_acyclic ≤ n − 1`). So sustained access
beyond a fixed architecture's acyclic depth requires a functional role
cycle. Adding the further requirement of return informativeness after
persistence-neutralization:

```
A_sustained^(H > H_acyclic) + return informativeness  ⟹  R(k) > 0  for some k
```

This finite form is the operational version of an earlier infinite-horizon
statement.

## 5. Unity theorem

Define `A_unity > 0` independently as: every admissible bipartition of the
declared access mechanism loses at least one frozen access capability when
cross-part causal influence is cut. Under alignment between those
access-critical interventions and the `Γ` battery:

```
A_unity > 0  ⟹  Γ > 0
```

The exact XOR fixture passed with `Γ = 0.5` and every admissible cut
reducing task accuracy from 1 to 1/2, while a decomposed parallel control
retained typed access with `Γ = 0` — confirming unity is an additional
access requirement, not a consequence of bare availability.

## 6. Conditional BTU representation theorem

Let `B` = multi-role breadth, `T` = sustained access beyond acyclic depth
with informative return, `U` = partition-irreducible access unity. Define
`A_BTU = B ∧ T ∧ U`. Then:

```
A_BTU > 0  ⟹  Δ > 0,  R(k) > 0,  Γ > 0
```

and, for an identified positive return horizon, `A_BTU > 0 ⟹ c_F > 0`
(where `c_F` is the geometric scalar summary). **This is a necessity
theorem under explicit access axioms — it is not a sufficiency theorem.**

## 7. What remains open

The central unresolved question is no longer "why these three causal
factors?" but:

```
A ≟ A_BTU,   or at least   A ⟹ A_BTU
```

— a construct-identification question. The three factors are now linked to
three independently stated functional requirements, but it remains to
establish that those requirements are constitutive of the intended notion
of functional access, rather than a stronger engineered access
architecture. Concretely, still open: can functional access exist with only
one consumer role? Can it exist only transiently within a fixed acyclic
depth? Can it exist in causally decomposed parallel modules without access
unity? A convincing positive case for any of these would mean the
corresponding BTU axiom is not constitutive of `A`.

## 8. Consequence for the compact scalar

Until BTU identification is resolved, the geometric scalar
`c_F = (Γ·Δ·R(1))^(1/3)` must remain a secondary organizational summary. Its
zero-gating semantics corresponds exactly to the conjunction `B ∧ T ∧ U`,
not yet to all possible meanings of functional access — this is the same
caution the manuscript already applies to `c_O` in §18/Annexe A, now given
an independent derivation.

## 9. Relation to the rest of this folder

This necessity bridge is what motivates targeting `Γ`, `Δ`, `R(k)`
specifically in the FCI-3 intervention families Z_G, Z_D, Z_R
(`fci3-protocol-specification.md`, §4) — it is evidence for *why these
components*, not evidence that manipulating them causally produces access
(that is exactly what FCI-3 itself would need to test).
