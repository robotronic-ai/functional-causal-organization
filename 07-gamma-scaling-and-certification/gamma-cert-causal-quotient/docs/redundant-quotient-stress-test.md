# Causal minimal-state quotient stress test under representational redundancy

## Purpose

The certification protocol in `gamma-cert-target-transformer/` requires a causal minimal-state
quotient to pass a sufficiency gate before Γ is certified on it (see that folder's
`docs/certification-protocol.md`, "Non-negotiable quotient gate"). This experiment tests, on an
exact finite system, whether such a quotient can remove purely representational redundancy
without hiding causal distinctions that become real under a richer intervention algebra. The
quotient is never selected to make Γ smaller or easier to compute — it is accepted only when it
remains causally sufficient under the declared intervention algebra.

## Construction

A latent system has four binary causal roles arranged in a recurrent ring. The physical
implementation contains **three replicas of each latent role**; training uses only *coherent*
states in which all replicas of one latent role share the same value, so the learned transition
realizes a well-defined latent transition on the coherent intervention manifold. A null-space
component is then added to the learned weight matrix: inside each replica group it sums to zero
(coherent states are unchanged), but raw interventions that separate replicas can expose new
causal distinctions the coherent view cannot see.

Two intervention algebras are compared:

- **Coherent**: all replicas of one latent role are intervened on together. The quotient is
  admissible only if the transition factors through the quotient map, `π∘T_micro = T_macro∘π`.
- **Raw physical**: replicas may be intervened on independently. The same quotient is sufficient
  only if every raw microstate inside one quotient class produces the same future macrostate.

Γ is computed exactly in both intervention contexts. These are genuinely different intervention
algebras, so the experiment does **not** claim raw Γ and macrostate Γ must be numerically equal;
the actual invariance claim is: *functionally equivalent physical implementations that share the
same causally sufficient minimal macrostate under the same intervention algebra must share the
same macrostate Γ.*

## Result (executed, exact finite computation)

| Null-space scale | Raw physical Γ | Macrostate Γ | Raw quotient sufficient? | Ambiguous macro classes |
|---:|---:|---:|---|---:|
| 0.0 | 0.1667 | 0.5000 | yes | 0 / 16 |
| 0.5 | 0.1699 | 0.5000 | yes | 0 / 16 |
| 1.0 | 0.2925 | 0.5000 | **no** | 15 / 16 |
| 2.0 | 0.4850 | 0.5000 | **no** | 16 / 16 |
| 4.0 | 0.4847 | 0.5000 | **no** | 16 / 16 |

The coherent transition remains unchanged (coherent training accuracy `1.0`, output unanimity
`1.0`) throughout. `Γ_macro = 0.5` is exact and constant at every scale; `Γ_raw` drifts upward as
the null-space perturbation grows and the raw-algebra quotient sufficiency test starts failing.
At scale `1.0`, 15 of 16 macro classes already contain raw microstates that reach distinct future
macrostates (up to 3 distinct outcomes within one class); by scale `4.0`, every macro class is
ambiguous under the raw algebra (up to 16 distinct outcomes within one class). Full per-scale
data, including explicit micro-state counterexample pairs: `results/nullspace_stress_results.json`.

## Interpretation

- **Representation sensitivity of raw Γ.** Duplicating a causal role changes the raw coordinate
  dimension and the admissible raw cut family, so raw Γ can vary even when the coherent
  functional computation is unchanged.
- **Implementation invariance of minimal-state Γ.** When the quotient is causally sufficient
  under the coherent algebra, the same latent transition gives the same macrostate Γ regardless
  of physical encoding.
- **Dependence on the intervention algebra.** A quotient valid under coherent interventions can
  become invalid under independent physical interventions — this is not a contradiction, the
  causal context has changed.
- **Correct response to quotient failure.** When raw interventions reveal distinct future
  macrostates inside one proposed quotient class, those microstates are causally distinct *for
  that algebra*. The correct response is a refined quotient, partial identification, or an
  `NA`/`V3` result per the larger protocol — the newly exposed distinctions must never be
  discarded to preserve a clean point value.

## Consequence for Γ certification: the five-step safe procedure

1. declare the intervention algebra;
2. propose candidate equivalence classes using discovery data;
3. freeze the quotient;
4. test causal sufficiency on independent interventions;
5. certify Γ for every still-admissible quotient, and report `V1` only if the resulting Γ
   interval is stable across the admissible quotient set.

This prevents a quotient from being selected merely because it improves the numerical result,
and it directly informs the "Non-negotiable quotient gate" step of the
`gamma-cert-target-transformer/` protocol: a quotient designed specifically for recurrence (or
for any other purpose) must not automatically replace the macrostate used by Γ — it must
preserve all causal consequences relevant to the macrostate, not only return-capable paths.

## Script

`scripts/gamma_redundant_quotient.py` is a single self-contained exact experiment (PyTorch used
only to train the small coherent transition network; all Γ/entropy computation is exact over the
finite `4096`-state raw alphabet and the `16`-state macro alphabet). No external model or target
checkpoint is required.
