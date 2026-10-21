# Source-normalized directional Γ: definition, bound, and invariance evidence

## The problem this candidate fixes

The exact three-block intervention battery (entry `0–4`, core `5–30`, terminal `31–35`,
uniform independent binary block interventions, endogenous autoregressive re-entry) exposed a
normalization failure rather than a causal one. At a fixed evaluation horizon, the directional
causal-information terms `J_q(A→B)`/`J_q(B→A)` were unchanged while the K/V-support closure
depth grew from 1 to 8 cycles — but the denominator used by the raw-cardinality normalization
(`log₂` of the numerically distinct future K/V states) grew strongly, so the normalized Γ fell
even though nothing about the measured causal channel had changed. For the weak entry cut at
horizon 1: `J_entry→rest = 1.0` bit and `J_rest→entry = 0.0` bit at every tested closure depth,
while the old denominator grew from `4.0` to `≈9.275` bits and the old normalized score fell
from `0.25` to `≈0.109`. This is a normalization artifact, not evidence against the entry→rest
causal channel. See `results/quantum_gamma_closure_sensitivity_v11_results.json` and
`scripts/quantum_gamma_closure_sensitivity_v11.py`.

## Candidate definition

For a cut `π = (A, B)` and intervention distribution `q`, with

\[
J_q(A\to B') = \mathbb{E}_{b\sim q_B}\, I_q(A;B'\mid do(B=b)),
\]

define the **source-normalized directional efficiencies**

\[
\eta_q(A\to B') = \frac{J_q(A\to B')}{H_q(A)},
\qquad
\eta_q(B\to A') = \frac{J_q(B\to A')}{H_q(B)},
\]

with a zero convention when the corresponding source entropy is zero, and the cut/system score

\[
\Gamma_q^{\mathrm{dir}}(\pi) = \tfrac12\big[\eta_q(A\to B') + \eta_q(B\to A')\big],
\qquad
\Gamma_q^{\mathrm{dir}} = \min_\pi \Gamma_q^{\mathrm{dir}}(\pi).
\]

The denominator is the entropy of the *declared intervention source after the admissible causal
quotient*, not the numerical cardinality of the raw physical state. The recommended name is
**source-normalized interventional information efficiency**.

## Boundedness

For any fixed intervention of the complementary side, `I_q(A;B'∣do(B=b)) ≤ H_q(A)` (standard
mutual-information bound `I(X;Y) ≤ H(X)`); averaging over `b∼q_B` preserves the bound, so
`0 ≤ J_q(A→B') ≤ H_q(A)`, hence `0 ≤ η_q(A→B') ≤ 1` whenever `H_q(A)>0`. The same argument holds
in reverse, so `0 ≤ Γ_q^{dir}(π) ≤ 1`. This requires no future-state cardinality assumption.

## Agreement with the previous normalization on equal-capacity cuts

If `H_q(A) = H_q(B) = H` (in particular for a uniform full-support battery over finite
equal-capacity macro variables, where `H_q(A)=log₂|A|` and `H_q(B)=log₂|B|`), then
`Γ_q^{dir}(π) = [J_q(A→B') + J_q(B→A')] / (2H)`, which is exactly the manuscript's previous
equal-capacity normalization. The autonomous regression suite confirms this equality numerically
on every tested equal-capacity cut (`scripts/gamma_directional_normalization_regression_v13.py`).

## Closure invariance (executed, real target model)

Under the candidate normalization, the same weak entry cut that fell from `0.25` to `0.109`
under the old normalization stays exactly `0.5` for closure depths 1, 2, 4, and 8. The same
closure invariance holds at evaluation horizons 2 and 4. This isolates the earlier instability
as a denominator problem, not a change in the measured causal channel.

## Autonomous regression audit (v13) — self-contained, no target model required

`scripts/gamma_directional_normalization_regression_v13.py` passes all hard checks against a
reference suite (`ring_copy_5`, `xor_neighbors_5`, `mixed_boolean_5`,
`shared_noise_synergy_3`, plus boundary/monotonicity/permutation/random-kernel checks):

- reproduction of the previous exact benchmark values under the previous formula;
- boundedness of the candidate formula on the reference suite;
- exact zero for a disconnected identity system;
- `0.5` for a two-role one-way feed-forward system;
- `1.0` for a two-role bidirectional swap;
- monotonic increase under progressively stronger reverse coupling;
- invariance under role permutation;
- boundedness on random stochastic kernels;
- exact agreement with the previous formula on equal-capacity cuts.

| System | Previous Γ | Directional Γ |
|---|---:|---:|
| ring_copy_5 | 0.500000 | 0.416667 |
| xor_neighbors_5 | 1.000000 | 0.625000 |
| mixed_boolean_5 | 0.663910 | 0.476410 |
| shared_noise_synergy_3 | 0.500000 | 0.500000 |

These differences on unequal-capacity minimizing cuts are an expected *semantic* change (see
`campaign-chronology.md`, §"open semantic question"), not a numerical failure of either formula.

## Replication invariance (v14) — self-contained, no target model required

`scripts/gamma_directional_quotient_invariance_v14.py` replaces every macro role by 1, 2, 3, or
5 perfectly coherent physical copies, retaining one independent bit per macro role in the
intervention algebra. The candidate Γ is exactly invariant to the number of coherent physical
replicas in every tested system:

| System | Γ_dir (1, 2, 3, 5 replicas) |
|---|---:|
| ring_copy_5 | 0.416667 (all four) |
| xor_neighbors_5 | 0.625000 (all four) |
| mixed_boolean_5 | 0.476410 (all four) |
| shared_noise_synergy_3 | 0.500000 (all four) |

If source entropy were incorrectly replaced by physical-coordinate count, three coherent
replicas would spuriously divide the score by roughly three. The rule this establishes: **the
denominator must be computed from the declared intervention distribution after the admissible
causal quotient, not from raw physical dimensionality.**

## Scripts

- `scripts/quantum_gamma_kv_experiment_v1.py` — shared experiment infrastructure (KV-cache
  snapshot/restore, intervention plumbing); imported by every other script in this bundle. Not
  itself a headline result.
- `scripts/quantum_gamma_block_multihorizon_v10.py` — the final entry/core/terminal
  intervention battery at horizons 1/2/4; imported directly by v11 and v12.
- `scripts/quantum_gamma_closure_sensitivity_v11.py` — the closure-sensitivity diagnostic above.
- `scripts/quantum_gamma_directional_efficiency_v12.py` — computes `Γ_dir` on the real target
  model (see `campaign-chronology.md`).
- `scripts/gamma_directional_normalization_regression_v13.py` — the autonomous regression audit.
- `scripts/gamma_directional_quotient_invariance_v14.py` — the replication-invariance audit
  above; imports v13.
