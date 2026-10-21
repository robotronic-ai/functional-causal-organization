# Turn 137 — Q2 positive distinction certificate and identified-set update

## Decision

The T135 Q2 candidate-anchor calibration adds the first hard same-layer distinction constraint on the natural Qwen3-8B role quotient in bf16.

At L23, the two previously resolved active witnesses

- `heads[24,25,26,27]`
- `mlp3`

are certified distinct by the T135 symmetric fixed-factor refactorization test using the predeclared L24 target `heads[28,29,30,31]`.

The refactorization effect is above `2 * delta_T` in all 10 exact-recoding replicas. Therefore the completion in which the two L23 anchors are one role is no longer admissible.

## Q2 lower transport relation

With `delta_T = 0.12200575938990278`:

- `heads[24,25,26,27] -> heads[28,29,30,31]` is `TRANSPORTED` under both ABLATION and SUBSTITUTION.
- `mlp3 -> heads[28,29,30,31]` is `TRANSPORTED` under SUBSTITUTION.
- The corresponding MLP ABLATION edge is `AMBIGUOUS` and remains unresolved; it is not an absence claim.

Cross-layer fusion remains forbidden. These are transport/alignment constraints only.

## Consequences

The L23 anchor subsystem is now provably nontrivial: it contains at least two distinct active causal roles. This does not complete the L23 role inventory because all other candidates remain unresolved.

Accordingly:

- Gamma remains NA. A nontrivial anchor subsystem exists, but the full admissible role inventory and unrestricted cut minimum are not identified.
- Delta remains NA. The earlier one-role completion for the two anchors is excluded, but Shapley coverage over the admissible role inventory has not been measured.
- Lambda_O can become a partial typed lower relation once the T135 L1 files are merged. UNRESOLVED cells never become negative edges.
- R remains NA because temporal return-role identity is a separate certification problem.

## Next action

Run `turn137_update_identified_set.py` with the T132 partial RoleQ file, T135 Q2 output, T135 Q2 validation, and both T135 L1 outputs. The result is a machine-readable identified set, not a point quotient.
