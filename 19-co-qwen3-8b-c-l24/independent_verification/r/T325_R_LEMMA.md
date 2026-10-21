# L24 Cache-Bypass Lemma and Distinct R Gates

## Proposition

For a strictly causal decoder-only transformer, let D_t modify only a physical write at L24 **after** the current-token K/V of L24 have been computed, and before execution of higher layers. Hold the complete prior history, previously acquired causal persistence and generated next token fixed in the two interventions. Then the L24 computation on the next token is identical, independent of D_t. Any effect of D_t on L24 at the next decoding cycle must therefore be mediated through a change in the generated token, for this fixed intervention family.

## DAG proof

At t, the L24 self-attention K/V are functions of the representation entering L24 and are committed before D_t. Every lower-layer K/V is also computed before D_t. Changes induced by D_t can affect the output of L24, later layers, final logits, generated token, and K/V written by **later layers** at t. At t+1, the states entering L24 depend only on the same prior prefix, same lower-layer caches, same L24 K/V history, and the newly supplied token. Upper-layer K/V caches do not feed backwards into earlier layers. When that token is clamped to the same value, the L24 next-step function has identical inputs and parameters, hence identical outputs in exact deterministic arithmetic. This is a model-structural conditional independence, not a proof that the real donor perturbation changes the emitted token.

## Measurements required for R

1. The actual frozen **u26 L24 donor** edits at time t should be evaluated at the same declared boundary and under the same q_donor as the effective-state denominator.
2. Measure change in the emitted token and in **the same causal role** at L24(t+1), using a frozen reader and an independently justified temporal role mapping.
3. Repeat with the token fixed. The L24 difference should vanish (bitwise on a deterministic implementation or at a separately calibrated numerical floor).
4. Report the token mediation channel and state-normalized R separately. Do not identify K_eff with I(P;token) without a same-boundary equivalence proof.
5. Compute within each Q1 gauge; any positive certification must state its restricted causal contract.

## Counterexample for denominator substitution

Suppose P=(A,B), independent unbiased bits, is a two-bit effective state. The emitted token carries A only, and the next occurrence of the same role recovers A. Then B_reentry=1 bit; I(P;token)=1 bit; effective-state K_eff=2 bits. The token-mediated ratio is 1, but state-normalized R is 1/2.

## Scope of supplied GPU code

The supplied generic-prompt smoke test perturbs `layers[24].mlp.down_proj` forward *output* at t and captures L24 at t+1 under a forced token. It tests the lemma's structural prediction only. In particular, it is not an implementation of the T276/T286 frozen donor edits on world u26; that direct experiment must reuse the existing experimental preparation and frozen Q1 transport.
