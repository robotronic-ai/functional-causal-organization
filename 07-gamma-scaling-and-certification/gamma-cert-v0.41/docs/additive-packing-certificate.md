# Additive fractional-packing and spectral certificates for Γ at scale

Translated and restated from the source note `GAMMA_CERT_V04_NOTE.md` (French). The mathematics,
theorem statements, and reported numbers are unchanged; nothing about the definition of Γ is
edited or proposed for the manuscript.

## Motivation: why v0.3's max-only witness does not scale

The v0.3 hierarchy, built on the **maximum** of a contextual witness, is correct but cannot, at
fixed witness-block order, certify a non-vanishing Γ as the number of roles `n` grows. If a
source witness contains at most `k` bits, a certificate retaining only the best witness per
direction is at most order `O(k/n)` on a balanced cut. An **additive, non-double-counting**
construction is required instead.

## I. Transportable conditional witness

Under a product intervention battery `q(x) = ∏ᵢ qᵢ(xᵢ)`, for three disjoint sets `S, T, C`,
define

\[
d_q(S\to T\mid C) = I_q(X_S; X'_T \mid X_C).
\]

For a cut `A|B` with `S ⊆ A` and `T∪C ⊆ B`, and using `J_q(A\to B) = I_q(X_A; X'_B \mid X_B)`
in the product-interventional representation: by projection,
`I(X_A;X'_B∣X_B) ≥ I(X_S;X'_T∣X_B)`; and since `X_S` is independent of `X_{B∖C}` given `X_C`
under a product `q`, `I(X_S;X'_T∣X_B) ≥ I(X_S;X'_T∣X_C)`. Hence

\[
\boxed{J_q(A\to B) \ge d_q(S\to T\mid C).}
\]

The witness is measured under a single global battery — it is never redefined after observing
the cut.

## II. Fractional packing theorem

For a family of active witnesses `𝒲_{A,B} = {(Sw,Tw,Cw,dw) : Sw⊆A, Tw∪Cw⊆B}`, choose
coefficients `αw ≥ 0` such that every source coordinate `i∈A` has total packing weight
`∑_{w: i∈Sw} αw ≤ 1`. Then

\[
\boxed{J_q(A\to B) \ge \sum_{w \in \mathcal W_{A,B}} \alpha_w\, d_w.}
\]

**Why the sum does not double-count.** Complete the packing with singleton witnesses of the
missing weight to obtain a fractional partition of the source set. Under product `q`,
`H(X_A∣X_B) = ∑_{i∈A} H(Xᵢ)`; the fractional Shearer/Madiman–Tetali entropy inequality applied
to `H(X_S∣X'_B,X_B)` gives the needed opposite inequality on conditional entropies. Subtracting
the two: `I(X_A;X'_B∣X_B) ≥ ∑w αw I(X_{Sw};X'_B∣X_B) ≥ ∑w αw dw`. The source of additivity is the
independence of the intervention coordinates plus one entropy inequality — not an independence
assumption on the outputs.

## III. Witness estimation by decoding rather than high-dimensional MI

A witness `d(S→T∣C)` can itself be lower-bounded without estimating a high-dimensional joint
distribution directly. Freeze a decoder `X̂_S = g(X'_T, X_C)`; let `p` be its classification
error on an independent holdout, `K = |𝒳_S|`. Fano gives
`H(X_S∣X'_T,X_C) ≤ h₂(p) + p·log₂(K-1)`, and since `X_S ⊥ X_C` under product `q`,

\[
\boxed{d(S\to T\mid C) \ge H(X_S) - h_2(p) - p\log_2(K-1).}
\]

Substituting a binomial UCB `p^U` (from holdout) for `p` gives a statistical LCB of the witness.
This lets a multi-bit source block contribute several certified bits without ever estimating a
mutual information over the system's full alphabet.

## IV. Global optimization of the certificate (exact dual MILP)

For a fixed cut, the best directional packing is the linear program
`P(A,B) = max_{α≥0} ∑w αw dw` subject to the source-capacity constraints. Its dual assigns a
price `λᵢ ≥ 0` to each source coordinate: `P(A,B) = min_{λ≥0} ∑_{i∈A} λᵢ` subject to
`∑_{i∈Sw} λᵢ ≥ dw` for every active witness. Introducing binary cut-side variables `xᵢ` (so
constraints activate only when `Sw⊆A, Tw∪Cw⊆B`) gives an exact global MILP for the **lower
bound**, not for Γ itself:

\[
L_{\mathrm{pack}} = \min_{A|B} \frac{P(A,B)+P(B,A)}{2\min(\log|A|,\log|B|)} \le \Gamma_q.
\]

Autonomous validation (`results/fractional_packing_results.json`) confirms the dual MILP
reproduces the packing enumeration exactly on every tested small system.

## V. Decoding-graph corollary and a one-sided spectral certificate

The singleton case `S={i}, T={j}, C=∅` gives certified weights `d_ij ≤ I(Xᵢ;X'_j)`. Retaining at
most `kᵢ` targets per source, for any cut: `J(A→B) ≥ ∑_{i∈A} max_{j∈B} d_ij ≥ ∑_{i∈A,j∈B} d_ij/kᵢ`.
Adding the reverse direction, define the certified undirected graph weight
`w_ij = d_ij/kᵢ + d_ji/kⱼ`. Then `J(A→B) + J(B→A) ≥ cut_W(A,B)`, and for binary roles

\[
\Gamma_q \ge \min_{A|B} \frac{\mathrm{cut}_W(A,B)}{2\min(|A|,|B|)}.
\]

With `δ_min` the minimum weighted degree and `φ_W` the conductance of `W`,
`Γ_q ≥ (δ_min/2)·φ_W`; Cheeger's inequality for the normalized Laplacian gives
`φ_W ≥ λ₂/2`, hence

\[
\boxed{\Gamma_q \ge \frac{\delta_{\min}\,\lambda_2}{4}.}
\]

This is a **one-sided spectral certificate of Γ**, because the spectral graph is now a proven
additive causal sub-model — not the older, weaker claim "spectral approximates Γ."

## VI. Copy-expander causal benchmark

See `copy-expander-scale-benchmark.md` for the construction where Γ is *exactly* the conductance
of a `d`-regular graph, so Cheeger applies directly to Γ, and for the calibration/scaling results
up to `n = 1024` roles.

## VII. What this changes for the experimental program: three stages

1. **Fast graph certificate.** Build decoding weights with holdout LCBs. If `δ_min λ₂/4` is
   already high enough, Γ has an immediate positive lower bound at very large scale.
2. **Block packing.** If synergistic interactions make singletons weak, add `(S,T,C)` witnesses
   and solve the fractional-packing MILP.
3. **Local refinement.** Only if `[L,U]` remains too wide, increase block size on the specific
   regions of the candidate cut that control the gap.

The block order no longer needs to grow everywhere with `n`.

## VIII. Scientific status

The definition of Γ is unchanged. The new result is a family of **certified, extensive lower
bounds**: (1) graph-decoder + Cheeger, very scalable, potentially looser; (2) fractional block
packing, more general and tighter; (3) the full-block limit, which recovers exact Γ. The spectral
bound regains a role — as the certificate of a proven causal sub-graph, never as a direct
approximation of Γ.

## Executed results: fractional-packing tightness

`scripts/gamma_fractional_packing_autonomous.py` runs the packing certificate on four
finite systems (`ring_copy_6`, `xor_neighbors_6`, `mixed_boolean_6`, `random_boolean_6`) at four
witness-block configurations `(max_source, max_target, max_context)`, and asserts the dual MILP
exactly matches exhaustive enumeration on every case (`dual_MILP_matches_exhaustive: true`
throughout). Selected rows (full table: `results/fractional_packing_results.json`):

| System | Block config `(S,T,C)` | `Γ_exact` | `L_fractional` | fraction of Γ certified |
|---|---|---:|---:|---:|
| ring_copy_6 | (1,1,1) | 0.3333 | 0.3333 | 1.00 |
| xor_neighbors_6 | (1,1,1) | 0.6667 | 0.0000 | 0.00 |
| xor_neighbors_6 | (3,2,1) | 0.6667 | 0.6667 | 1.00 |
| mixed_boolean_6 | (1,1,1) | 0.6093 | 0.1556 | 0.26 |
| mixed_boolean_6 | (3,2,1) | 0.6093 | 0.5519 | 0.91 |
| random_boolean_6 | (1,1,1) | 0.6800 | 0.0447 | 0.07 |
| random_boolean_6 | (3,2,1) | 0.6800 | 0.1687 | 0.25 |

This confirms the theorem numerically and shows its practical behavior directly: singleton
witnesses alone can certify `0%` of Γ on a purely synergistic system (`xor_neighbors_6` at block
order 1), while widening the witness block to order `(3,2,1)` recovers the exact value on that
same system. `random_boolean_6` (no exploitable low-order structure) remains loosely certified
even at block order `(3,2,1)` (`25%`), illustrating stage 3 ("local refinement") is genuinely
needed for systems without small sufficient witnesses.
