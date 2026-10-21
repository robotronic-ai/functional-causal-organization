# The C_L24 contract

Everything reported in this bundle is relative to the finite causal contract below. It is declared before the
measurements it governs and is the same for Γ, Δ, R and Λ₀, except where the R and Λ₀ extensions say otherwise.

## System and individual

- **Model.** Qwen3-8B, bf16 weights, single CUDA device, deterministic kernels as provided by the pinned runtime.
- **World.** u26, case `SUBSTITUTE|H1` of the pinned development schedule. Only this world is required. World u25
  is reported as an optional replication.
- **Active position.** The first post-cycle token (token id 23 in u26), processed at time t.
- **Interface.** Layer 24 (intervened) → layer 25 (read), at the active position.

## Sites, donors and the intervention law

- **Twelve L24 sites.**
  - Eight KV head groups `heads[4g..4g+3]` (g = 0..7), at the input of `self_attn.o_proj`.
  - Four quarter blocks `mlp0..mlp3` of the input of `mlp.down_proj`.
- **Mask bit order.**
  - Bits 0..8: `heads[0-3], heads[4-7], heads[12-15], heads[20-23], heads[28-31], mlp0, mlp1, mlp2, mlp3`.
  - Bits 9..11: `heads[8-11], heads[16-19], heads[24-27]`.
- **Frozen donors.** Each site has exactly two donor values.
  - The first nine sites use frozen donor arrays.
  - The last three use frozen background pairs taken from the D0 worlds.
  - An intervention overwrites the site's slice of the `o_proj` / `down_proj` input at the active position.
- **Intervention law.** q_donor = independent Bernoulli(0.5) on the 12 bits, so q(x) = 2⁻¹² for each of the
  4096 masks. No class balancing and no reweighting.

## Readers

There are twelve frozen L25 readers, one per site. Reader r is the scalar `Y_r = (w_r − o_r) · d_r`, where:
- w_r is the residual-space write of site r at layer 25;
- the origin o_r and the direction d_r are frozen.

## Gauges and resolution

- **Q1 gauge replicas.** The identity plus nine exact recodings of layer 24: `UPDOWN_SCALE`, `VO_DIAG` and
  `QKNORM_RECIP`, each with seeds 0, 1 and 2. They preserve the function in exact arithmetic. Donor values are
  transported into each recoding.
- **Resolution.**
  - Primary: `δ_abs,r = max_g max_x |Y_{g,r}(x) − Y_{id,r}(x)|` over the nine recodings and the 4096 masks.
  - Control: the contrast tolerance `δ_contrast,r`, the largest range across masks of the gauge deviation.
- **Comparisons.** Every comparison uses a strict inequality "more than δ", read separately under each gauge.

## Roles: frozen inclusion rule

Site i is a **role** if there exists x₋ᵢ such that, for every gauge g, some reader r satisfies
`|Δ_i Y_{g,r}(x)| > δ_r`. In short: ∃x ∀g ∃r. Sites that fail the rule are absorbed and conditioned on as context.

On u26 this gives **11 roles**, with `heads[24-27]` as the single context site.

## Γ on C_L24

For every gauge separately, every context, and every bipartition A|B of the roles (1023 cuts):
- **Readout side.** J(A→B) is read on the readers of B (the destination side), and J(B→A) on those of A.
- **Classes.** Source assignments are grouped by the connected components of the "all destination readers within
  δ" relation (local joint closure). J is the entropy of these classes under q_donor, averaged over the fixed-side
  assignments.
- **Cut value.** `(J(A→B) + J(B→A)) / (2 min(|A|,|B|))`, averaged over contexts.
- **Γ.** The minimum over cuts, reported per gauge and as the minimum over the 10 gauges.

## Δ on C_L24 (C_L24-Δ)

- **Variables.**
  - P is the configuration of the 11 roles: 2048 equiprobable configurations per context.
  - Y_S is the readout of the readers of the roles in S.
- **Game.** v(S) = I(P ; Y_S | C).
- **Formulas.**
  - Δ_cap = v(N) / log₂|P\*|.
  - φ is the exact Shapley vector over the 2048 coalitions, and s = φ / v(N).
  - Π_H = (2^{H(s)} − 1)/(m − 1) and Π_S = (1/Σ s² − 1)/(m − 1), with m = 11.
  - Δ_H = Δ_cap · Π_H and Δ_S = Δ_cap · Π_S.
- **Two class conventions** for Y_S, computed separately per gauge and per context:
  - *per-reader*: Y_S is the tuple of the scalar δ-component labels of the individual readers in S;
  - *joint closure*: Y_S induces the connected components of "all readers of S within δ", the same closure as Γ.

## R on C_L24 (C_L24-R)

- **Transition.**
  - Sampled (primary): T_t ~ Categorical(softmax(logits_t)), temperature 1, full vocabulary, no top-k, no top-p,
    no other logit warper.
  - Greedy: argmax, reported as a separate case.
- **Declared precision (primary).** The system is Qwen3-8B as run in bf16: the model probabilities are exact values of
  the declared system, post-processed in float64 (numerical floor 1e-12 bit). A positive lower bound is required in each
  of the 10 replicas. A gauge-noise threshold δ_A on event probabilities is reported as a robustness reading only.
- **Neutralisation.** Independent noises: each counterfactual world draws its own token (kernel functional).
- **Horizon.** k₀ = 1 transition, declared before measurement.
- **Scope.** Local re-entry L24(t) → token → L24(t+1); not the R of the terminal state.
- **Intervention law.** q_donor is unchanged.
- **Next-step state of role r.** The residual-space write of site r at layer 24 on the next token, at time t+1.
  - Resolution: `δ_t1,r`, the maximal deviation of the nine recodings from the identity.
- **Return.**
  - B_r = I(P_r ; [state of role r at t+1] | P₋ᵣ, C): information from role r's own donor bit back into role r
    one cycle later.
  - K_r = I(P_r ; T | P₋ᵣ, C): information carried by the token channel.
  - Two ratios, reported separately:
    - **token-channel ratio**, R_tok,r = B_r / K_r;
    - **state-normalized ratio**, R_state,r = B_r / K_state. This bundle reports the lower bound B_r / 11 bits,
      valid for any state information about P on this alphabet, because H(P | C) = 11 bits.
- **Role identity t → t+1.** The same donor battery is applied at the next active position (clean next token).
  - Strong sufficient test: same retained roles and identical per-reader δ-partitions at t and t+1, up to relabelling.
  - Per-role test: the readers on which role r depends, in every replica and context, are the same at t and t+1.
  - A failure of the strong test is not by itself a refutation of the functional role identity.

## Λ₀ on C_L24 (C_L24-Λ)

- **Contents.** z_i is the frozen donor distinction at retained L24 role i (its two frozen donor values), for the 11
  retained roles. Distinct contents act on disjoint sites, so every pair of contents is resolved.
- **Roles (destination).** The 12 L25 sites, each seen through its frozen reader Y_k.
- **Cell rule (primary).** (z_i, k) is RESOLVED_POSITIVE when there exists a configuration x of the other donor bits
  such that, in every one of the 10 replicas, |Y_{g,k}(x^i) − Y_{g,k}(x)| > δ_k, with the same x and the same reader in every
  replica (x^i is x with bit i set). δ_k is δ_abs of reader k, the same strict "more than δ" comparison as the other
  components. A reading with 2 δ_k is reported as a stricter secondary variant.
- **No negative cell.** An unresolved cell is never evidence that content z_i does not reach role k.
- **Claim.** Λ₀ ≠ 0 when the lower relation is non-empty: every admissible completion of the unresolved cells contains
  it, so no completion makes Λ₀ identically zero. Degree multisets, row supports and one-sided permutation sensitivity
  are descriptive properties of the labeled lower relation only; selectivity of Λ₀ itself is not claimed.
- **Times.** The relation is read at t (gauge cube) and, with the same contents, at t+1 (role-identity cube).
