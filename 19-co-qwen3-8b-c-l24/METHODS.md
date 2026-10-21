# Methods

## Acquisition and bit-level guards

`scripts/acquire_gauge_cube_gpu.py` measures the 12 L25 reader scores for the 4096 donor masks under each of the
10 gauge replicas. It uses a two-layer window (L24 → L25) on the cached prefix of the active position.

Guards:
- **Frozen arms.** The ten frozen historical arms are replayed bit for bit (writes of the 12 sites).
- **Reference sub-cube.** On u26, the identity sub-cube of 512 arms × 6 readers equals the pinned reference
  sub-cube bit for bit.
- **Gauge sub-cubes.** When the reference gauge sub-cubes are present, every recoded replica is checked against them.

The R acquisitions (`r_greedy_return_gpu.py`, `r_sampled_return_gpu.py`) instead run full forwards of the model at t
and t+1. They first check that the full forward reproduces the window scores, and the gauge cube, bit for bit.

## Γ certificate

For a cut A → B, fix a destination-side assignment and the context. Under q_donor this fixed configuration has
probability 2^-(|B|+|context|).

Suppose the sorted values of one destination reader, over the 2^|A| source assignments, contain an adjacent gap
larger than δ. Then no δ-edge crosses the gap, and the local closure cannot cross it either. The induced binary
partition contributes h₂(k / 2^|A|), where k is the number of source assignments below the gap. Hence:

```
J(A → B) ≥ 2^-(|B|+|context|) · h₂(k / 2^|A|)
```

Dividing by 2 min(|A|,|B|) gives a lower bound for each cut and each gauge. The independent verifier
(`independent_verification/gamma/T322_VERIFY_U26_GAMMA_CERT_CPU.py`) rechecks all 10 230 witness records
(1023 cuts × 10 gauges) against the raw NPZ: cut enumeration, reader placement, gap inequalities, entropies and minima.

## Δ

**Per-reader convention.** The game is computed exactly over the 2048 coalitions for each gauge, together with its
Shapley vector (`independent_verification/delta/T325_*`). An independent verifier replays the full game from the
archived NPZ.

**Refinement theorem** (`T327_DELTA_FINITE_NONZERO_PROOF_CPU.py`). Five distinct roles each carry at least 1 bit in
their own reader. For any monotone refinement of the readout that preserves the 11 roles, the donor alphabet and the
original reader classes:
- each of these five roles keeps a Shapley value ≥ 1/11 bit, through the empty-coalition term;
- each normalized share is ≥ 1/121;
- Σ s² ≤ 13693/14641.

This gives Π_H, Π_S ≥ 474/68465. With Δ_cap ≥ 1/11, it follows that Δ_H, Δ_S ≥ 474/753115 ≈ 6.29e-4, without
identifying the refined P\* exactly.

**Joint closure convention** (`scripts/delta_joint_closure_cpu.py`). The coalition values are obtained by depth-first
intersection of the δ-adjacency matrices: adj(S ∪ {r}) = adj(S) ∧ A_r. The entropies come from the connected
components, and the Shapley values are exact over the 2048 coalitions, for each gauge.

## R

**Token-mediation lemma.**
- *Setting.* In a strictly causal decoder-only transformer, let an intervention at time t modify only the inputs of
  `o_proj` / `down_proj` of layer 24 at the active position.
- *Why the token is the only path.*
  - The layer-24 keys and values at t are computed before these writes.
  - Every lower-layer cache is computed earlier still.
  - Layer 24 at t+1 never reads the caches of higher layers.
- *Consequence.* With the emitted token held fixed, L24(t+1) is identical whatever the intervention: the only path
  from L24(t) to L24(t+1) is the emitted token. Upper-layer caches and final logits can still differ.
- *Checks.*
  - Bitwise, on a generic prompt (`independent_verification/r/`).
  - On the frozen u26 donors, in both acquisitions: (a) arm cache + its token = clean cache + same token; (b) arm
    cache + clean token = clean cache + clean token.

**Consequence for the measurement.** The state of role r at t+1 is a deterministic function f_r(τ) of the emitted
token. Under the sampled transition, the law of the return is therefore the exact pushforward of p(τ | x) through f_r.
No sampling is needed.

**Exact channel.** `r_sampled_return_gpu.py` computes, in float64 over the whole vocabulary, for every mask and every
replica:
- log p(τ | x) on a fixed token set V (fixed before acquisition);
- the probability mass outside V, as a direct sum;
- H(p_x);
- the exact I(X ; T) and I(X ; T | bit), by accumulating p_x over the vocabulary.

It also computes f_r(τ) for every τ in V, under the 10 replicas.

**Worst-case lower bound** (`r_return_bound_cpu.py`).
- *Return event.* A = {state of role r at t+1 within δ_t1,r of the state reached through a reference token ρ}. A is
  a deterministic function of the state of role r.
- *Interval.* For tokens outside V, membership in A is unknown. P(A | x) is therefore confined to an interval of
  width equal to the mass outside V.
- *Bound.* B_r is bounded below by the mean over x₋ᵣ of the minimal binary Jensen–Shannon divergence over the two
  intervals. This minimum is exact because JS is convex and vanishes on the diagonal.
- *Primary bound (declared precision: bf16 as run).* B_r is the minimum over the 10 replicas of this bound, with
  the reference token ρ maximizing it. A float64 post-processing floor of 1e-12 bit is declared.
- *Robustness reading.* Witness pairs whose intervals are separated by more than the gauge deviation δ_A of P(A | x)
  under every replica (∃x ∀g), or within each replica (∀g ∃x); reported separately, not part of the primary bound.
- *Token channel.* K_r is bracketed by the fixed grouping (V plus one bucket for the rest) and the mass outside V.

**Independent cross-check of the return bound.** `export_t330_schema_cpu.py` converts the acquisition into the schema
of the independent verifier `independent_verification/r/T330_S27_DETERMINISTIC_R_BOUNDS_CPU.py`, with a fixed scalar
return readout per role: the relative L2 distance of the role-r L24 write at t+1 to the clean next-token write. The
verifier uses threshold events at observed gaps larger than δ_t1 of this readout, while `r_return_bound_cpu.py` uses
balls of radius δ_t1 around a reference token in the full write space; the two constructions are independent.

**Role identity t → t+1.** `role_identity_t1_gpu.py` applies the frozen donor battery at the next active position:
the clean prefix includes the clean active token at t, and the active token at t+1 is the clean next token, checked in
every replica. `role_identity_t1_cpu.py` compares the frozen inclusion rule and the per-reader scalar δ-partitions at t
and t+1 (δ recalibrated at each time). `role_signature_t1_cpu.py` compares, role by role, the set of readers whose
δ-partition depends on the role in every replica and context.

## Λ₀

**Lower relation on C_L24** (`scripts/lambda_relation_cpu.py`). For each retained content i and each reader k, the
differences Y_{g,k}(x^i) − Y_{g,k}(x) are formed over the 2048 configurations x with bit i = 0 and the 10 replicas. A
cell is positive when at least one x exceeds δ_k in absolute value in all 10 replicas (primary), or 2 δ_k (secondary). The
number of such x is the witness count. The same computation is applied to the cube at t+1 with the contents retained at t.
`scripts/extract_lambda_witnesses_cpu.py` (option `--factor 1` or `2`) writes, for every positive cell, the witness with the
largest margin together with its 20 raw scores, its 10 differences, δ_k and the replica and mask at which δ_k is attained,
so that each cell and each δ_k can be checked by hand from the cube.

**Restricted Λ₀ replay** (`scripts/lambda0_l23_replay_cpu.py`). The L1 instrument (ROLELAMBDA_v1,
`historical_lambda0/instrument/role_lambda.py`) records, for each replica, the write of each anchor role at its layer
for every calibration context t, with the residual entering the layer replaced by the reference content or by a test
content. With rel(a, b) = rms(a − b) / (½(rms a + rms b)):
- δ_w is the maximal rel between any two replicas over every recorded write (orbit diameter); δ_x likewise for the
  content vectors;
- e_g(z, r) = max_t rel(write | z, write | reference) in replica g;
- the cell (z, r) is RESOLVED_POSITIVE when e_g(z, r) > 2 δ_w in every replica, otherwise UNRESOLVED;
- contents z, z' are distinct when rel(x_z, x_z') > 2 δ_x in every replica.
The replay recomputes all of these with numpy from the per-replica raw writes and compares them with the recorded
outputs; it does not import the instrument.
