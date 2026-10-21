# Certificate for Claude: nonzero Lambda0 on Qwen3-8B, L24

**Certified proposition**

`Lambda0_intrinsic_Qwen3_8B_C_L24 = NONZERO_CERTIFIED`

This is a certificate of **nonemptiness of the typed content-to-role lower relation** on the declared finite `C_L24` causal individual in world `u26`. The **primary threshold is strictly greater than delta_abs**, not twice delta. The latter is an optional stronger robustness control. This certificate does not assert complete identification of the relation, functional closure of C_O, or an all-layers result.

## Frozen causal contract

- **System**: Qwen3-8B; world `u26`, `SUBSTITUTE|H1`; L24 donor intervention, frozen L25 readers at one-layer horizon.
- **Causal sources**: 11 retained L24 sites, each with two frozen donors defining a distinct source content `z_i`. The remaining L24 site, `heads[24-27]`, is conditioned context.
- **Intervention law**: `q_donor = Bernoulli(0.5)^12` over 4,096 physical donor masks. No uniformization over quotient classes.
- **Destination roles**: 12 frozen L25 scalar site readers, including the one associated with the contextual donor site. At least 60 positive relations already connect two distinct retained roles.
- **Gauge family**: identity plus nine Q1 exact recodings (ten gauges in total).
- **Reader resolution**: `delta_k = max_{g in Q1, x} abs(Y_{g,k}(x) - Y_{identity,k}(x))`, with all 4,096 masks included.
- **Primary edge rule**: for a source site i and the *fixed* destination reader k,

  `L_delta(i,k) = 1` iff there exists one assignment `x_-i` of the other 11 physical donor bits such that **for each of the ten gauges**

  `abs(Y_g,k(x_-i,bit_i=1) - Y_g,k(x_-i,bit_i=0)) > delta_k`.

  A common assignment and a common destination reader are used in the reported witness for a particular edge. The comparison is strict; unmatched edges remain **unresolved**, not negative.

The source report contains an outdated prose description calling `2*delta` the primary threshold. That passage is **superseded** by the corrected contract above; the stored `relation_t_delta` and `witness_counts_t_delta` are the primary data, and `relation_t_two_delta` is secondary. The historic L23 L1 instrument continues using its own `2*delta_w` rule without modification.

## Verified archived evidence (u26, time t)

| Metric | At primary >delta | At secondary >2delta |
|---|---:|---:|
| Resolved positive source-reader cells (132 possible) | **68** | **39** |
| Positive source rows (11 retained sources) | **11** | 8 |
| Positive cells targeting retained readers | **66** | 38 |
| Positive cross-role cells with distinct retained source and destination | **60** | **33** |
| Shared donor assignments for `mlp3 -> mlp2` | **2,048/2,048** | **2,048/2,048** |

The archived report is internally consistent: every secondary witness is a primary witness, and the per-cell counts satisfy `0 < count_2delta <= count_delta <= 2048`. The `mlp3 -> mlp2` witness is cross-role and robust in every gauge with **the same assignment**, even at the stronger threshold.

At `t+1`, the archived report additionally has 27 cells at the *secondary* threshold, with 22 overlapping secondary cells at `t`. A primary-`delta` relation at `t+1` has **not** been submitted and is **not needed** to establish the nonzero typed-address relation at `t`.

## Mathematical conclusion

Let `L_delta` denote the lower relation consisting of the resolved-positive cells above. The reported primary `L_delta` contains at least 68 cells and, in particular, the distinct-role cell `(mlp3,mlp2)`. It is therefore nonempty. Every admissible completion of unresolved cells contains `L_delta` and remains nonempty. Admissible joint bijective relabeling of source contents and destination roles preserves nonemptiness. Therefore the organizational address relation is not the empty relation on this declared causal individual:

`Lambda0(Qwen3-8B | C_L24, u26, delta_abs, q_donor, Q1) != 0`.

**Decision:** `Lambda0_intrinsic_Qwen3_8B_C_L24 = NONZERO_CERTIFIED`.

This is a lower-relation certificate of non-nullity, not a claim that all negative cells, unique role selectivity, global P*, or global closure of C_O have been identified. There is no requirement to rerun Gamma, Delta, R, or the original restricted L23 Lambda0 certificate.

## Evidence and reproducibility

1. `T339_SOURCE_L24_RELATION_U26.json`: unchanged archived primary source from T337/T338, SHA-256 `a17cddb9febab9dfeacd9fc43f1e05377c058e1e4d44f320f5eca3e8be828489`.
2. `T339_SOURCE_L24_GAUGE_CUBE_U26_METADATA.json`: source acquisition metadata identifying the original time-`t` cube SHA-256 `8d421b29a240c61ba10ae005674bc27b2b4d4fcc4e575aa055d32f32d4da3cc8`.
3. `T339_SOURCE_L24_T1_GAUGE_CUBE_U26_METADATA.json`: time-`t+1` source metadata, supporting evidence only.
4. `T339_PRIOR_T338_DELTA_PRIMARY_AUDIT.json`: the existing independently checked primary-delta audit.
5. `T339_LAMBDA0_L24_DELTA_INDEPENDENT_AUDIT_CPU.py`: fresh independent report verification, and optional exact replay against the original time-`t` NPZ.
6. `T339_AUDIT_RESULT.json`: independent audit run bundled here.

**Evidence boundary**: The original raw FULL12 `.npz` was not supplied in this conversation, so the included audit verifies the archived relation, hashes of reports, witness-count consistency, and the explicit nonempty lower relation; **it does not claim fresh raw scalar remeasurement**. An optional `--raw PATH` reproduces all 264 source-reader count checks (132 pairs times two thresholds), after SHA-256 verification of the full NPZ. The source cube's provenance already matches the previously certified Gamma/Delta campaign.
