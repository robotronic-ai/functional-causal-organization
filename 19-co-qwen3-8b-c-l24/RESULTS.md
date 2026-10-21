# Results — Qwen3-8B, contract C_L24, world u26

Every value below is relative to [CONTRACT.md](CONTRACT.md). "Measured" means computed exhaustively on the finite
contract. "Certified" means that a lower bound has been verified independently of the measurement chain.

## Gauge cube

- **Size.** 4096 donor masks × 12 readers × 10 replicas (`results/gauge_cube_u26.json`, 1912.6 GPU-seconds).
- **Raw NPZ.** SHA-256 `8d421b29a240c61ba10ae005674bc27b2b4d4fcc4e575aa055d32f32d4da3cc8`.
- **Gauge resolution δ_abs** per reader, from 0.116 (mlp2) to 0.762 (heads[8-11]). All 12 readers are identifiable.

## Γ — NONZERO_CERTIFIED on C_L24

Source: `results/gamma_frozen_rule_u26.json`, `independent_verification/gamma/`.

- **Roles.** The frozen rule retains 11 roles; `heads[24-27]` is the context (0 witness configurations). Number of
  witness configurations x₋ᵢ per role:

  | heads[0-3] | heads[4-7] | heads[12-15] | heads[20-23] | heads[28-31] | mlp0–mlp3 | heads[8-11] | heads[16-19] |
  |---|---|---|---|---|---|---|---|
  | 1147 | 2048 | 2048 | 2048 | 2048 | 2048 each | 709 | 232 |

- **Measured Γ (δ_abs).** Computed over 1023 cuts × 2 contexts × 10 gauges:

  | Gauge | Γ |
  |---|---|
  | IDENTITY_0 | 0.29810 |
  | UPDOWN_SCALE_0 / 1 / 2 | 0.28906 / 0.30371 / 0.29712 |
  | VO_DIAG_0 / 1 / 2 | 0.30322 / 0.29785 / 0.30396 |
  | QKNORM_RECIP_0 / 1 / 2 | 0.29077 / 0.29688 / 0.30151 |

  - Minimum over the 10 gauges: **0.2890625**. Minimum per context: 0.2832.
  - Critical cut, the same in every gauge: `{heads[16-19]}` | the other ten roles.
  - Control with δ_contrast: minimum 0.00195.
- **Certified lower bound.** Γ ≥ **1.5673619086930833e-4**.
  - Each of the 10 230 gap witnesses (1023 cuts × 10 gauges) was verified against the raw NPZ, with a minimum
    scalar gap margin of 2.12e-5.
  - Status: `RAW_WITNESSES_VERIFIED`.
- **Scope.** Certified on C_L24 only. Not certified for other layers, horizons, intervention alphabets or causal
  individuals.

## Δ — NONZERO_CERTIFIED on C_L24

Sources: `independent_verification/delta/`, `results/delta_joint_closure_u26.json`.

| Convention | Δ_cap | Π_H | Π_S | Δ_H | Δ_S |
|---|---|---|---|---|---|
| per-reader (verified exact game, all 10 gauges) | 1 | 0.4 | 0.4 | **0.4** | **0.4** |
| any admissible refinement (theorem) | ≥ 1/11 | ≥ 474/68465 | ≥ 474/68465 | **≥ 6.2938595e-4** | **≥ 6.2938595e-4** |
| joint closure, δ_abs (min over 10 gauges) | 0.9682 | 0.7665 | 0.6703 | 0.7423 | 0.6501 |
| joint closure, δ_contrast (control) | 0.9815 | 0.7210 | 0.6494 | 0.7080 | 0.6379 |

- **Per-reader game.**
  - v(S) = 1 bit if S meets F = {heads[4-7], mlp0, mlp1, mlp2, mlp3}, and 0 otherwise.
  - Shapley value: 0.2 bit for each role of F.
  - Operational |P\*| = 2.
  - All 20 480 game entries (2048 coalitions × 10 gauges) were verified from the archived NPZ.
- **Refinement theorem.** It holds for any monotone refinement of the future readout that keeps the 11 roles, the
  donor alphabet and the original reader classes. It does not cover a change of causal boundary.
- **Joint closure.** v(N) = 9.739 bits (identity). Maximum over the gauges: Δ_H = 0.7477, Δ_S = 0.6566.

## R — local L24 re-entry on C_L24-R

Sources: `results/r_greedy_return_u26.json`, `results/r_token_channel_u26.json`, `results/r_sampled_return_u26.json`,
`results/r_return_bound_u26.json`, `results/role_identity_t1*_u26.json`, `independent_verification/r/`.

### Token-mediation lemma

It holds bitwise on the frozen u26 donors:
- greedy acquisition: checks (a) and (b), 32/32;
- sampled acquisition: arm cache + τ = clean cache + τ, 512/512 (8 masks × 64 tokens).

### Greedy transition: exactly zero

- **Census.** All 4096 donor masks emit token 19, under all 10 gauges, from the active token 23. I(P ; token | C) = 0,
  so R_greedy = 0 on C_L24.
- **Margins.** log p(19) − log p(21) ≥ 10.0 nats (median 14.25), and p(19) ≥ 0.9999546.

### Sampled transition (T = 1, full vocabulary, independent noises, k₀ = 1)

**Acquisition (6005 GPU-seconds).**
- **Guards.** The 10 frozen arms and the L25 scores are bit for bit equal to the gauge cube in all 10 replicas. The
  clean next token is 19 in all replicas.
- **Token set.** |V| = 64 (fixed before analysis). Mass outside V ≤ 3.1e-23.
- **Exact token channel.** I(X ; token) over the full vocabulary lies between 3.01e-6 and 5.80e-6 bit across the
  10 replicas.
- **Readability at t+1.** Next-step writes of the 12 sites: 1992 to 2016 of the 2016 token pairs of V are separated
  beyond δ_t1 (δ_t1 from 0.048 to 0.130).

**Return lower bounds** (primary declared precision: bf16 as run; minimum over the 10 replicas; worst-case mass outside V):

| Role | B_r ≥ (bit) | independent verifier, min over replicas | max over replicas | K_r bracket (bit) | R_tok ≥ | R_state ≥ |
|---|---|---|---|---|---|---|
| heads[0-3] | 7.4501e-08 | 7.4501e-08 | 1.3760e-07 | 7.450e-08 – 1.376e-07 | 0.541 | 6.773e-09 |
| heads[4-7] | 4.0638e-08 | 4.0638e-08 | 9.1996e-08 | 4.064e-08 – 9.200e-08 | 0.442 | 3.694e-09 |
| heads[12-15] | 2.2482e-08 | 2.2482e-08 | 3.9221e-08 | 2.248e-08 – 3.922e-08 | 0.573 | 2.044e-09 |
| **heads[20-23]** | **1.0523e-06** | **1.0523e-06** | 1.9915e-06 | 1.052e-06 – 1.991e-06 | 0.528 | **9.566e-08** |
| heads[28-31] | 3.4218e-07 | 3.4218e-07 | 6.6873e-07 | 3.422e-07 – 6.687e-07 | 0.512 | 3.111e-08 |
| mlp0 | 8.1238e-07 | 8.1238e-07 | 1.5461e-06 | 8.124e-07 – 1.546e-06 | 0.525 | 7.385e-08 |
| mlp1 | 3.5657e-08 | 3.5657e-08 | 5.0422e-08 | 3.566e-08 – 5.042e-08 | 0.707 | 3.242e-09 |
| mlp2 | 7.1340e-07 | 7.1340e-07 | 1.4448e-06 | 7.134e-07 – 1.445e-06 | 0.494 | 6.485e-08 |
| mlp3 | 4.1797e-08 | 4.1797e-08 | 6.8774e-08 | 4.180e-08 – 6.877e-08 | 0.608 | 3.800e-09 |
| heads[8-11] | 2.3040e-08 | 2.3040e-08 | 4.1541e-08 | 2.304e-08 – 4.154e-08 | 0.555 | 2.095e-09 |
| heads[16-19] | 2.9485e-08 | 2.9485e-08 | 4.8020e-08 | 2.949e-08 – 4.802e-08 | 0.614 | 2.680e-09 |

- **Positivity.** All 11 roles are positive in all 10 replicas.
- **Agreement of the two constructions.** The independent verifier (threshold events on a scalar return readout) and
  `r_return_bound_cpu.py` (balls in the full write space) give the same minimum for every role, to four significant
  digits.
- **Structure across replicas.** The ordering of the roles is the same in every replica.
- **Normalizations.**
  - R_tok is the conservative ratio B_r (minimum over replicas) / K_r (maximum over replicas).
  - Within a replica, B_r and the lower end of K_r coincide: the return event recovers the token channel almost
    entirely.
- **Robustness reading with δ_A.** With a gauge-noise threshold δ_A on event probabilities, there is no witness:
  δ_A = 3.95e-5 is the size of the whole non-clean probability mass. This reading is not the declared primary
  precision.

### Role identity t → t+1

Battery at the next active position: 2062 GPU-seconds, guards passed.
- **Retained roles.** At t+1 they are the same 11 roles except heads[16-19], which is absorbed into the context.
- **Per-reader δ-partitions.** They are identical at t and t+1, up to relabelling and in all 10 replicas and both
  contexts, for 11 of the 12 readers (NMI = 1). The exception is reader heads[4-7]: two classes at t, one at t+1.
- **Informative readers.** mlp0–mlp3 at both times; heads[4-7] only at t.
- **Strong sufficient test.** Not passed, because of heads[4-7] and heads[16-19].
- **Per-role closure signature.** Uninformative for 10 of the 11 roles: the scalar δ-closure resolves a single bit,
  that of mlp0. For mlp0, 4 of the 5 dependent readers are conserved; reader heads[4-7] drops at t+1.
- **Per-role effect signature** (frozen C_L24 inclusion criterion read reader by reader: ∃x ∀g, |Δ_i Y_{g,k}(x)| > δ_k,
  δ recalibrated at each time):

  | Role | Readers acted on at t | at t+1 | Jaccard | Identical |
  |---|---|---|---|---|
  | **mlp0** | 11 (all except heads[24-27]) | the same 11 | **1.00** | **yes** |
  | mlp1 | 9 | 9 | 0.80 | no |
  | heads[12-15] | 4 | 3 | 0.75 | no |
  | heads[20-23] | 6 | 4 | 0.67 | no |
  | mlp2 | 8 | 8 | 0.60 | no |
  | mlp3 | 8 | 9 | 0.55 | no |
  | heads[0-3] | 4 | 2 | 0.50 | no |
  | heads[4-7] | 8 | 3 | 0.38 | no |
  | heads[28-31] | 6 | 2 | 0.33 | no |
  | heads[8-11] | 2 | 1 | 0.00 | no |
  | heads[16-19] | 2 | 0 | 0.00 | no |

  This test was added after the closure-signature result; its criterion is the inclusion rule already declared in
  C_L24. Only mlp0 keeps an identical causal signature.

### Status

Proposed certificate, role mlp0 (L24 `down_proj` input, quarter block 0, active position):
1. **Return.** B_mlp0 ≥ 8.1238e-7 bit, the minimum over the 10 replicas. The independent verifier gives the same value.
2. **Role identity t → t+1.** The effect signature is identical (11 readers, 10 replicas, ∃x ∀g).
3. **No competing path.** The token-mediation lemma holds: 512/512 bitwise, plus the structural argument.
4. **Normalization.** Same source and same q_donor, 0 < K_eff ≤ 11 bits, hence **R_state ≥ 7.385e-8**. The
   token-channel ratio is R_tok ≥ 0.525.

Status:
- **R_local_L24 on C_L24-R: NONZERO_CERTIFIED, R_state ≥ 7.385298596317709e-8 > 0** (role mlp0).
- **Independent certificate:** `independent_verification/r/certificate/`.
  - B_mlp0 was recomputed per replica from the raw export, and the minimum matches the primary bound.
  - The mlp0 effect signature was recounted in the gauge cubes. The 49 witnesses of reader heads[16-19] at t+1 have a
    minimal margin of +1.18e-4.
  - The bypass controls were checked.
- **Not certified:** the δ_A robustness reading; a global or intrinsic R beyond the declared finite contract; the R of
  the terminal state.

## Λ₀ — lower content → role relation on C_L24

Sources: `results/lambda_relation_C_L24_u26.json` (SHA-256 `a17cddb9febab9dfeacd9fc43f1e05377c058e1e4d44f320f5eca3e8be828489`;
cubes: t `8d421b29…3cc8`, t+1 `51b9d5d7…47cf`), `results/lambda_witnesses_2delta_u26/`,
`independent_verification/lambda/T339_LAMBDA0_L24_DELTA_CERTIFICATE.md`.

Status: **NONZERO_CERTIFIED** at the primary threshold (strictly greater than δ_abs), by independent audit. The certificate
asserts non-emptiness of the typed content → role lower relation on C_L24 (u26). It does not assert complete identification
of the relation, unique role selectivity, functional closure of C_O or an all-layers result.

| Reading at t | Resolved positive cells (of 11 × 12 = 132) | Contents with a non-empty row | Cells between distinct retained roles |
|---|---|---|---|
| > δ (primary) | **68** | **11 of 11** | 60 |
| > 2δ (secondary) | 39 | 8 of 11 | 33 |

Every cell positive at 2δ is also positive at δ. Cell `mlp3 → mlp2` has all 2048 configurations as witnesses in every
gauge, at both thresholds. At t+1 the archived relation is at 2δ only (27 cells, 22 of them also positive at t); a primary
relation at t+1 was not submitted and is not needed for the claim.

Primary lower relation at t (> δ), with the number of witness configurations x (of 2048):

| Content (L24 role) | L25 roles reached |
|---|---|
| heads[0-3] | heads[12-15] (507), mlp0 (608), mlp1 (296), mlp2 (129) |
| heads[4-7] | heads[0-3] (71), heads[12-15] (392), mlp0 (12), mlp1 (532), mlp2 (26), mlp3 (2048), heads[16-19] (2047), heads[24-27] (76) |
| heads[12-15] | mlp0 (361), mlp2 (2048), mlp3 (920), heads[8-11] (1926) |
| heads[20-23] | heads[4-7] (2), heads[12-15] (2048), heads[28-31] (2048), mlp0 (1942), mlp1 (2048), mlp2 (2048) |
| heads[28-31] | heads[12-15] (1350), heads[28-31] (2048), mlp0 (853), mlp1 (1889), mlp2 (93), mlp3 (40) |
| mlp0 | heads[0-3] (2048), heads[4-7] (2048), heads[12-15] (2048), heads[20-23] (2048), heads[28-31] (1071), mlp0 (2048), mlp1 (2048), mlp2 (2048), mlp3 (2048), heads[8-11] (2048), heads[16-19] (2048) |
| mlp1 | heads[0-3] (3), heads[12-15] (2037), mlp0 (2048), mlp1 (2048), mlp2 (2048), mlp3 (2048), heads[8-11] (2048), heads[16-19] (1650), heads[24-27] (2048) |
| mlp2 | heads[0-3] (1973), heads[12-15] (632), heads[28-31] (2048), mlp0 (2048), mlp1 (1534), mlp2 (2048), mlp3 (2048), heads[16-19] (2048) |
| mlp3 | heads[0-3] (254), heads[4-7] (33), heads[12-15] (2048), mlp0 (2048), mlp1 (946), mlp2 (2048), mlp3 (2048), heads[8-11] (65) |
| heads[8-11] | mlp2 (546), heads[8-11] (80) |
| heads[16-19] | mlp0 (1), mlp3 (184) |

- **Evidence boundary of the audit.** The independent audit verified the archived relation, the report hashes and the
  consistency of the witness counts; it did not remeasure the raw cube. The raw check is supplied separately: the witness
  extract at 2δ (`results/lambda_witnesses_2delta_u26/`, from the original cubes, source SHA-256 in `summary.json`) gives for
  each of the 39 secondary cells at t a witness with its 20 raw scores and 10 differences, all above 2δ_k and hence above δ_k.
  Smallest margins over 2δ_k: at t, +0.0091 (heads[28-31] → heads[12-15], 3 witnesses) and +0.0109 (heads[28-31] → mlp0,
  6 witnesses); at t+1, +0.0046 (mlp1 → heads[24-27], 26 witnesses) and +0.0051 (heads[4-7] → mlp1, 4 witnesses). The claim
  does not depend on these cells: many cells have all 2048 configurations as witnesses with margins far above δ_k
  (for example mlp0 → mlp0, +7.55 over 2δ_k).
- **Descriptive properties of the labeled lower relation (t, > δ).** Row degrees [2, 2, 4, 4, 6, 6, 8, 8, 8, 9, 11]; 11 distinct row
  supports. Without negative cells these are not properties of Λ₀ itself.
- **Current script.** The relation computation of `scripts/lambda_relation_cpu.py` is unchanged since this file was produced;
  the current version reports δ as the primary reading, adds the t+1 relation at δ, and renames the status. The t-relations
  in the file are reproduced by it.
- **Not claimed.** Selectivity of Λ₀; a complete or point-identified typed relation; anything beyond the declared contents,
  destination roles and alphabet.

## Λ₀ — restricted verdict under the earlier L23/L24 anchor contract (separate threshold rule)

Sources: `historical_lambda0/`, `results/lambda0_restricted_replay.json`.

This verdict uses the threshold rule of its own instrument (e > 2 δ_w, unchanged); it is distinct from the C_L24 rule above. It was issued for a restricted domain (anchor roles, calibration battery u15–u18, holdouts
20–24 unused, bf16, 10 exact-recoding replicas). Its primary evidence is the pair of L1 runs `V_T135_L1_L23.json` and
`V_T135_L1_L24.json`. `lambda0_l23_replay_cpu.py` recomputes every quantity from the 20 per-replica raw write files
(SHA-256 in `VERIFY_NPZ_SHA.py`) and matches the recorded outputs exactly (tolerance 1e-9 relative), with the recorded
checkpoint signatures.

| Layer | δ_w | Content → anchor role (RESOLVED_POSITIVE) | Band e/δ_w over replicas | Unresolved cells |
|---|---|---|---|---|
| L23 | 0.108486 | u16\|REVOKE\|H2 → mlp3 | 3.244 – 3.383 | 4 of 6 |
| L23 | | u17\|SUBSTITUTE\|H2 → mlp3 | 3.311 – 3.559 | |
| L24 | 0.113525 | u16\|REVOKE\|H2 → heads[28-31] | 9.933 – 10.109 | 0 of 3 |
| L24 | | u17\|SUBSTITUTE\|H2 → heads[28-31] | 10.040 – 10.134 | |
| L24 | | u18\|REVOKE\|H1 → heads[28-31] | 3.621 – 4.248 | |

All content pairs are resolved distinct at both layers. The lower relation is non-empty, so Λ₀ is not identically zero
on this restricted domain. The replay was run on the original raw write files of the
measurement machine and on an independent copy of them: the two result files are identical, including the SHA-256 of
all 20 replica files and of both recorded JSON files.

## Replication on world u25 (optional)

Source: `results/replication_u25/`.

- **Roles.** 10 retained roles on u25 (u26 has 11); context `heads[16-19]`, `heads[24-27]`. This is a different causal
  individual from the 11-role C_L24 individual of u26.
- **Γ on this 10-role individual.** The frozen rule gives status `GAMMA_NOT_POSITIVE_ON_FROZEN_RULE_MACROSTATE`: the cut
  isolating `heads[8-11]` has zero integration under every gauge. This is a statement about the u25 10-role individual
  only. It is not a measurement of Γ for the u26 11-role contract and does not establish Γ = 0 for u25 under that contract.
  World u25 is not required by the contract.

## Results under other declared contracts

These results come from other interfaces. Except for Λ₀, their evidence is not part of this bundle.

| Item | Contract | Status |
|---|---|---|
| Λ₀ | L23/L24 anchor roles, calibration battery | NONZERO_CERTIFIED (primary evidence included and replayed, see above) |
| Δ (finite, L23) | 12 physical parts, L23 | Δ_H ≥ 0.0302, Δ_S ≥ 0.0164 |
| R_probe | 52 contexts, L23 token probe (u15) | R_probe = 1 |
| R_state (finite) | same probe, conditional on stated assumptions | ≥ 1/52 |

## Exploratory calibration pilot T346 (not part of the certified C_L24 statements)

Source: `exploratory_t346/` (README, scripts, episodes, results, manifest).

- **Question.** Does a graded activation attenuation of the `mlp0` role at layer 24 (input columns [0, 3072) of the layer-24
  `down_proj` multiplied by 1 - dose, doses 0.25 / 0.5 / 1.0, with a dose-0 SHAM arm) change two performance targets, the PCI_T
  differentiation score at layer 27 (`PCI_T_DS_L27`) and the FCA-Bench E1 score?
- **Scope.** World u26, frozen weights, six pilot episodes for PCI_T, one family. Descriptive only: no significance test, no
  monotone-law claim. The reduction of Δ under the ablation is not measured, so a null target response is not a test of C_O.
- **PCI_T branch (float16).** SHAM is identical to the unhooked baseline. The manipulation reaches layer 27 (source relative change
  2.9 %, 5.8 %, 11.8 % at doses 0.25, 0.5, 1.0). The median paired DS differences against SHAM are -0.002, -0.023 and +0.016; none exceeds
  the descriptive band of 0.10 and they are not monotone in the dose. The same run in bfloat16 saturates the instrument (baseline DS 1.049
  instead of 1.465) and is kept as a diagnostic only.
- **FCA_E1 branch (bnb8, float16 compute, historical R6 configuration, 20 instances).** Sham is bit-identical to the unhooked baseline on all instances and all 100 instance-arms are valid. FCA_E1 is 0.6875 at baseline and 0.6857, 0.6772, 0.6888 at doses 0.25, 0.5, 1.0; the mean paired differences against sham are -0.0018, -0.0103, +0.0013 (descriptive intervals all contain 0, none exceeds the band of 0.05, no monotone trend). The run took about 3.98 h (measured instance time). As for PCI_T, this is a descriptive null for a local-strength perturbation.
- **FCA_E1 in bfloat16 without quantization (same script, instances and seeds).** Baseline 0.7515; mean paired differences against sham +0.044, +0.053, +0.053 at doses 0.25, 0.5, 1.0 (descriptive intervals above 0; 16, 18, 17 of 20 instances positive), entirely through the distributional part of the score. The effect of the same zero-attenuation therefore depends on the numerical regime (no effect in bnb8 / float16). Descriptive, one world, no inference.
- **T347 float16 transport of C_L24 (u26, operator run, not certified).** Gauge cube acquired in float16 (GPU 1886.9 s); Gamma is 0.5 on all 10 gauges (critical cut `heads[0,1,2,3]`, 12 retained sites), Delta_cap 1.0 with Pi_H 0.953-0.955 and Pi_S 0.916-0.919 (delta_abs; delta_contrast 0.939-0.940 and 0.896-0.897). The numerical tolerance differs from the bf16 certification (gauge deltas 0.014-0.056 versus 0.116-0.76), so the values are not compared like for like; Gamma = 0.5 is the floor of the cut table and Delta_cap is saturated. Status `AWAITING_INDEPENDENT_CERTIFICATE`; no formal promotion. Files: `exploratory_t346/results/t347_float16_cl24_u26/`.
- **Targeted FCA_E1 calibration, bfloat16 without quantization (6 instances, descriptive).** SHAM identical to BASELINE (0.7705). Mean paired differences against sham: zeroing the mlp0 channels +0.032, replacing them by reference means +0.034, replacing all MLP channels of layer 24 by reference means +0.063 (interval above 0), content substitution of matched length -0.008 (about 8 times weaker at layer 27), strength-matched random perturbation -0.025 (5 of 6 negative). The ZERO arm reproduces the T353 differences on the same six instances. The effect of Delta under these interventions was not measured, so no mediation claim is made. Files: `exploratory_t346/results/fca_e1_calibration_bf16_V2/`.
- **Classification.** The manipulation is an unqualified local-strength perturbation: its effect on Γ, Δ, R, Λ was not measured, so this pilot is not a test of the C_O bridge. A specificity panel (sham, C_O target, strength-matched, generic-state-matched, gauge control, optional readout control) is planned; it has not been executed.
- **Status effect.** None. The statuses of Γ, Δ, R and Λ₀ on C_L24 are unchanged, and functional closure remains open.
