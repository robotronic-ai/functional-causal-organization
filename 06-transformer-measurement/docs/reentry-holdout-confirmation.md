# Re-entry rank-signature discovery and candidate-fixed holdout confirmation (v0.4 → v0.5)

This document covers the second half of the re-entry discovery arc: replacing the
argmax (top-1) readout with an ordinal top-k signature (v0.4, exploratory/post-hoc
selection), then freezing the resulting top-2 candidate and testing it on entirely new
auxiliary keys and new return contexts (v0.5, candidate-fixed holdout — the strongest
reproducibility claim this campaign makes). See `reentry-measurement-protocol.md` for
the earlier v0.1–v0.3 arc that motivated this step.

## v0.4 — rank-signature protocol

Protocol: `Protocole_TRANSFORMER_REENTRY_RANK_SIGNATURE_v0_4_FROZEN.md`.

**Question.** v0.3 showed that adding four causally-accessible auxiliary keys does not
refine the top-1 quotient, even though that quotient captures only about 17.4% of the
raw return information. v0.4 asks whether the loss comes mainly from the argmax
compression itself. No numeric distance or probability threshold is introduced.

**Fixed objects.** Exactly v0.3's model, revision, source layer/position, 8 donors,
and 7 auxiliary keys (`K_1, K_2, K_3, E_0, E_1, E_2, E_3`) with fixed auxiliary filler
`F_1`, on the v0.2/v0.3 return holdout `(K_4,F_4), (K_4,F_5), (K_5,F_4), (K_5,F_5)`.
The runner must reproduce the same top-1 quotient as `C_7` from v0.3.

**Ordinal top-k signatures.** For each intervention and each auxiliary key, candidates
are sorted by decreasing probability (token-id tie-break only on exact floating-point
ties). Four nested signatures `C^(1), C^(2), C^(4), C^(8)` are built: for
`k ∈ {1,2,4,8}`, the signature keeps, per key, the ordered list of the `k` most
probable candidates. By construction `C^(2)` refines or equals `C^(1)`, `C^(4)`
refines or equals `C^(2)`, and `C^(8)` refines or equals `C^(4)`.

Rank is used because it is discrete, requires no post-hoc threshold, is invariant to
any common strictly monotone transform of a probe's scores, and enriches the argmax
readout gradually rather than jumping straight to full floating-point vectors. Margins
are reported but never used to drop observations.

**Required monotonicities.** Because the partitions are nested, `H(C^(1)) ≤ H(C^(2)) ≤
H(C^(4)) ≤ H(C^(8))`, and by the data-processing inequality `I(C^(k); Y|U)` is
non-decreasing in `k`. The runner invalidates the run if either relation is violated
by more than `1e-12` bit. No monotonicity of `R_k` itself is required.

### Result

Verdict `TR4-V1`, `RANK-REFINING` — the top-1 compression is strongly refuted as a
sufficient quotient.

**Top-1** (`|C^(1)| = 3`): `H = 1.2987949406953985 bit`, `B_1 = 0.0023373482510879065
bit`, `R_1 = 0.00179962839232839`, `η_1 = 0.17365189811429613`.

**Top-2** (`|C^(2)| = 7`): `H = 2.75 bit`, `B_2 = 0.011633131823477853 bit`,
`R_2 = 0.004230229753991946`, `η_2 = 0.8642766097095288`.

Partition: `{0} | {1} | {2} | {3,7} | {4} | {5} | {6}`.

**Top-4** (`= Q`, boundary/coherence check, not a substantive result on its own):
`H = 3 bit`, `B_4 = B_Q = 0.013459963734744118 bit`, `R_4 = 0.004486654578248039`,
`η_4 = 1`. Top-8 is identical to top-4 as a partition.

**Post-hoc robustness diagnostic** (descriptive, not part of the protocol's verdict):
the top-2 partition `{0}|{1}|{2}|{3,7}|{4}|{5}|{6}` is exactly reproduced using (1) the
three historical keys alone, (2) the four new keys alone, (3) all seven keys together,
and (4) each of the seven leave-one-key-out subsets. Interventions 3 and 7 share the
same ordered top-2 pair `(4, 7)` on every probe. Reported margins: minimum top1–top2 ≈
`0.0018049976817320623`; median top1–top2 ≈ `0.5751`; minimum top2–top3 ≈
`0.00015635884746536405`; median top2–top3 ≈ `0.02434`. The prospective candidate
carried forward is the ordered top-2 quotient.

---

## v0.5 — candidate-fixed top-2 holdout

Protocol: `Protocole_TRANSFORMER_REENTRY_TOP2_HOLDOUT_v0_5_FROZEN.md` (frozen
*after* v0.4, before v0.5 execution — this is the campaign's strong reproducibility
test: an independent fixture set generated after the candidate's rule is frozen, run
through the unmodified candidate rule).

**Candidate frozen.** The source equivalence is the **ordered top-2 signature** over
the seven auxiliary probes causally prior to the source readout. The historical
partition

```
Π_old_2 = {0} | {1} | {2} | {3,7} | {4} | {5} | {6}
```

becomes a prospective prediction for entirely new probes; the top-2 rule itself can no
longer change.

**New auxiliary battery.** After the 25 eligible tokens already used through v0.4, the
next seven eligible tokens `N_0 … N_6` are taken, used only as position-0 keys, with
probes `[N_j, A, G_A]` for a newly-selected auxiliary filler `G_A`. Readout stays at
position 1. For each probe, the two most probable candidates are kept **in order**,
giving the new quotient `C_new_2`.

**Source verdict.** `TOP2-HOLDOUT-PASS` if `Π_new_2 = Π_old_2` (versus
`TOP2-HOLDOUT-REFINED`, `-COARSENED`, or `-INCOMPATIBLE`). Only the PASS case counts as
prospective non-refutation of the candidate.

**New return holdout.** After the seven new auxiliary keys and filler, four more new
tokens are selected: two return keys `R_0, R_1` and two return fillers `S_0, S_1`,
never used in any auxiliary probe. Contexts: `(R_a, A, S_b, R_a)` for `a, b ∈ {0,1}` —
entirely new relative to v0.1–v0.4.

**Causal controls required:** `I(Q; Y_pre | U_new) < 1e-8 bit` and
`I(C_2_new; Y_pre | U_new) < 1e-8 bit`; label-duplication invariance re-tested.
Top-4 on the same new probes is computed as a descriptive control only.

### Result

**Verdict: `TOP2-HOLDOUT-PASS`.** The seven new auxiliary keys reproduce the historical
top-2 partition exactly.

On the four entirely new return contexts:

```
B_Q       = I(Q; Y_ret | U_new)        = 0.015099602235039825 bit
K_eff     = H(C)                        = 2.75 bit
B_reentry = I(C; Y_ret | U_new)         = 0.01362349150610347 bit
R                                        = 0.004953996911310352
η = I(C;Y_ret|U) / I(Q;Y_ret|U)          = 0.9022417474341858
residual  = I(Q; Y_ret | C, U_new)      = 0.001476110728936355 bit
```

Causal controls: `I(Q; Y_pre | U_new) = 6.93×10⁻¹⁶ bit`;
`I(C; Y_pre | U_new) = 7.04×10⁻¹⁶ bit` — both pass. Label-duplication invariance:
**PASS**. Top-4 on the new probes again becomes the identity, as a descriptive check
only.

**Two disjoint replications, not a confidence interval.** On the v0.4 holdout:
`R = 0.004230229753991946`, `η = 0.8642766097095288`. On the v0.5 holdout:
`R = 0.004953996911310352`, `η = 0.9022417474341858`. These are two point measurements
on two disjoint return batteries, not treated as bounds of a confidence interval.

## Closure statement (`POINT4A_TRANSFORMER_REENTRY_CLOSURE_v0_1.md`)

Status: **CLOSE-CONDITIONAL**.

Normalization used: `K_eff = H(C|U)`, `R = I(C;Y_ret|U) / H(C|U)`; since `C` is
independent of `U` on this campaign, `K_eff = H(C)`.

**Authorized claim statement:**

> On Pythia-70M-deduped, for the layer-2/position-1 hook, the frozen eight donors, and
> the top-2 source quotient validated on an independent holdout, a weak positive
> re-entry is reproduced on two disjoint return batteries, with about 86–90% of the raw
> return information captured by the quotient.

**Explicitly disallowed statements:** "R intrinsic to Pythia-70M = 0.004…" — the
measurement remains conditional on the hook, the donor battery, the candidate support,
the top-2 rule, and the specific return contexts.

**What follows.** The remaining target of the original transformer-measurement program
is a **closed latent loop** with an explicitly declared transport and a multi-round
measurement — covered in `latent-closed-loop-result.md`.
