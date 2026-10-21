# Re-entry measurement protocol (v0.1 → v0.3)

This document covers the discovery arc of the transformer re-entry measurement: the
first frozen protocol (v0.1), a robustness follow-up that surfaced and archived a real
bug (v0.2), and the bug-fixed repeat (v0.3). See `reentry-holdout-confirmation.md` for
the later rank-signature discovery (v0.4) and its candidate-fixed holdout (v0.5).

## 0. Frozen model

- Model: `EleutherAI/pythia-70m-deduped`
- Revision: `5ff092d907c3d8ba420d9fbef792426789eb2cd2`
- Expected architecture: `GPTNeoXForCausalLM`, 6 layers, hidden size 512, 8 attention
  heads. The runner refuses to continue if these properties do not match.

## 1. Hypothesis tested

The test does not attempt to demonstrate "consciousness" in the model. It tests only
whether a discrete causal intervention injected into an intermediate residual state can
be recovered at a later, functionally "return"-like position.

Operational chain:

```
Q → C → Y_ret → B_reentry → K_eff → R
```

## 2. Source hook

Source layer `L_s = 2` (0-indexed, third of six layers). The intervention replaces the
entire residual output vector (dimension 512) at source position `p_s = 1` with the
corresponding vector captured from a donor run. No internal coordinate of the vector is
selected — the whole residual vector is swapped.

## 3. Experimental vocabulary

Using the frozen model's exact tokenizer, the runner deterministically selects:

- 8 candidate tokens `V_0 … V_7`
- 6 key tokens `K_0 … K_5`
- 6 filler tokens `F_0 … F_5`
- 1 anchor token `A`

Selection criteria, applied in order: non-special token; printable text decoding;
exact tokenizer round-trip to a single token; matches the runner's declared regex
form; ascending token id. The result file publishes the exact ids and strings
selected. The runner aborts if the expected vocabulary cannot be constructed.

## 4. Source interventions Q

For each candidate `V_i`, build the canonical donor run `[K_0, V_i, F_0]` and capture
the output vector of layer `L_s` at `p_s = 1`. This finite collection of eight vectors
defines the interventions `Q = {q_0, …, q_7}`, with a uniform battery distribution
`q(Q)`.

## 5. Source quotient C — built without touching Y_ret

The quotient is constructed *before* any measurement on the return channel. Three
auxiliary prompts are frozen: `[K_1, A, F_1]`, `[K_2, A, F_2]`, `[K_3, A, F_3]`. For
each intervention `q_i`: patch the donor vector at `L_s, p_s`; read the logits **at
the source position `p_s`**; restrict to the 8 candidate tokens; record the argmax
candidate. The auxiliary signature is `c(q_i) = (v̂_i^(1), v̂_i^(2), v̂_i^(3))`. Two
interventions are equivalent iff their three symbols match exactly, giving a discrete
deterministic quotient `C = χ(Q)` derived from an auxiliary channel that never
consults the return position. No Euclidean threshold on the latent space is used.

## 6. Return contexts U

The 16 return contexts are all pairs `(K_j, F_k)` for `j, k ∈ {1,2,3,4}`, each realized
as `[K_j, A, F_k, K_j]`. Source position remains `p_s = 1`; return position is
`p_r = 3`. The surface anchor token `A` always appears at the source position, so the
intervention's identity is never visible in the prompt tokens.

## 7. Return channel

For each `(q_i, u)`: patch the donor vector at `L_s, p_s`; read logits at `p_r`;
restrict to the 8 candidates; renormalize by softmax on that support. This defines
`p(Y_ret = v | q_i, u)`. When several `q_i` share a class `C = c`, their distributions
are averaged uniformly to get `p(Y_ret | C = c, U = u)`.

## 8. Metrics

With `q(Q)` and `q(U)` uniform:

- `K_eff = H(C|U) = H(C)`
- `B_reentry = I(C; Y_ret | U)`
- `R = B_reentry / K_eff` (or `NA` if `K_eff = 0`)
- `κ_R = H(C) / log2(|C*|)` when `|C*| > 1`
- `R_cap = B_reentry / log2(|C*|) = κ_R · R`

## 9. Negative temporal causal control

For every patch, logits are also read at `p_pre = 0`, which is causally *prior* to the
patched position `p_s = 1`. Define `B_pre = I(C; Y_pre | U)`. Under correct
autoregressive causality, `B_pre ≈ 0`. Frozen numeric criterion: `B_pre < 1e-8 bit`. A
violation invalidates the run.

## 10. Label-duplication control

The runner virtually rebuilds a battery `Q'` where every intervention is duplicated
twice without changing its signature or class mass, and checks that `H(C)` and `R`
stay unchanged to `1e-12`.

## 11. Anchor / no-patch control

For every context `U`, the unpatched prompt is also run. These results are descriptive
only and do not enter the computation of `C`, `K_eff`, or `R`.

## 12. Numerical precision

Model in `float32` after loading; `model.eval()`; no sampling; `torch.no_grad()`;
softmax computed in `float64` on CPU after extracting the candidate logits. The runner
publishes Python/PyTorch/Transformers versions, device, model config, token ids, all
`C` signatures, all return distributions, and final metrics.

## 13. Verdicts

- **TR-V1** — valid, point-identified run: causal controls pass, `|C*| > 1`, metrics
  computable.
- **TR-V2** — trivial source quotient: `|C*| = 1`, so `K_eff = 0` and `R = NA`. Not a
  code failure.
- **TR-V3** — invalid causal control (e.g. `B_pre ≥ 1e-8`). No `R` value is published.
- **TR-V4** — model/tokenizer/hook interface incompatible with the frozen protocol.

## 14. Claim status (v0.1)

Even a `TR-V1` verdict does not prove functional consciousness. It produces only a
first reproducible estimate of `(κ_R, R)` for one frozen intervention battery and one
frozen return boundary. Any generalization to other layers, tasks, models, or prompts
requires a new campaign or a pre-declared holdout.

**Result (v0.1).** TR-V1, 3-class top-1 source quotient, `R ≈ 1.91×10⁻³`.

---

## v0.2 — robustness follow-up (and an archived bug)

Protocol: `Protocole_TRANSFORMER_REENTRY_ROBUSTNESS_v0_2_FROZEN.md` (frozen after
observing v0.1; explicitly a robustness follow-up, not a pre-registered confirmation
of the v0.1 value).

**What was fixed.** Model, revision, source layer/position, the eight donor
interventions, and the tokenization rule from v0.1 were all kept exactly.

**What was varied.** Three nested source quotients `C_3 ⊂ C_6 ⊂ C_9` were built from
auxiliary probes using keys and fillers `K_j, F_k` with `j, k ∈ {1,2,3}` — `C_3` uses
the diagonal `(1,1),(2,2),(3,3)`; `C_6` adds `(1,2),(2,3),(3,1)`; `C_9` is the full
3×3 grid. Each probe kept the argmax over the 8 candidates as its signature symbol
(no distance threshold). The return channel used an entirely disjoint holdout:
`(K_j, F_k)` for `j, k ∈ {4,5}` — 4 contexts never used to build the quotient.

**Result.** Verdict `TR2-V1` with `ROB-STABLE-PARTITION` (`C_3 = C_6 = C_9` as a
partition of `Q`). On the disjoint return holdout:

```
B_Q  = 0.013459963734744118 bit
B_C3 = 0.0023373482510879065 bit
R_C3 = 0.00179962839232839
η_C3 = 0.17365189811429613
```

Relative to v0.1, `R` decreased by about 5.8%; the signal stayed the same order of
magnitude; the negative temporal control remained zero to machine rounding. This is a
genuine replication on a disjoint return holdout.

**The bug.** The v0.2 auxiliary probes had the form `[K_j, A, F_k]`, with `K_j` and
`F_k` both varied — but the logits used to build the quotient signature were read at
the *source* position `p_s = 1`, while `F_k` sits at position 2, which is causally
**after** position 1. In a causal autoregressive transformer, the readout at `p_s = 1`
cannot depend on `F_k` at all. This is visible directly in the data: for every
intervention and every key `K_j`, the argmax margins are identical across `k = 1,2,3`,
so varying the filler adds no new information and the observed `C_3 = C_6 = C_9`
stability is partly an artifact of this indexing choice rather than a genuine
robustness finding.

**Adjudicated status.** `TRANSFORMER_REENTRY_ROBUSTNESS_v0_2_ADJUDICATION.md` records
this explicitly as:

```
HOLDOUT-RETURN-STABLE + SOURCE-REFINEMENT-TEST-INFORMATIVE-ONLY-PARTIALLY
```

This is neither a code bug in the sense of an incorrect computation, nor a refutation
of `TR2-V1` — the holdout replication of the re-entry signal is real. But `C_3 = C_6 =
C_9` cannot be read as evidence that up to nine causally distinct auxiliary probes
converged on the same quotient: several of the "extra" probes were causal duplicates
because they varied a token the source readout could not see. This archived defect is
kept here deliberately, per the project's reproduction discipline, rather than
removed or silently corrected in place.

**Prescribed repair for v0.3:** keep the eight v0.1/v0.2 donors and `L_s = 2, p_s = 1`
exactly; fix the future filler in every auxiliary probe; vary only **keys in position
0** (causally prior to the readout); keep the v0.2 return holdout `(K_4,K_5) ×
(F_4,F_5)`; add new auxiliary keys disjoint from that holdout.

---

## v0.3 — source-quotient repeat with the filler bug fixed

Protocol: `Protocole_TRANSFORMER_REENTRY_SOURCE_QUOTIENT_v0_3_FROZEN.md` (frozen after
the v0.2 adjudication, before any v0.3 execution).

**Fix applied.** Auxiliary probes now have the fixed form `[X, A, F_1]`: the filler
`F_1` is fixed, and only the key `X` in position 0 — causally prior to the source
readout — varies. No variation of a future token counts as a new probe. The runner
verifies that the first 21 eligible tokens reproduce the v0.2 token groups exactly
before selecting new keys.

**New auxiliary keys.** After the 21 historical eligible tokens, four new eligible
tokens `E_0, E_1, E_2, E_3` are selected deterministically, distinct from the 8
candidates, `K_0…K_5`, `F_0…F_5`, and the anchor `A`.

**Nested quotients.** `C_3` uses keys `K_1, K_2, K_3`; `C_5` adds `E_0, E_1`; `C_7`
adds `E_2, E_3`. Each key is patched at layer 2 / position 1 and the candidate logits
are read back at position 1; the signature keeps the argmax. By construction `C_5`
must refine or equal `C_3`, and `C_7` must refine or equal `C_5`.

**Return holdout.** Identical to v0.2: `(K_4,F_4), (K_4,F_5), (K_5,F_4), (K_5,F_5)`.
The new keys `E_0…E_3` never appear in the return channel.

**Result.** Verdict `TR3-V1`, partition status `SQ-STABLE`: `C_3 = C_5 = C_7` remain
exactly the same partition `{0,5} | {1,3,4,6,7} | {2}`.

```
B_Q      = I(Q; Y_ret | U_H)        = 0.013459963734744118 bit
K_eff    = H(C)                     = 1.2987949406953985 bit
B_reentry= I(C; Y_ret | U_H)        = 0.0023373482510879065 bit
R                                    = 0.00179962839232839
η                                    = 0.17365189811429613
residual = I(Q; Y_ret | C, U_H)     = 0.011122615483656211 bit
```

Causal controls: `I(Q; Y_pre | U_H) ≈ -1.21×10⁻¹⁶ bit`; per-quotient
`I(C_m; Y_pre | U_H) ≈ -6.28×10⁻¹⁷ bit`. Both pass.

**Diagnosis.** The four new keys do change individual signatures (e.g. intervention 2
gets at least one different argmax under the new keys) — this time the stability test
is causally informative — but intervention 2 was already a singleton in `C_3`, and no
new key separates the five interventions of the large block or the two of the small
block. So `SQ-STABLE` is now a genuine finding, but `η ≈ 17.4%` remains low. The
bottleneck is diagnosed as the top-1/argmax compression of the readout, not the number
of auxiliary probes — motivating the ordinal rank-signature test in v0.4 (see
`reentry-holdout-confirmation.md`).

**Claim status.** On the frozen hook, donors, and holdout, the argmax-over-seven-
causally-accessible-contexts source quotient is stable under the enrichment tested,
but explains only about 17.4% of the return information distinguishable between the
eight interventions. This should not be read as the maximal or canonical causal
quotient.
