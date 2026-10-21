# 06 — Transformer measurement

Clean English edition of the transformer-measurement campaign: the first empirical
application of the K_eff/R re-entry normalization (defined analytically in campaign
`05-keff-r-normalization/`) to a real model.

This repository deliberately removes debugging/hotfix narrative. It keeps only the
canonical scientific stages, frozen protocols, final result statements, and the two
genuinely adjudicated implementation defects found along the way — those are
documented openly, not hidden, per this project's reproduction discipline (below).

## What was tested

- **Model:** `EleutherAI/pythia-70m-deduped`, revision
  `5ff092d907c3d8ba420d9fbef792426789eb2cd2` (`GPTNeoXForCausalLM`, 6 layers, hidden
  size 512, 8 attention heads).
- **Site:** residual stream, layer 2 (0-indexed), position 1.
- **Question:** does a discrete causal intervention injected at that site show up
  again — "re-enter" — at a later return position in the same forward pass, through a
  source quotient `C`, a return channel `Y_ret`, and a normalized signal
  `R = I(C; Y_ret | U) / H(C | U)`? And, separately: can that re-entry be closed into
  an explicit multi-round loop, and does any signal survive several rounds?

Every run also checks a frozen negative temporal control,
`I(C; Y_pre | U) < 1e-8 bit`, using a position causally *prior* to the patch — a
violation invalidates the run.

## The discovery-then-holdout arc (v0.1 → v0.5)

The six protocol versions are a sequence of adversarial refinements of the source
quotient `C`, not a chronological log meant to be read for its own sake. Full detail
in `docs/reentry-measurement-protocol.md` and `docs/reentry-holdout-confirmation.md`.

- **v0.1** — first quotient: argmax over 3 frozen auxiliary probes, 8 donor
  interventions built from `[K_0, V_i, F_0]`. Verdict `TR-V1`,
  3-class top-1 partition, `R ≈ 1.91×10⁻³`.
- **v0.2 (robustness follow-up)** — nested 3/6/9-probe quotients on a disjoint return
  holdout. Verdict `TR2-V1` / `ROB-STABLE-PARTITION`
  (`R = 0.00179962839232839`, `η = 0.17365189811429613`) — a genuine holdout
  replication. **But** the auxiliary probes varied a filler token positioned causally
  *after* the source readout, so the readout could not actually see it: part of the
  observed stability was structurally forced rather than tested. This is an
  **adjudicated defect, archived openly** (translated in full in
  `docs/reentry-measurement-protocol.md`, source file
  `TRANSFORMER_REENTRY_ROBUSTNESS_v0_2_ADJUDICATION.md`), status
  `HOLDOUT-RETURN-STABLE + SOURCE-REFINEMENT-TEST-INFORMATIVE-ONLY-PARTIALLY`.
- **v0.3** — same nested-quotient idea, filler bug fixed (only causally-prior keys
  vary now). Same 3-class partition survives (`SQ-STABLE`), this time genuinely
  tested: `R = 0.00179962839232839`, but only `η ≈ 17.4%` of the raw signal
  `I(Q; Y_ret)` is recovered — diagnosing the top-1/argmax compression, not the
  auxiliary battery, as the bottleneck.
- **v0.4** — replaces argmax with ordinal top-k rank signatures. Top-1 stays at
  `η ≈ 17.4%`; **top-2 recovers `η ≈ 86.4%`** (`R = 0.004230229753991946`), with the
  7-class partition `{0}|{1}|{2}|{3,7}|{4}|{5}|{6}`; top-4 saturates to the identity
  on `Q` (`η = 1`, a boundary check, not a substantive result).
- **v0.5 — candidate-fixed holdout** (the strongest reproducibility claim in this
  bundle): the top-2 rule from v0.4 is frozen, then run on 7 brand-new auxiliary keys
  and 4 brand-new return contexts, never seen by the candidate before.
  **Verdict `TOP2-HOLDOUT-PASS`** — the exact same 7-class partition is reproduced.
  Return-channel result on the new holdout: `K_eff = 2.75 bit`,
  `B_reentry = 0.01362349150610347 bit`, `R = 0.004953996911310352`,
  `η = 0.9022417474341858` (≈90.2%).

**Sub-part 4A closure (`CLOSE-CONDITIONAL`):** on Pythia-70M-deduped, layer 2 /
position 1, the frozen 8-donor battery, and the top-2 source quotient validated on an
independent holdout, a weak positive re-entry is reproduced on two disjoint return
batteries (`R ≈ 0.0042` on v0.4, `R ≈ 0.0050` on v0.5), capturing 86–90% of the raw
return information. This is **not** "R intrinsic to Pythia-70M" — it is conditional on
the hook, the donor battery, the candidate support, the top-2 rule, and the return
contexts.

## The closed-loop result (v0.6 execution error → v0.6.1)

Full detail in `docs/latent-closed-loop-result.md`.

Protocol v0.6 closes the loop explicitly, `C_t → Y_t^ret → C_{t+1}`, via a **declared**
transport `y_i ↦ q_i ↦ χ(q_i)` (not inferred after the fact), and iterates the induced
Markov kernel for `n ∈ {1,2,4,8,16}` rounds, reporting `R_n = I(C_0;C_n|U) / H(C_0)`.

- **v0.6 first attempt failed before any verdict.** An `enumerate()` counter meant to
  index 4 `(loop_key, loop_filler)` contexts was bound to the outer loop and reused
  across the inner loop, so only 2 of 4 contexts were ever populated; a `KeyError`
  surfaced downstream. This is archived as
  **`EXECUTION-ERROR / NO-VERDICT`** — explicitly *not* a refutation — translated in
  full in `docs/latent-closed-loop-result.md` (source file
  `TRANSFORMER_LATENT_CLOSED_LOOP_v0_6_EXECUTION_ERROR_ADJUDICATION.md`).
- **v0.6.1** fixes only the indexing (explicit enumeration of all 4 contexts) and adds
  the previously-missing explicit duplication-invariance control (L3), changing no
  scientific decision. **Verdict `LOOP-V1`.** `H(C_0) = 2.75 bit`. Multi-round profile:

  ```
  (R_1, R_2, R_4, R_8, R_16) =
  (0.00396355274282772,
   1.80125874286848e-05,
   5.0926939219345e-10,
   6.56648047639975e-18,
  -5.91445820305333e-18)
  ```

  i.e. `R_1 ≈ 3.96×10⁻³` decaying to `≈5×10⁻¹⁰` by round 4 and numerically zero by
  rounds 8–16. **No durable multi-round causal memory was detected in this system
  under this protocol.**

Only `transformer_latent_closed_loop_pythia70m_v0_6_1.py` (the fixed, reported
version) is included in `scripts/`; the pre-fix `..._v0_6.py` is superseded and
omitted — but the adjudication describing exactly what went wrong is kept.

## Final status

**`POINT4 — CLOSE-CONDITIONAL`** (`POINT4_TRANSFORMER_AND_LATENT_LOOP_CLOSURE_v0_1.md`,
translated in full in `docs/latent-closed-loop-result.md`), combining:

- **4A** — re-entry quotient discovered (v0.4), reproduced exactly on an independent
  candidate-fixed holdout (v0.5): `K_eff = 2.75 bit`, `B_reentry ≈ 0.0136 bit`,
  `R ≈ 0.00495`, `η ≈ 90.2%`.
- **4B** — closed loop: `LOOP-V1`, `R_1 ≈ 3.96×10⁻³` decaying to numerical zero by
  round 4.

**Scope.** This is an exact computation on one frozen model / hook / donor battery /
holdout — the model is deterministic and every kernel here is computed exactly, no
sampling noise — but nothing here generalizes to other layers, models, or tasks
without a new campaign. `POINT4A_*`, `POINT4_*`, and `POINT4_MANUSCRIPT_PATCH_*` (see
`docs/manuscript-patch.md`) are the only statements in this folder meant to be quoted
as a claim; do not quote an intermediate version's number as if it were the campaign's
final result.

## Reproduction discipline

This campaign follows the same discipline as the rest of the repository (see the root
`README.md` for the full statement):

1. A **protocol** is written and frozen (hashed conceptually, by being committed as
   a `_FROZEN` file) *before* any candidate code is written or run.
2. A **generator + private oracle** pattern is not separately instantiated here in the
   RMT sense used by other campaigns — instead, the deterministic token-selection rule
   and frozen donor battery play the equivalent role of fixtures, and each script
   prints its frozen protocol parameters via `--plan-only` without touching the model.
3. Where this campaign claims **"candidate-fixed holdout"** (v0.5, and the source-side
   part of v0.2/v0.3), a second, independent set of auxiliary keys and/or return
   contexts is generated only *after* the candidate rule (here, the top-2 signature
   rule) is frozen, and the unmodified candidate is re-run against it. This is the
   strongest reproducibility claim this bundle makes.
4. Verdicts are never silently converted into a single point value when the protocol
   doesn't license one: `TR-V2` (trivial quotient) yields `R = NA`, not `R = 0`;
   `TR-V3` / `TR-V4` / execution errors publish no `R` at all. The two genuine
   implementation defects found in this campaign (v0.2's future-filler indexing issue,
   v0.6's loop-context enumeration bug) are kept as adjudication notes rather than
   quietly patched out of the history.

## How to verify / rerun

Each script accepts `--plan-only` to print its frozen protocol configuration without
loading the model:

```bash
python3 scripts/transformer_reentry_pythia70m_v0_1.py --plan-only
python3 scripts/transformer_reentry_top2_holdout_pythia70m_v0_5.py --plan-only
python3 scripts/transformer_latent_closed_loop_pythia70m_v0_6_1.py --plan-only
```

To reproduce the actual measurements, run any script without `--plan-only`. This
requires `torch`, `transformers`, and network access to `huggingface.co` to download
`EleutherAI/pythia-70m-deduped` at the pinned revision above. Recommended order to
follow the discovery arc:

```bash
python3 scripts/transformer_reentry_pythia70m_v0_1.py                         # v0.1
python3 scripts/transformer_reentry_robustness_pythia70m_v0_2.py              # v0.2 (archived bug)
python3 scripts/transformer_reentry_source_quotient_pythia70m_v0_3.py         # v0.3 (fixed)
python3 scripts/transformer_reentry_rank_signature_pythia70m_v0_4.py          # v0.4 (top-2 discovery)
python3 scripts/transformer_reentry_top2_holdout_pythia70m_v0_5.py            # v0.5 (holdout confirmation)
python3 scripts/transformer_latent_closed_loop_pythia70m_v0_6_1.py            # v0.6.1 (closed loop, fixed)
```

`results/STATUS.json` records the final numbers and both archived defects in
machine-readable form.

## Contents

```
README.md
docs/
  reentry-measurement-protocol.md       — v0.1 protocol, v0.2 robustness run + archived bug, v0.3 fix
  reentry-holdout-confirmation.md       — v0.4 rank-signature discovery, v0.5 candidate-fixed holdout, 4A closure
  latent-closed-loop-result.md          — v0.6 protocol, v0.6 execution-error adjudication, v0.6.1 result, overall closure
  manuscript-patch.md                   — exact proposed manuscript diff (K_eff/R definition, claim-status vocabulary)
scripts/
  transformer_reentry_pythia70m_v0_1.py
  transformer_reentry_robustness_pythia70m_v0_2.py
  transformer_reentry_source_quotient_pythia70m_v0_3.py
  transformer_reentry_rank_signature_pythia70m_v0_4.py
  transformer_reentry_top2_holdout_pythia70m_v0_5.py
  transformer_latent_closed_loop_pythia70m_v0_6_1.py    — fixed version; pre-fix v0_6.py is superseded and omitted
results/
  STATUS.json                           — final numbers, verdicts, and archived-defect record
```

See the root `README.md` for the full protocol → manuscript-section table covering all
four work streams in this repository.
