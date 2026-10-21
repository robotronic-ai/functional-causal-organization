# 10 — Continuous predictive geometry

Clean English edition of the Continuous Predictive Geometry Protocol: a theoretical
extension of the project's exact finite predictive-state theory (see campaign
`12-readout-relative-minimal-predictive/`) to continuous or high-dimensional systems,
where an exact future-law quotient can be too fine — or computationally unusable — to
serve as a working object.

This is a **mathematical / protocol-design campaign**. It proves a family of theorems
about a discounted predictive pseudometric and verifies the resulting numeric claims
exactly on a small closed-form toy family (discounted Bernoulli processes). It is not
an empirical measurement on a trained model or any other real system — see
`docs/rmt-bridge-status-and-open-questions.md` for the explicit "next target" framing
in the source material itself.

## What question this answers

Exact predictive-state theory (the Koopman-closure construction used elsewhere in this
project) identifies states that are *exactly* predictively equivalent under a declared
readout class. On continuous or high-dimensional systems this exact quotient is often
too fine to be useful, and there is no canonical way to relax it to an approximate
"epsilon-quotient" — thresholding a distance at `epsilon` is not in general transitive.

This protocol instead builds a genuine **pseudometric**,

```
d_gamma(x, y) = (1 - gamma) * sum_{m>=1} gamma^(m-1) * d_m(x, y)
```

where `d_m(x, y) = sup_pi D_m(P_x^pi, P_y^pi)` is the worst-case discrepancy between
length-`m` future observation laws from `x` and `y` under a declared policy class, and
`0 < gamma < 1` is a discount factor. This preserves *exact* predictive equivalence at
zero distance while giving every finite horizon a continuous, non-discrete notion of
"nearby" predictive behavior — see `docs/why-discounting-is-needed.md` for why the
undiscounted infinite-horizon total-variation metric fails to do this.

## Main results

All results in this campaign are **exact mathematical theorems plus exact numerical
verification on a closed-form toy family** — nothing here is a statistical estimate,
and nothing here has been applied to a trained model.

1. **`d_gamma` is a pseudometric with a universal truncation certificate**
   (`d_gamma <= d_{gamma,H} + gamma^H`, no mixing/contraction assumption required),
   zero distance recovers exact predictive equivalence, and — under compactness and
   continuity of every `d_m` — the predictive quotient is totally bounded, so finite
   `epsilon`-covers exist for every `epsilon > 0`. Full statement and proof sketch:
   `docs/theorem-and-proof-sketch.md`.
2. **Epsilon-thresholding a predictive distance is not transitive in general** — proved
   by an exact three-point counterexample (`p=0, q=0.4, r=0.8`, `epsilon=0.5`):
   `p` and `q` are `epsilon`-close, `q` and `r` are `epsilon`-close, but `p` and `r` are
   not. Status `PASS`. See `docs/theorem-and-proof-sketch.md` and
   `results/epsilon_nontransitivity_results.json`.
3. **Undiscounted infinite-path total variation is discrete on continuous families**:
   for i.i.d. Bernoulli(`p`) vs. Bernoulli(`q`) with `p != q`, infinite-horizon TV is
   exactly `1` regardless of `|p-q|`, by mutual singularity of the two infinite product
   laws (strong law of large numbers). The discounted metric stays continuous in `p`
   under the same family. See `docs/why-discounting-is-needed.md`.
4. **Exact numerical check on the discounted-Bernoulli family**: an 11-point grid
   `p, q in {0, 0.1, ..., 1.0}`, `gamma = 0.8`. Zero monotonicity failures, zero
   truncation-bound failures, zero approximate-triangle-inequality failures. Status
   `PASS`. Full table: `results/discounted_bernoulli_results.json`.
5. A separate **dynamic-rollout error recurrence** (`e_{t+1} <= L*e_t + delta`) is
   proved for an abstract transition model tracked against the true dynamics under a
   declared Lipschitz/contraction coefficient `L` and one-step defect `delta`, cleanly
   separating state-quantization error from transition-model error. This is a proved
   theorem, not yet numerically exercised in this bundle. See
   `docs/theorem-and-proof-sketch.md`.

## What this campaign explicitly does not claim

- It does not apply the discounted predictive metric to any trained model, RNN, or
  other real controlled system. The source material's own status note (translated in
  `docs/rmt-bridge-status-and-open-questions.md`) names this as the explicit "next
  target," not a completed step.
- It does not conflate behavioral covering complexity (`N_gamma(epsilon)`, how many
  representatives are needed to epsilon-cover the reachable set) with linear predictive
  rank (effective Hankel rank under a declared norm/tolerance). The protocol requires
  these to be reported separately — see `docs/complexity-notions-and-guardrails.md`.
- It does not treat predictive decodability as evidence of realized causal use; that is
  named as a separate, later scientific layer (Phase 6 of the protocol).

## Reproduction discipline

This campaign follows the same discipline as the rest of the repository (see the root
`README.md` for the full statement):

1. A **protocol** is written and frozen (hashed conceptually, by being committed as a
   `_FROZEN` file) *before* any candidate code is written or run.
2. A **generator + private oracle** pattern is not separately instantiated here in the
   RMT sense used by other campaigns — instead, the discounted-Bernoulli closed-form
   family and the explicit three-point counterexample play the equivalent role of
   fixtures, and each verifier is a small, self-contained, deterministic script.
3. Where this campaign claims **"candidate-fixed holdout"**, the general protocol
   (`docs/protocol-and-held-out-transport.md`, Phase 4) requires evaluating any learned
   predictive representation on at least one strict holdout — a later horizon, an
   unseen control, an unseen context, or an unseen downstream readout composition —
   before accepting transport. No learned representation is built or held out in this
   bundle; the requirement is stated as a protocol gate for future use of this theory,
   not exercised here.
4. Verdicts are never silently converted into a single point value when the protocol
   doesn't license one. This bundle contains no `NA`/no-verdict cases: every check that
   was run is exact and reports a definite `PASS`. The campaign is explicit, however,
   that a discounted predictive distance without a stated truncation horizon `H` and
   tail term `gamma^H` is not a licensed report — see
   `docs/complexity-notions-and-guardrails.md`.

## How to verify / rerun

```bash
python3 scripts/verify_discounted_bernoulli.py
python3 scripts/verify_epsilon_nontransitivity.py
python3 scripts/verify_all.py            # runs both and aggregates PASS/FAIL
```

All three scripts use only the Python standard library and print their JSON result to
stdout; `verify_all.py` also re-invokes the first two as subprocesses and aggregates
their status. `results/*.json` records the exact output already produced by this run.

## Contents

```
README.md
docs/
  theorem-and-proof-sketch.md          — discounted pseudometric, truncation certificate,
                                          compactness/covers, epsilon-nontransitivity,
                                          dynamic rollout recurrence, and their proofs
  why-discounting-is-needed.md         — Bernoulli infinite-path TV singularity example
  protocol-and-held-out-transport.md   — six-phase experimental protocol + held-out
                                          transport certification requirements
  complexity-notions-and-guardrails.md — covering complexity vs. linear (Hankel) rank,
                                          and the methodology guardrails against
                                          collapsing them or skipping the tail term
  rmt-bridge-status-and-open-questions.md — how to apply this to a frozen model
                                          representation (not yet done), current status,
                                          and open theoretical questions
scripts/
  verify_discounted_bernoulli.py       — exact numerical check on the Bernoulli family
  verify_epsilon_nontransitivity.py    — the three-point nontransitivity counterexample
  verify_all.py                        — runs both verifiers and aggregates PASS/FAIL
results/
  discounted_bernoulli_results.json    — full output of verify_discounted_bernoulli.py
  epsilon_nontransitivity_results.json — full output of verify_epsilon_nontransitivity.py
  verification_summary.json            — combined output as produced by verify_all.py
  STATUS.json                          — machine-readable summary of verdicts
```
