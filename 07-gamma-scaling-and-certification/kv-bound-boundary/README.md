# KV-Bound boundary v0.3 — the temporal boundary and a falsifiable bound-with-bypass theorem

Clean English edition of the KV-cache temporal-boundary package, folder
`07-gamma-scaling-and-certification/kv-bound-boundary/`. The source note
(`temporal_boundary_note.md`) was written in French; `docs/boundary-decomposition-theorem.md` is
a fresh English restatement of its theorem, proof, and consequences, not a literal translation of
the frozen file.

## Reproduction discipline

The theorem here is proved analytically (chain rule + a standard subadditivity bound; see
`docs/boundary-decomposition-theorem.md`), then verified exactly on a synthetic toy-attention
self-test built specifically to exercise every qualitatively distinct intervention mode the
theorem needs to discriminate (`docs/selftest-results.md`). This package is explicit about the
boundary between what has been proved, what has been verified on a toy system, and what remains
an unexecuted, fully specified protocol for the real model (§6 of
`docs/boundary-decomposition-theorem.md`) — the "17 bits" empirical claim is **not** confirmed or
refuted here; a falsifiable experimental criterion for testing it on a real model is defined but
not run.

## Main results

### A precise definition that resolves a notation ambiguity

`P_{t+1}` was ambiguous between "the part simply carried from the past" and "the new history
after `Y_t`." This package defines `C_t` — the carried persistence, invariant under every
admissible intervention on the source — separately from `Y_t`, closing a route by which one could
accidentally condition on a descendant of the source and erase the channel being measured.

### A proved decomposition theorem

\[
B_q \le H_q(Y_t\mid C_t,U_t) + C_{\mathrm{bypass},q}
\]

via the chain rule and `I(Z;Y∣C,U) ≤ H(Y∣C,U)`, requiring no assumption that the token is the
only channel. See `docs/boundary-decomposition-theorem.md`.

### An experimentally falsifiable "15–17 bit" corollary

The bound collapses to `B_q ≤ H_q(Y_t∣C_t,U_t) ≤ log₂|V|` **if and only if** `C_bypass,q = 0` is
measured to hold — turning a previously assumed property of the abstraction into a testable
claim, and fixing where in the cycle the source must be measured (a cycle-boundary state, after
persistent writes close) for the bound to apply.

### Toy self-test: theorem confirmed exactly, zero slack, on all four intervention modes

Executed on a synthetic single-layer attention module: the inequality holds with essentially zero
slack in all 32 tested cells (8 histories × 4 modes); `post_cycle` bypass is exactly zero in every
case, while `prewrite`, `raw_kv`, and `positive_latent` bypass is positive in every case — exactly
the qualitative discrimination the theorem is meant to provide. See `docs/selftest-results.md`.

### What has NOT been executed

The real-model falsification protocol (measure `C_bypass^terminal` and `C_bypass^prewrite(ℓ)` on
the actual target transformer) is fully specified in `docs/boundary-decomposition-theorem.md` §6
but has not been run. No claim about the real model's "17 bits" bound is made or implied by the
results in this package.

## Layout

```
README.md                                  this file
docs/
  boundary-decomposition-theorem.md        C_t/Y_t definition, estimand, proved theorem, unexecuted real-model protocol
  selftest-results.md                      executed toy-model verification, exact per-mode results table
scripts/
  kv_bound_boundary_selftest.py            self-contained synthetic self-test (final, only version)
results/
  summary.json                             mean per-mode results, PASS status (verbatim)
  cells.jsonl                              all 32 per-history-per-mode cells (verbatim)
```

## How to verify

```bash
python3 scripts/kv_bound_boundary_selftest.py
```

Fully self-contained; requires only `torch`. Regenerates `cells.jsonl` and `summary.json`
byte-for-byte (deterministic seeds).

## Relation to the rest of this folder

This package concerns a different bound (the source→next-state information bypassing the
emitted token) from the Γ scaling/certification work in the other four sub-bundles, but shares
the same causal-quotient discipline: a channel is only "captured" by a lower-dimensional
representation (here, the token `Y_t`) once that has been demonstrated, not assumed — the same
principle `gamma-cert-causal-quotient/` demonstrates for macro-state quotients.
