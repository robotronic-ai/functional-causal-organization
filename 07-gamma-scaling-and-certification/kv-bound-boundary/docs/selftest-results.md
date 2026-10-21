# Self-test of the boundary decomposition theorem — synthetic toy model, fully executed

## What this is (and is not)

`scripts/kv_bound_boundary_selftest.py` is a **pure PyTorch, no-external-model** self-test of the
decomposition theorem `B_q ≤ H(Y_t∣C_t,U_t) + C_bypass,q` (see
`boundary-decomposition-theorem.md`). It builds a tiny single-layer causal-attention module
(`TinyAttention`: embedding, Q/K/V/O projections, LM head, 8-dimensional model, 16-token
vocabulary) with a KV cache, and checks the inequality — and the qualitative distinctions between
intervention modes — cell by cell, exactly, over 8 fixed histories. **This is a verification of
the theorem on a synthetic system, not an application of the theorem's falsification protocol to
a real target transformer.** That real-model application is specified but not executed (see
`boundary-decomposition-theorem.md` §6).

## The four intervention modes

| Mode | Description | Expected bypass |
|---|---|---|
| `post_cycle` | source intervention applied *after* K/V writes; only path is `S→Y→N` | zero |
| `prewrite` | source intervention applied *before* Q/K/V writes; changes both `Y` and the persistent representation | positive |
| `raw_kv` | `Y` held fixed; direct K/V side-channel surgery | positive (equals `B` exactly, since `H(Y)=0`) |
| `positive_latent` | `Y` held fixed; explicit latent side channel added at the next step | positive (equals `B` exactly) |

## Executed results (mean over 8 histories, exact finite-alphabet entropies)

| Mode | `B` (bits) | `H(Y∣C,U)` (bits) | `C_bypass` (bits) | slack (`H(Y)+C_bypass-B`) |
|---|---:|---:|---:|---:|
| `post_cycle` | 0.4917 | 0.4917 | 0.0000 | 0.0000 |
| `prewrite` | 1.7500 | 0.5931 | 1.1569 | 0.0000 |
| `raw_kv` | 1.6875 | 0.0000 | 1.6875 | 0.0000 |
| `positive_latent` | 2.0000 | 0.0000 | 2.0000 | 0.0000 |

(Source: `results/summary.json`; per-history cells in `results/cells.jsonl`, 32 rows = 8
histories × 4 modes.)

## What the self-test confirms

- The theorem's inequality holds in **every** cell with essentially zero slack
  (`slack_bits ≥ -1e-12` asserted for all 32 cells) — on this synthetic system the bound is tight,
  not merely valid.
- **`post_cycle` bypass is exactly zero** in every history — confirming that when the source
  intervention only reaches the next state through the emitted token, the bound collapses to
  `B_q ≤ H(Y_t∣C_t,U_t)`, i.e. the "15–17 bit" style corollary applies.
- **`prewrite`, `raw_kv`, and `positive_latent` bypass is positive** in every history —
  confirming the theorem correctly detects a persistent-channel or latent side-channel that
  bypasses the token.
- A separate numerical-equivalence check (`equivalence_max_l2 < 1e-10`) confirms the cached
  incremental-decoding path and a full from-scratch recomputation agree to floating-point
  precision, so the KV-cache bookkeeping used to compute `N` is itself correct.

`results/summary.json` records `"PASS": true` and the interpretation: *"The exact finite-battery
inequality B ≤ H(Y∣carry) + C_bypass holds in every cell. Post-cycle interventions have zero
conditional bypass; pre-write, raw-KV and explicit latent channels produce positive bypass."*

## Honest scope statement

This confirms the **theorem** and the **qualitative discrimination power** of the bypass
diagnostic on a toy system built specifically to exercise all four modes. It does **not**
establish the numeric value of `C_bypass^terminal` or `C_bypass^prewrite(ℓ)` on any real target
transformer, and it does not by itself confirm or refute the manuscript's "17 bits" empirical
claim for a real model — that is the explicit, unexecuted next step described in
`boundary-decomposition-theorem.md` §6.
