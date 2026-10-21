# Gamma-Cert v0.4 — additive scale certificates for Γ

Clean English edition of the Γ scale-certification package, folder
`07-gamma-scaling-and-certification/gamma-cert-v0.41/`. Both experiments here are fully
**executed**, self-contained, exact/finite or exactly-calibrated computations — neither depends
on a real target transformer. The source note (`GAMMA_CERT_V04_NOTE.md`) was written in French;
this edition's `docs/` are a fresh English restatement of its mathematics and results, not a
literal translation of the frozen file (see the root repository's note on why hashed/frozen
protocol documents are restated in English rather than translated in place).

## Reproduction discipline

This package fixes the principal limitation of a prior v0.3 hierarchy: a max-only witness of
fixed block order cannot certify a non-vanishing Γ as system size grows. The fix is proved as a
theorem (fractional packing, no double counting — see `docs/additive-packing-certificate.md`),
then checked by an exact dual-MILP-vs-exhaustive-enumeration assertion on every tested system,
and separately calibrated against brute-force exact conductance wherever brute force remains
feasible (`n ≤ 20` for the sparse benchmark; all `n` for the complete-graph benchmark, which has
a closed form). Nothing here changes the definition of Γ or edits the manuscript — every result
is a **certified bound** on Γ, reported as a lower bound, an interval, or (where both sides match
exactly) an exact value, never presented as more precise than what was actually proved.

## Main results

### Additive fractional-packing certificate (exact theorem + executed audit)

A collection of conditional witnesses `d_q(S→T∣C) = I_q(X_S;X'_T∣X_C)` can be summed
fractionally — without double counting, by a fractional Shearer/Madiman–Tetali argument — into a
certified lower bound on the directional cut information `J_q(A→B)`, and the resulting global
lower-bound optimization has an exact dual MILP. Executed on four finite systems: singleton
witnesses alone certify as little as `0%` of the exact Γ on a purely synergistic system
(`xor_neighbors_6`), while block-order-`(3,2,1)` witnesses recover the exact value on that same
system; a system with no exploitable low-order structure (`random_boolean_6`) stays loosely
certified (`25%`) even at that block order. See `docs/additive-packing-certificate.md`.

### One-sided spectral (Cheeger) certificate, and an exact copy-expander benchmark to scale n = 1024

A decoding-graph corollary of the packing theorem gives `Γ_q ≥ δ_min·λ₂/4` — a proven additive
causal sub-model, not the older, weaker "spectral approximates Γ" claim. On a copy-expander
benchmark where Γ is *exactly* the conductance of a `d`-regular graph, this gives certified
two-sided Γ intervals up to `n = 1024` roles without ever enumerating a cut, e.g.
`Γ ∈ [0.173, 0.267]` at `n = 1024`, `d = 8`. Exact calibration on small systems (`n ≤ 20`) shows
the Fiedler-sweep upper bound recovers the exact optimum, and the complete-graph benchmark gives
gap-zero certification analytically up to `n = 1024`. See `docs/copy-expander-scale-benchmark.md`.

## Layout

```
README.md                                      this file
docs/
  additive-packing-certificate.md              witness/packing theorem, dual MILP, decoding-graph + Cheeger bound, packing results
  copy-expander-scale-benchmark.md             exact-conductance benchmark, calibration, n=64..1024 scaling
scripts/
  gamma_fractional_packing_autonomous.py       fractional-packing theorem audit (self-contained)
  gamma_copy_expander_scale.py                 copy-expander conductance benchmark (self-contained)
results/
  fractional_packing_results.json              exact packing results per system/block-config
  copy_expander_scale_results.json             exact/certified Gamma per n, complete-graph and random-8-regular
```

## What was intentionally left out

- `GAMMA_CERT_V04_NOTE.md` (the French source note) is not redistributed verbatim; its content is
  restated fresh in English across the two `docs/*.md` files above, per this repository's rule
  against translating frozen/hashed protocol text in place.
- `LANGUAGE_AUDIT.json`, `SHA256SUMS.json` — process bookkeeping specific to the source archive.

## How to verify

```bash
python3 scripts/gamma_fractional_packing_autonomous.py
python3 scripts/gamma_copy_expander_scale.py
```

Both are self-contained (`numpy`, `scipy`, `networkx`, `torch`). Regenerates the two results
files above; values should match exactly (the packing script asserts
`dual_MILP_matches_exhaustive` and `lmilp ≤ exact + 1e-9` internally; the expander script asserts
`lower ≤ exact ≤ upper` wherever exact conductance is computed).

## Relation to the rest of this folder

This package's certified bounds are a different, more scalable route to the same goal as
`gamma-cert-target-transformer/`'s decoder-graph-only lower bound and cross-entropy upper bound —
see that folder's README for how the two constructions compare. `gamma-cert-causal-quotient/`
tests the quotient-sufficiency assumption both certificates depend on.
