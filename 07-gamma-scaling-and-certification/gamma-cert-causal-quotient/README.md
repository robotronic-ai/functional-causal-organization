# Gamma-Cert causal quotient — redundant-quotient stress test

Clean English edition of the causal minimal-state quotient stress test, folder
`07-gamma-scaling-and-certification/gamma-cert-causal-quotient/`. This is a fully **executed**,
self-contained exact finite experiment (not a real-transformer run, not blocked on any external
artifact) that tests whether a causal minimal-state quotient can remove purely representational
redundancy without hiding causal distinctions that become real under a richer intervention
algebra.

## Reproduction discipline

This package follows the same discipline as the rest of the repository: the intervention algebra
(coherent vs. raw physical) is declared before the quotient is evaluated, the quotient is never
selected to make Γ smaller, and every result — including every point at which the quotient
*fails* the raw-algebra sufficiency test — is reported, not silently dropped. There is no
V1/V3/NA verdict language here because this package is a diagnostic stress test of the quotient
*gate* used elsewhere (`gamma-cert-target-transformer/`), not itself a Γ point-estimate or
certified-interval claim about a target model.

## Main result

A four-role recurrent ring, physically implemented with three coherent replicas per role plus a
tunable null-space perturbation, gives an exact macrostate `Γ_macro = 0.5` at every tested
perturbation scale (`0.0` to `4.0`), while the raw physical `Γ` drifts from `0.167` to `0.485` as
the perturbation grows and the raw-algebra quotient sufficiency test starts failing (0 of 16
macro classes ambiguous at scale `0.0`–`0.5`; all 16 ambiguous by scale `2.0`–`4.0`). The
coherent functional transition never changes. See `docs/redundant-quotient-stress-test.md` for
the full table, the exact counterexamples, and the five-step safe certification procedure this
motivates.

## Layout

```
README.md                                this file
docs/
  redundant-quotient-stress-test.md      full construction, result table, interpretation, safe procedure
scripts/
  gamma_redundant_quotient.py            self-contained exact experiment (final, only version)
results/
  nullspace_stress_results.json          exact per-scale results, including microstate counterexamples
```

## Relation to the rest of this folder

This stress test motivates and justifies the "Non-negotiable quotient gate" step of
`gamma-cert-target-transformer/`'s certification protocol, and the same causal-quotient
discipline underlies the block structure used in `gamma-quantum-campaign/`.

## How to verify

```bash
python3 scripts/gamma_redundant_quotient.py
```

Fully self-contained; requires only `numpy`, `torch`. No external model or checkpoint needed.
Regenerates `nullspace_stress_results.json` (values above should match exactly, up to floating
point, since every quantity is computed by exact finite enumeration over the raw `4096`-state and
macro `16`-state alphabets).
