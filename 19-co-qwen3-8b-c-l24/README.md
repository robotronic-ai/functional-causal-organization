# Qwen3-8B — C_O components on the declared finite contract C_L24

This bundle reports finite, contract-relative measurements of the four C_O components Γ (integration), Δ
(differentiation), R (re-entry) and Λ₀ (typed content → role addressing) on **Qwen3-8B** (bf16), for a single declared
causal interface, the **C_L24 contract**: layer 24 →
layer 25, world **u26**, twelve L24 donor sites with frozen donors, twelve frozen L25 readers, and the ten Q1 gauge
replicas (identity plus nine exact recodings).

All statements are relative to this contract. R is the local L24 re-entry of the contract C_L24-R, not the R of the terminal state. Nothing here claims certification over all layers, horizons,
intervention alphabets or causal individuals of the model.

## Status

| Component | Status on C_L24, u26 | Value |
|---|---|---|
| Γ | NONZERO_CERTIFIED (independent raw-witness verification) | measured minimum 0.2890625 over 1023 cuts × 10 gauges; certified lower bound 1.5673619e-4 |
| Δ_H, Δ_S | NONZERO_CERTIFIED (independent exact-game verification) | 0.4 (per-reader convention); ≥ 6.2938595e-4 under admissible refinements; 0.742 / 0.650 under joint closure |
| R, greedy transition | measured exactly zero | no donor configuration changes the emitted token (4096/4096 masks) |
| R local L24 (C_L24-R), sampled transition (T = 1, full vocabulary) | NONZERO_CERTIFIED (independent recomputation from the raw export and the gauge cubes) on role mlp0 | R_state ≥ 7.385298596317709e-8 (B_mlp0 ≥ 8.1238284559494797e-7 bit); all 11 roles have a positive return in all 10 replicas |
| Λ₀ (C_L24-Λ), typed content → role lower relation | NONZERO_CERTIFIED at the primary threshold > δ (independent audit) | 68 of 132 cells resolved positive at t, 11 of 11 contents; 39 cells and 8 of 11 contents at 2δ (secondary) |

The profile (Γ, Δ, R, Λ₀) on this single contract therefore has four components certified non-zero. This is a
component-wise statement on a finite contract; it is not the functional closure of the framework (§17.7 of the reference
document, an exhaustive collision search over a finite universe), which has not been measured.

Under another, earlier contract (L23/L24 anchor roles on the calibration battery), the restricted Λ₀ verdict
NONZERO_CERTIFIED (threshold rule of its own instrument, 2 δ_w) is supported by primary L1 evidence that is included here and replayed exactly from the raw role writes
(`historical_lambda0/`, `results/lambda0_restricted_replay.json`).

An exploratory, non-certified calibration pilot (T346) is described at the end of [RESULTS.md](RESULTS.md) and in `exploratory_t346/`; it does not change any status above.

Details, scopes and limitations: [RESULTS.md](RESULTS.md). Contract definition: [CONTRACT.md](CONTRACT.md).
Methods: [METHODS.md](METHODS.md). How to rerun every step: [REPRODUCE.md](REPRODUCE.md).

## Layout

```
README.md, CONTRACT.md, METHODS.md, RESULTS.md, REPRODUCE.md, DEPENDENCIES.md
VERIFY_NPZ_SHA.py           checks the raw NPZ arrays on disk against the SHA-256 recorded in the results
MANIFEST_SHA256.txt         SHA-256 of every file in this bundle
scripts/                    acquisition (GPU) and analysis (CPU) scripts
results/                    result summaries (JSON), world u26
results/replication_u25/    optional replication on world u25
independent_verification/   verifiers and certificates written and run independently of the measurement chain, included byte-identical
                            (gamma/, delta/, r/, lambda/ with the Λ₀ certificate T339)
historical_lambda0/         primary L1 evidence of the restricted Λ₀ verdict (L23/L24 anchors) and its producing instrument
dependencies/               reference copies of the shared modules imported by the GPU scripts
exploratory_t346/           exploratory calibration pilot (not part of the certified C_L24 statements): attenuation of mlp0@L24, PCI_T branch results, FCA-E1 runner
```

Raw arrays (NPZ, and the per-replica .pt role writes of the L1 runs) are not included. `VERIFY_NPZ_SHA.py` checks that a
local copy matches the digests recorded here.

## Provenance of the result summaries

Several JSON summaries in `results/` were written by functionally identical earlier versions of the scripts in
`scripts/`; their keys and descriptive strings have been normalized to the names used by the current scripts. Every
numeric value and every SHA-256 digest is unchanged. NPZ array names are kept as acquired, so regenerated tables
reproduce the recorded digests. In `r_token_channel_u26.json`, fields of a float32 screen of the sampled channel were
removed; the exact float64 sampled channel is computed by `scripts/r_sampled_return_gpu.py`. Equivalence of the CPU
scripts with their earlier versions was checked on synthetic inputs (identical numbers and identical table digests).
