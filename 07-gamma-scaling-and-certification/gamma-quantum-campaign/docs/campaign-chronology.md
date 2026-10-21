# Campaign chronology (v1–v15) and final Γ_dir result

This campaign ran against a real target model — internally called "Quantum"
(`QuantumModelForCausalLM`, 36 layers, custom paged/quantized KV cache; see
`results/ARCHITECTURE_CONFIG_EXCERPT.json`) — not a synthetic system. It is a genuinely executed
empirical measurement, in contrast to the `gamma-cert-target-transformer/` sub-bundle in this
same folder, whose target run against a *different*, recurrent-memory transformer is blocked on
missing model artifacts (see that folder's `docs/execution-status.md`). The two must not be
confused: this campaign reports real point-estimate `Γ_dir` values measured on a real model
under a *candidate* (not yet statistically certified with confidence intervals) normalization;
`gamma-cert-target-transformer/` is a statistically certified `V1`/`V3`/`NA` machinery that has
not yet been run on any real model.

Numbers below are copied verbatim from `results/CAMPAIGN_STAGE_RESULTS.json`, the curated
machine-readable summary of every stage.

## v1–v3: layerwise persistent-state probe — self-separability is not essentiality (rejected path)

The initial layerwise K/V intervention battery exposed a forward-triangular causal structure,
but a self-separability readout could not serve as a proxy for causal essentiality: v2's
quotient validation failed at layer 34 (readout global margin `0.040`), and a fresh v3
multihorizon validation, built to fix this, instead exposed a moving terminal-layer failure at
layer 33 that turned out to be a token-clamp boundary artifact rather than a real essentiality
result. **Neither v2 nor v3 is kept as a script here** — the scripts were one-off diagnostics
superseded by the token-mediated re-entry diagnostic (v4) that explains the artifact; their
numeric findings are preserved in `results/CAMPAIGN_STAGE_RESULTS.json` under
`v2_quotient_validation` / `v3_multihorizon_quotient_validation`.

## v4: token-mediated re-entry — layer 35 is causal only through the emitted token

The re-entry diagnostic distinguished "endogenous" (natural autoregressive) from "clamped"
(token forced to a fixed value) re-entry. Layer 35's one-step effect is exactly zero, its
endogenous-reentry effect is `≈0.109`, and its clamped-reentry effect is exactly zero — i.e. it
is causal *exclusively* through the emitted-token path. Conclusion: token clamping cannot define
global essentiality for Γ, and layer 35 must not be dropped by a token-clamped essentiality gate.
This finding motivates using endogenous re-entry (not clamped re-entry) in every later block
battery. The v4 script (`quantum_gamma_reentry_pilot_v4.py`) is not kept here — it is not a code
dependency of the final battery — but its finding is load-bearing for how v10–v12 are built, and
is preserved verbatim above and in `results/CAMPAIGN_STAGE_RESULTS.json`.

## v5–v6: rejected response quotients

- **v5** (endogenous-reentry gate): layer 0 factorized robustly, but sparse token-mediated
  effects and a degenerate binary-pole readout (5 degenerate coordinates, margin `0.037`) made
  the gate unsuitable — rejected, not tuned until it passed.
- **v6** (unsupervised PCA response quotient): 8 of 35 role positions failed the readout, and
  the readout margin (`0.0045`) was far too small for a causal macro quotient — rejected.

Neither v5 nor v6 is kept as a script; both are documented refutations, not silently dropped
results, per `results/CAMPAIGN_STAGE_RESULTS.json` (`v5_endogenous_reentry_gate`,
`v6_unsupervised_pca_probe_quotient`).

## v7: architecture regimes

A runtime diagnostic (not kept as a script; not a code dependency of v10–v14) established two
distinct layer-setting regimes in the active checkpoint — outer layers `{0–4, 31–35}` and core
layers `{5–30}` — and showed `fused` and `s_attn` are numerically identical under the active
configuration (logits max abs. difference `0.0`, cache max difference `0.0`). This is the origin
of the entry/core/terminal block structure used by every later block battery (v8–v12).

## v8–v10: the exact three-block intervention battery

Moving to exhaustive `2³` coherent block interventions over entry `0–4` / core `5–30` / terminal
`31–35` removed learned response classifiers, but exposed the normalization problem described in
`directional-normalization.md`: v8 is flagged `diagnostic_only` in
`results/CAMPAIGN_STAGE_RESULTS.json` because some cut scores exceeded 1 (source/future macro
alphabets were inconsistent); v9 is `completed_with_normalization_warning` (denominator sensitive
to the chosen support closure). **v8 and v9 are not kept as scripts** — both are superseded,
diagnosed-and-fixed intermediate stages; their numeric findings (horizon-1 mean Γ `0.5` for v8,
`0.25` for v9) are preserved in `results/CAMPAIGN_STAGE_RESULTS.json`.

`scripts/quantum_gamma_block_multihorizon_v10.py` **is** kept: it is the final, corrected block
battery (all class audits passed, zero repeat noise) and is imported directly by v11 and v12.
Its own mean-context Γ under the old normalization still fell with horizon
(`0.155 → 0.193 → 0.221` for horizons 1/2/4) — this residual instability is exactly what v11
diagnoses.

## v11: closure-sensitivity diagnosis (decisive)

See `directional-normalization.md`. Verdict: the instability is a normalization artifact, not a
causal one.

## v12: source-normalized directional efficiency — the supported candidate

Final measured result on the real Quantum target model, entry/core/terminal blocks, uniform
independent binary block interventions, endogenous autoregressive re-entry, mean over eight
contexts:

| Horizon | Mean Γ_dir |
|---|---:|
| 1 | 0.500000 |
| 2 | 0.543926 |
| 4 | 0.578125 |
| 8 | 0.606426 |

The score is bounded context by context and exactly closure-invariant at every tested depth
(`results/quantum_gamma_directional_efficiency_v12_results.json`,
`results/gamma_directional_evidence_v15.json`). The weak-cut return information increases with
horizon, consistent with feed-forward organization within a cycle and increasing recurrent
return across cycles.

## v13–v14: autonomous exact audits

Both audits passed all hard checks (see `directional-normalization.md`); neither requires the
Quantum checkpoint.

## v15: theory status — status `current`, not yet adopted by the manuscript

See `open-semantic-question.md`.

## Reproduction

The kept scripts (`quantum_gamma_kv_experiment_v1.py`, `quantum_gamma_block_multihorizon_v10.py`,
`quantum_gamma_closure_sensitivity_v11.py`, `quantum_gamma_directional_efficiency_v12.py`) call
into a locally-hosted "Quantum" model checkpoint and its runtime (`quantum_model.py`,
`quantum_layer.py`, `paged_sparse_kv.py`, tokenizer, checkpoint shards). Those runtime files are
intentionally **not** duplicated in this archive — they belong to a separate, local model
repository, may contain unrelated implementation comments or machine-specific paths, and are not
part of what this campaign is certifying. Without them, a command such as

```bash
python3 scripts/quantum_gamma_directional_efficiency_v12.py \
  --root /path/to/quantum --model-dir /path/to/quantum/quantum_vibe_thinker
```

cannot run end-to-end in this edition; the exact recorded output is reproduced verbatim in
`results/`. The fully autonomous audits do run standalone:

```bash
python3 scripts/gamma_directional_normalization_regression_v13.py
python3 scripts/gamma_directional_quotient_invariance_v14.py
```
