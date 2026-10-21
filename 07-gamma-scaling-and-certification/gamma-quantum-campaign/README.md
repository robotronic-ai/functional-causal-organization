# Gamma-Quantum campaign — Γ on a real target model, candidate normalization

Clean English edition of the Quantum-model causal-integration campaign, folder
`07-gamma-scaling-and-certification/gamma-quantum-campaign/`. This is a genuinely **executed**
empirical campaign against a real target transformer (internally called "Quantum"), not a
design/protocol awaiting a model. It developed from a layerwise K/V intervention probe into a
bounded, closure-invariant candidate normalization for bidirectional causal integration.

## Reproduction discipline

This campaign follows the same discipline as every other campaign in this repository: a
protocol/battery is declared before a stage is scored; each stage's exact numeric result is
recorded whether it passed or was rejected; and a candidate is never silently replaced by a
tuned version without the rejected attempt being kept in the record. Where a stage is a
self-contained exact audit (v13, v14) it requires no external model and can be re-run directly;
where a stage measures the real target model (v11, v12) the exact recorded numbers are kept
verbatim here even though the model runtime itself is not redistributed (see
`docs/campaign-chronology.md`, "Reproduction"). Verdicts are never silently converted into a
single clean number: this campaign explicitly reports **an unresolved semantic choice** (see
`docs/open-semantic-question.md`) rather than picking one candidate and presenting it as settled.

## Main results

### A closure-sensitivity artifact in the previous normalization (v11, decisive)

The previous Γ normalization divides by the raw numerical cardinality of the future K/V support.
At fixed intervention content, this denominator grows with autoregressive closure depth even
though the measured directional causal information does not — so Γ appeared to fall from `0.25`
to `≈0.109` on the entry-block weak cut purely from closure depth. This is a normalization
artifact, not a causal one. See `docs/directional-normalization.md`.

### A source-normalized candidate that fixes it, exactly closure-invariant, and executed on the real model (v12)

\[
\Gamma_q^{\mathrm{dir}}(\pi) = \tfrac12\Big[\tfrac{J_q(A\to B')}{H_q(A)} + \tfrac{J_q(B\to A')}{H_q(B)}\Big],
\qquad \Gamma_q^{\mathrm{dir}} = \min_\pi \Gamma_q^{\mathrm{dir}}(\pi).
\]

Bounded by construction (`0 ≤ Γ_q^{dir} ≤ 1`), agrees exactly with the previous formula on
equal-capacity cuts, and is exactly invariant to closure depth and to coherent physical
replication (v13/v14 autonomous audits, both self-contained). Measured on the real Quantum
target model with the entry/core/terminal block battery: mean `Γ_dir` = `0.500` (horizon 1),
`0.544` (horizon 2), `0.578` (horizon 4), `0.606` (horizon 8). See
`docs/directional-normalization.md` and `docs/campaign-chronology.md`.

### An unresolved semantic choice — the manuscript definition is NOT changed

The candidate and the previous normalization measure genuinely different things on
unequal-capacity cuts (bottleneck saturation vs. mean bilateral utilization). This campaign's
explicit recommendation is **not** to replace the manuscript's Γ definition until this axiom is
settled, and to keep reporting the raw directional `J_q` terms alongside any normalized value.
See `docs/open-semantic-question.md`.

### Rejected intermediate stages, kept as documented findings, not as scripts

v1–v3 (self-separability is not essentiality), v5 (endogenous-reentry gate, degenerate readout),
v6 (PCA response quotient, unstable readout), v8 (`diagnostic_only`, inconsistent alphabets), v9
(`completed_with_normalization_warning`) were all rejected or superseded. Their exact numeric
findings are preserved verbatim in `results/CAMPAIGN_STAGE_RESULTS.json` and narrated in
`docs/campaign-chronology.md`; the scripts themselves are not carried forward because none of
them is a code dependency of the final v10–v14 chain (see "What was intentionally left out"
below).

### v4 and v7: real findings, scripts not kept, findings load-bearing for the final battery

v4 established that layer 35 is causal *only* through the emitted-token re-entry path (not
one-step, not under token clamping) — this is why every later battery uses endogenous, not
clamped, re-entry. v7 established the entry/core/terminal architecture regimes used by every
later block battery, and the exact numerical identity of `fused`/`s_attn` under the active
configuration. Neither script is a code dependency of v10–v14, so neither is kept as a script,
but both findings are narrated in `docs/campaign-chronology.md` and preserved verbatim in
`results/CAMPAIGN_STAGE_RESULTS.json`.

## Relation to `gamma-cert-target-transformer/`

The `baseline_certifier/` package that shipped inside the original Quantum-campaign archive is
**byte-identical** to the sibling folder `07-gamma-scaling-and-certification/gamma-cert-target-transformer/`
in this repository (same `RUN_STATUS.json`, same certificate code, same self-test). It is not
duplicated here. That sibling folder documents a statistically certified `V1`/`V3`/`NA` Γ
machinery whose real target run is **blocked on missing model artifacts** for a *different*
model (a recurrent-memory transformer, not Quantum). Do not read this campaign's real,
point-estimate `Γ_dir` numbers on Quantum as satisfying that certificate's target-run obligation
— they are a different model, a different (not yet statistically certified) normalization, and
answer a different question.

## Layout

```
README.md                              this file
docs/
  directional-normalization.md          Gamma_dir definition, boundedness proof, invariance evidence
  campaign-chronology.md                v1-v15 narrative, rejected stages, final result table
  open-semantic-question.md             the unresolved bottleneck-saturation vs mean-utilization choice
scripts/
  quantum_gamma_kv_experiment_v1.py            shared experiment infrastructure (imported everywhere below)
  select_quantum_gamma_sites_v2.py             final intervention-site selection (v1 superseded, not kept)
  quantum_gamma_block_multihorizon_v10.py      final entry/core/terminal block battery
  quantum_gamma_closure_sensitivity_v11.py     closure-sensitivity diagnostic (decisive)
  quantum_gamma_directional_efficiency_v12.py  final candidate measurement on the real target model
  gamma_directional_normalization_regression_v13.py  autonomous regression audit
  gamma_directional_quotient_invariance_v14.py       autonomous replication-invariance audit
results/
  CAMPAIGN_STAGE_RESULTS.json            master per-stage summary (v1-v15), including rejected stages
  gamma_directional_evidence_v15.json
  gamma_directional_normalization_regression_v13_results.json
  gamma_directional_quotient_invariance_v14_results.json
  quantum_gamma_closure_sensitivity_v11_results.json
  quantum_gamma_directional_efficiency_v12_results.json
  quantum_gamma_kv_pilot_v1.json
  quantum_gamma_site_selection_v2.json
  ARCHITECTURE_CONFIG_EXCERPT.json       portable excerpt justifying the entry/core/terminal block structure
```

## What was intentionally left out

- `baseline_certifier/` — byte-identical to `gamma-cert-target-transformer/`; see above.
- `inspect_quantum_internal_state_v3.py`, `quantum_gamma_block_exact_v8.py`,
  `quantum_gamma_block_closed_v9.py`, `quantum_gamma_reentry_pilot_v4.py`,
  `quantum_layer_regime_diagnostic_v7.py`, `quantum_gamma_probe_quotient_v6.py`,
  `validate_quantum_gamma_reentry_quotient_v5.py`, `validate_quantum_gamma_quotient_v2.py`,
  `validate_quantum_gamma_quotient_v3.py`, `select_quantum_gamma_sites.py` (v1) — rejected,
  superseded, or one-off diagnostics that are not code dependencies of the final v10–v14 chain.
  Every one of their findings is preserved in `results/CAMPAIGN_STAGE_RESULTS.json` and narrated
  in `docs/campaign-chronology.md`; nothing that was refuted or rejected was hidden, only the
  duplicate/superseded script files.
- `LANGUAGE_AUDIT.json`, `MANIFEST.json`, `SHA256SUMS.txt` — process bookkeeping specific to the
  source archive's own release process.
- `notes/gamma_directional_normalization_note_v12.md`, `notes/gamma_directional_theory_audit_v15.md`
  — their content is restated in `docs/directional-normalization.md` and
  `docs/open-semantic-question.md`.

## How to verify

```bash
# Self-contained; no target model required.
python3 scripts/gamma_directional_normalization_regression_v13.py
python3 scripts/gamma_directional_quotient_invariance_v14.py
```

The remaining scripts require a locally hosted "Quantum" model checkpoint and runtime
(`quantum_model.py`, `quantum_layer.py`, `paged_sparse_kv.py`, tokenizer, checkpoint shards),
none of which is redistributed here (see `docs/campaign-chronology.md`, "Reproduction"). The
exact recorded numbers from those runs are reproduced verbatim in `results/`.

See the top-level `../README.md` of this folder for how this campaign relates to the other four,
and `04-gamma-delta-characterization/` for the analytic Γ axiom work this campaign builds on.
