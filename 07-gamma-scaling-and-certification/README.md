# 07 — Γ scaling and certification

Clean English edition of five related bundles about scaling the bidirectional causal-integration
score Γ to larger systems and certifying it rigorously, plus one related bound on the KV-cache
temporal boundary. This folder groups them because they all extend the Γ/Δ work in
`04-gamma-delta-characterization/` (which established what Γ *is*, axiomatically) into the
question of what Γ *is worth*, and how confidently it can be claimed, at scales and on real
models where exhaustive cut enumeration or exact computation is no longer possible.

## Reproduction discipline

This folder follows the same discipline as every other campaign in this repository: a protocol
or theorem is stated before it is tested; a candidate or certificate is checked against exact or
statistically-bounded evidence; and no result here silently turns a partial, blocked, or
unexecuted finding into a clean single number. Two of the five sub-bundles
(`gamma-quantum-campaign/`, `gamma-cert-target-transformer/`) concern applying Γ to a real target
transformer; **their real-model execution status is reported plainly and is not the same for
both** — one is genuinely executed, the other is blocked on missing model artifacts (see the
combined status table below). The other three sub-bundles (`gamma-cert-causal-quotient/`,
`gamma-cert-v0.41/`, `kv-bound-boundary/`) are self-contained exact or exactly-calibrated
computations that do not depend on any target model, and are fully executed. Verdicts are
reported as `V1`/`V3`/`NA` where the source protocol defines them (certified-interval work),
and as explicit execution-status flags (`EXECUTED_REAL_MODEL`, `EXECUTED_EXACT_FINITE`,
`NOT_EXECUTED_REAL_MODEL`, etc.) elsewhere — never collapsed into a single "done" checkmark.

## The five sub-bundles

### 1. `gamma-quantum-campaign/` — EXECUTED on a real target model

A 15-stage campaign (v1–v15) run against a real target model ("Quantum",
`QuantumModelForCausalLM`, 36 layers). Diagnosed and fixed a normalization artifact (Γ falling
with autoregressive closure depth even though the underlying causal information did not change),
proposed a source-normalized candidate (`Γ_dir`, bounded, closure-invariant, replication-invariant,
agrees with the previous formula on equal-capacity cuts), and measured it on the real model:
mean `Γ_dir` = 0.500 / 0.544 / 0.578 / 0.606 at horizons 1/2/4/8. Explicitly leaves open a
semantic choice (bottleneck saturation vs. mean bilateral utilization) and recommends **not**
changing the manuscript's Γ definition until that axiom is settled.

### 2. `gamma-cert-target-transformer/` — NOT executed on the real target model

A statistically certified-interval protocol (`V1`/`V3`/`NA`, `L_Γ ≤ Γ_q ≤ U_Γ` with 95%
simultaneous confidence, via a Fano/spectral lower bound and a cross-entropy upper bound) for a
*different* target — the manuscript's frozen recurrent-memory transformer. The certificate
implementation is complete and self-tested (synthetic system, exact Γ `0.2`, certified interval
`[0.093, 0.252]`, itself only reaching `V3` since its width exceeds the `0.10` precision band).
**`target_experiment_executed: false`, `status: blocked_on_target_model_artifacts`** — the
checkpoint, tokenizer, intervention hook, and quotient-discovery procedure are all missing from
this runtime. This status must not be confused with sub-bundle 1's genuinely executed result on a
different model.

### 3. `gamma-cert-causal-quotient/` — EXECUTED, exact finite stress test

A self-contained test of the causal-minimal-state quotient gate both certificate protocols above
depend on: a four-role recurrent ring, implemented with redundant coherent replicas plus a
tunable null-space perturbation, shows the macrostate `Γ = 0.5` is exactly invariant to the
perturbation while the raw physical `Γ` drifts from `0.167` to `0.485` as the raw-algebra
quotient-sufficiency test starts failing. Motivates a five-step safe certification procedure.

### 4. `gamma-cert-v0.41/` — EXECUTED, exact/calibrated, no target model

A scalable family of certified Γ lower and upper bounds: an additive fractional-packing theorem
(with an exact dual MILP, verified against exhaustive enumeration) that fixes a scaling
limitation of a prior max-only witness hierarchy, and a decoding-graph + spectral (Cheeger)
one-sided lower bound, calibrated exactly on small systems and demonstrated at `n` up to `1024`
roles on a copy-expander causal benchmark where Γ is exactly graph conductance by construction.

### 5. `kv-bound-boundary/` — theorem PROVED and toy-verified; real-model application NOT executed

A different but related bound: how much source→next-state information can bypass the emitted
token via a persistent (KV-cache) or latent channel. Resolves a `P_{t+1}` notation ambiguity,
proves `B_q ≤ H_q(Y_t∣C_t,U_t) + C_bypass,q`, and verifies it exactly (zero slack, correct
qualitative discrimination across four intervention modes) on a synthetic toy-attention self-test.
The "15–17 bit" manuscript claim becomes an experimentally falsifiable corollary — **but the
real-model falsification protocol itself has not been run**; this is stated plainly in that
folder's docs.

## Relation to `04-gamma-delta-characterization/`

Folder 04 establishes what Γ is: under minimal axioms alone it is not uniquely forced (arithmetic
mean, geometric mean, and min all qualify), and is characterized uniquely only after adopting an
explicit marginal-independence axiom. Every sub-bundle here takes that axiomatic Γ (specifically
the arithmetic-mean, `G-WEAK` reading) as given and asks a different question: how to compute,
bound, or certify it at scales or on real models where folder 04's exact finite-grid audits no
longer apply. `gamma-quantum-campaign/`'s open semantic question (bottleneck saturation vs. mean
utilization on unequal-capacity cuts) is a direct descendant of folder 04's finding that the
aggregator is not axiomatically forced without further hypotheses — it is the same kind of
open choice, now surfaced by a normalization question instead of an aggregation question.

## Combined status table

| Sub-bundle | Kind | Execution status | Headline verdict |
|---|---|---|---|
| `gamma-quantum-campaign/` | real-model empirical campaign | **EXECUTED_REAL_MODEL** | source-normalized `Γ_dir` supported; semantic choice on unequal-capacity cuts left open |
| `gamma-cert-target-transformer/` | certified-interval protocol | **NOT_EXECUTED_REAL_MODEL** (`blocked_on_target_model_artifacts`) | implementation ready + self-tested (`V3` on synthetic self-test); no result on the real target |
| `gamma-cert-causal-quotient/` | exact finite stress test | EXECUTED_EXACT_FINITE | quotient-sufficiency gate validated; raw Γ representation-sensitive, macrostate Γ invariant |
| `gamma-cert-v0.41/` | exact/calibrated scale certificates | EXECUTED_EXACT_OR_CALIBRATED | fractional-packing + spectral certificates proved and demonstrated to n=1024 |
| `kv-bound-boundary/` | proved theorem + toy verification | theorem PROVED; toy self-test EXECUTED; real-model protocol NOT_EXECUTED | `B_q ≤ H(Y∣C,U)+C_bypass` confirmed exactly on synthetic system; "17 bits" claim not yet tested on a real model |

See `results/STATUS.json` in this folder for the same table in machine-readable form, and each
sub-bundle's own `results/STATUS.json` for full detail.

## Layout

```
README.md                          this file
results/
  STATUS.json                      combined machine-readable status across all five sub-bundles
gamma-quantum-campaign/            README.md + docs/ + scripts/ + results/
gamma-cert-target-transformer/     README.md + docs/ + scripts/ + results/
gamma-cert-causal-quotient/        README.md + docs/ + scripts/ + results/
gamma-cert-v0.41/                  README.md + docs/ + scripts/ + results/
kv-bound-boundary/                 README.md + docs/ + scripts/ + results/
```

## What was intentionally left out (folder-wide)

No `*_PRIVATE.json` oracle files exist anywhere in any of the five source bundles — none were
withheld for that reason. What was left out, per sub-bundle, is documented in each sub-bundle's
own README under "What was intentionally left out": rejected/superseded diagnostic scripts in
`gamma-quantum-campaign/` (findings kept, scripts dropped), a byte-identical duplicate of
`gamma-cert-target-transformer/` that shipped inside the Quantum campaign's own archive as
`baseline_certifier/` (not copied a second time), and process-bookkeeping files
(`LANGUAGE_AUDIT.json`, `MANIFEST.json`, `SHA256SUMS.*`) specific to each source archive's own
release process.
