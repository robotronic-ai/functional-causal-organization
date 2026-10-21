# Metrology 09: Repeatability and Compression-Fidelity Audit

## Purpose

This phase precedes any further Qwen3-8B closure refinement. It asks two separate questions:

1. Are the internal measurements and behavioral outcomes repeatable when the same systems are run again in the current CUDA/bfloat16 stack?
2. How much information about the rich cross-target causal-response proxy is lost by low-dimensional or categorical summaries?

These are metrology questions, not a new closure claim.

## Scope

The source object is the completed Cross-Target 07 panel. Its rich target-blind proxy has 288 coordinates per system:

`operation x source_address x target_address x horizon x internal_metric`

with:

- operations: `ID`, `KEEP`, `REVOKE`, `SUBSTITUTE`;
- source and target addresses: `H1`, `H2`, `H3`;
- horizons: 1 and 2;
- metrics: final-hidden, primary-block, post-primary-block, and reference-block relative RMS;
- active perturbation: `ZERO_SUFFIX` on layers 21-23.

This panel does **not** contain certified measurements of Gamma, Delta, or R. Therefore this phase cannot validate a scalar approximation of the complete `C_O` profile. Any one-dimensional summary is treated only as a diagnostic summary of the current Lambda/active-kernel proxy.

## A. Repeatability audit

`qwen3_8b_metrology_prepare.py` deterministically selects six systems from the completed Cross-Target 07 panel, preferring distinct 12-cell behavioral target vectors before filling remaining slots by unit order.

The exact six-system schedule is sealed. `qwen3_8b_metrology_run.py` then runs this same schedule three times. Each repeat is a separate Python subprocess and reloads Qwen3-8B from disk. All three runs use the current production setting `cuda / bfloat16`.

The score reports:

- exact stability of each 12-cell behavioral target vector;
- exact stability of each operation-address behavioral outcome;
- exact equality and relative ranges of continuous internal measurements;
- pooled within-system repeat SD divided by between-system SD for each internal coordinate;
- stability of every declared Cross-Target 07 categorical descriptor family.

An exploratory engineering label is also reported:

- `REPEATABILITY_CLEAN`: target stability = 1, selected descriptor stability = 1, and p95 noise/signal <= 0.05;
- `REPEATABILITY_CAUTION`: target stability = 1, selected descriptor stability >= 0.95, and p95 noise/signal <= 0.20;
- otherwise `REPEATABILITY_UNRESOLVED`.

These thresholds are engineering diagnostics, not theoretical closure criteria.

### Limitation

The repeatability audit measures variability in the current bfloat16 CUDA execution stack. It does not estimate systematic bfloat16 bias relative to a float32 reference model.

## B. Compression-fidelity audit

The preparation step also runs a target-blind CPU audit on all 24 Cross-Target 07 systems. The rich reference is the log-transformed 288-dimensional cross-target trajectory.

The following summaries are compared with that rich proxy:

- `GLOBAL_LOG_MEAN_1D`;
- `GLOBAL_LOG_MEAN_STD_2D`;
- `OPERATION_MOMENTS_8D`;
- `OP_SOURCE_MOMENTS_24D`;
- `DIAG_OFFDIAG_TEMPORAL_48D`;
- the existing `ALL_FLOW_H2_FORWARD` categorical topology code.

For continuous summaries, the audit reports:

- Spearman rank preservation of pairwise distances;
- 3-nearest-neighbor overlap;
- leave-one-out ridge reconstruction R2 of the rich response vector.

For the categorical topology code, it reports:

- Hamming-distance versus rich-distance rank correlation;
- nontrivial coverage;
- within-fiber rich distortion;
- rich variance explained by the induced fibers.

No behavioral target is used to define or fit these summaries.

## Decision rule

No new closure candidate is promoted from this phase. The output is diagnostic.

The next modeling step should be chosen only after both audits are inspected:

- if repeatability is poor, improve the measurement protocol before changing the causal descriptor;
- if repeatability is clean but low-dimensional summaries have poor fidelity, enrich the existing four-component measurements rather than interpreting collisions as evidence for an additional primitive component;
- if repeatability and fidelity are both strong, remaining closure failures become stronger evidence that the current measured object is structurally insufficient.
