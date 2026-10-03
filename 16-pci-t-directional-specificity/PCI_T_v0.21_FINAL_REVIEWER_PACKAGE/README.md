# PCI_T v0.21 — Final Reviewer Package

This package contains the compact evidence needed to review the final PCI_T v0.21 directional-specificity result for Qwen3-8B. It deliberately omits exploratory trial-and-error history and keeps only the final methodological rationale, the adversarial validation, the prospective replication gate, the final sealed measurement, provenance, and a standalone verifier.

## What is being measured

The historical PCI_T v0.20 score is a perturbational state-transition complexity measure. Subsequent adversarial testing showed that a high absolute PCI_T score can also occur in random or chaotic systems. The final v0.21 derivative therefore measures **directional specificity** rather than absolute complexity alone.

At a fixed source layer and perturbation amplitude, one designated direction is compared with 15 orthogonal RMS-matched control directions. The core quantity is the ratio between PCI_T in the designated direction and the median PCI_T across the matched controls. A value near 1 means that the designated direction is not special. A value above 1 means that it evokes a systematically richer downstream perturbational response than matched orthogonal directions.

## Why the adversarial benchmark matters

The synthetic benchmark includes two adversarial families with high generic dynamical richness: an untrained random linear system and an uncoupled chaotic system. Their median directional enrichment is approximately 1.0. Three directionally structured systems — feed-forward, branching, and recurrent — show median enrichment between approximately 7.1 and 8.4, with a median directional win rate of 1.0.

This result does **not** mean that PCI_T is a recurrence detector, a consciousness detector, or a general measure of task usefulness. The structured feed-forward control also passes. What the benchmark supports is narrower: directional specificity can distinguish selective, directionally organized propagation from generic random or chaotic richness under this synthetic test.

## Qwen3-8B prospective chain

Development data localized the strongest directional effect to source layer index 27. Because this localization was discovered on development data, it was not treated as confirmatory evidence. The rule was frozen before opening the CAL3 holdout.

CAL3 prospectively replicated the late-layer pattern. Layer 27 was the deepest-dominant layer in 5 of 6 episodes, the system-level L27 enrichment was 1.329707, both frozen token fractions exceeded 1.20, and the late-versus-lower-layer ratio was 1.314875. This pass authorized opening SCORE2 under the already-frozen final measurement contract.

SCORE2 then produced the final primary scalar:

**PCI_T_DS_L27 = 1.381374456016**

with a 95% episode-bootstrap interval of:

**[1.200589200476, 1.609210775101]**

The secondary absolute late-layer source score is:

**PCI_T_ST_L27_SOURCE = 343.9609375**

with a 95% episode-bootstrap interval of:

**[300.640625, 380.2578125]**

## How to interpret 1.381

The primary value 1.381 is a **ratio**, not an absolute PCI_T score. Under the frozen aggregation, it means that the endogenous residual direction at source layer index 27 produced about 38% more PCI_T state-transition complexity than the median of 15 orthogonal RMS-matched directions.

It does not mean that model parameters were increased, and it does not mean that PCI_T rose from 277.96 to 343.96. Model weights remain frozen. The intervention is applied to an activation state in the residual stream. The historical v0.20 score of 277.9609375 and the X8 secondary score of 343.9609375 use different panels and aggregations, so they should not be interpreted as a before/after pair.

## Interpretation boundary

The result supports a reproducible, prospectively confirmed late-layer directional specificity of perturbational complexity in Qwen3-8B under the frozen protocol. It provides an independent perturbational observable that can be compared with C_O in paired prospective experiments.

It does not by itself establish the incremental validity of C_O, functional usefulness, recurrence specifically, consciousness, or a consciousness scale. Those stronger claims require independent targets and paired tests defined outside PCI_T.

## Files

- `PCI_T_V021_REVIEWER_SUMMARY.md` — compact reviewer-facing narrative.
- `PCI_T_V021_FINAL_RESULTS.json` — consolidated final numerical results and interpretation boundaries.
- `PCI_T_V021_PROVENANCE.json` — hashes and frozen-contract provenance.
- `verify_pci_t_v021_final.py` — standalone verification of the included raw result files and key derived statistics.
- `raw_results/R01_DIRECTIONAL_SPECIFICITY_RESULT_v0.21-X4.json` — final synthetic directional-specificity benchmark.
- `raw_results/R01_CAL3_LATE_LAYER_GATE_RESULT_v0.21-X7.json` — prospective CAL3 replication gate.
- `raw_results/R01_SCORE2_FINAL_RESULT_v0.21-X8.json` — final prospective SCORE2 measurement.

Run:

```bash
python verify_pci_t_v021_final.py
```

A successful verification exits with status code 0 and prints the final primary and secondary values.
