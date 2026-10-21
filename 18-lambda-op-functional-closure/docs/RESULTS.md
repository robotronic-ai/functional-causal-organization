# Results — Λ at operation level, functional closure

Closure criterion throughout: a frozen description is refuted on a panel as soon as one fiber contains two systems with different behavioral targets (a *heterogeneous collision*). A candidate is promotable only with zero collisions, at least 20 % of systems in non-trivial fibers, at most 80 % fiber fraction and zero sufficiency loss; later phases add adjusted mutual information (AMI) above a permutation 95th percentile. Every validation opens a fresh panel once, with no refit.

## 1. Synthetic universes

| Width | Coverage | Fibers | Non-trivial fibers | Functional collisions | Verdict |
|---|---|---:|---:|---:|---|
| 2 bits | exhaustive, 256 systems | 256 | 0 | 0 | PASS (injective) |
| 3 bits | exhaustive, 16,777,216 maps | 1,710,388 | 666,292 | 0 | PASS |
| 4 bits | declared envelope, 1,024 systems | 512 | 512 | 0 | PASS |
| 6 bits | declared envelope, 1,024 systems | 512 | 512 | 0 | PASS |
| 8 bits | declared envelope, 1,024 systems | 512 | 512 | 0 | PASS |

A separate finite diagnostic suite (`synthetic_closure/results/CO_COMPONENT_DIAGNOSTICS.json`) passes five witnesses — Γ normalization, Δ composition, R horizon, Λ typed binding, ID-to-OP generalization. They validate the diagnostic machinery, not Qwen.

## 2. Qwen3-8B — compact descriptors refuted on fresh panels

| Phase | Descriptor | Fresh systems | Collision pairs | Verdict |
|---|---|---:|---:|---|
| Fixed descriptor | operation-address effect | 96 | 6 collision fibers | REFUTED |
| Validation 01 | refined operation-address effect | 128 | 7 collision fibers | REFUTED |
| Validation 02 | substitute-first refinement | 128 | 3 collision fibers | REFUTED |
| Screen 03 | operation-indexed passive descriptor | 352 (screen) | 2 | not promoted (1.7 % coverage) |
| Screen 04 / Validation 05 | active causal kernel, hybrid refinement | 32 / 64 | 18 | REFUTED |
| Screen 06 | joint relational kernel | 96 (screen) | 0 | no useful candidate (near-injective) |
| Screen 07 | cross-target routing topology | 24 (screen) | 63 | no useful candidate (compressive, lossy) |
| Validation 11 | PCA20 quotient with local split | 64 | 136 | REFUTED |
| Validation 12 | typed content effect, 2 features | 64 | 28 | REFUTED |
| Validation 13 | typed content effect, 3 features | 64 | 15 | REFUTED (fidelity transports: 16 % within-fiber distortion) |

What the refutations show: compressing the causal response loses functionally relevant information in two distinct ways — descriptors that keep only propagation magnitude merge systems that route content differently, and descriptors that keep typed content with too few coordinates transport their fidelity yet still merge systems that differ on `SUBSTITUTE`.

## 3. Controls on the measurement

- **Metrology 09.** Six systems, three independent process/model reloads each: behavioral target vectors, cell outcomes, descriptors and continuous internal measurements are exactly identical across reloads (CUDA/bfloat16). Refutations are not run-to-run noise. Systematic bias relative to float32 was not estimated.
- **Fidelity-First 10.** A PCA20 representation reproduces the 288-dimensional cross-target proxy with pairwise-distance Spearman 0.999, yet its quotient still collides (4 pairs): global fidelity does not guarantee local functional resolution.
- **Component audit 15 and formula contract 16.** A 16-system pilot showed that the available estimators were proxies rather than the manuscript quantities (bidirectional RMS for Γ, effective rank × propagation for Δ, a degenerate cycle estimator for R). The raw label `DIFFUSE` was therefore adjudicated `UNRESOLVED`, no partition was promoted, and formula contracts for Γ, Δ, R and Λ were frozen (SHA-256 `c17c401f…d0d04`) before any further selection. A defect in the parser of the source-localized routing map was found and corrected at this stage (the parser now follows the rendered clause order, and empty or partial routing panels are rejected before scoring); the correction is recorded in `CO_BASE_OPERATOR_15_PROTOCOL.md` and `CAMPAIGN_STATUS.json`.

## 4. Qwen3-8B — formula-contract measurement and prospective holdout

- **Formula Measurement 17** (12 development systems, target-blind measurement): Γ, Δ, an adapted re-entry R (storage-level negative controls PASS, label-duplication invariance PASS) and a finite joint typed Λ signature. At fixed resolution 1/16 the full OP profile is injective (12 fibers for 12 systems), so its closure is trivial and rejected by the compactness criterion.
- **Quotient/Resolution 18** (same 12 systems, 66 declared screens): the coarsest admissible representation is Λ only — the joint typed signature truncated to its top-ranked response — with 7 fibers, zero collisions, AMI 0.853 above the grid-wide permutation threshold 0.437. Exploratory, not promotable on development data.
- **Holdout 19** (32 fresh systems, candidate frozen, no refit, no grid reopening):

| Measure | Value |
|---|---:|
| Fibers | 7 |
| Systems in non-trivial fibers | 29 / 32 (90.6 %) |
| Heterogeneous collision pairs | **0** |
| AMI | 0.806 |
| Permutation 95th percentile | 0.119 (p = 0.0002) |
| Normalized sufficiency loss | ≈ 0 |
| Verdict | `PROSPECTIVE_HOLDOUT_PASS` |

The information captured is concentrated on the three `SUBSTITUTE` cells of the 12-cell behavioral target.

- **Replication 20** (64 further fresh systems, identical candidate and rule): frozen, not yet executed.

## Reading

On the declared Qwen battery, one frozen OP-level Λ quotient is functionally closed on a prospective holdout, after a sequence of simpler or differently typed descriptors were refuted. This supports the manuscript's expectation that closure on a real model requires keeping *which content is bound to which role*, and is consistent with the compression limit of §17.7: every refuted compression is an instance of a non-injective summary losing a distinction the target draws. It does not establish closure of the complete C_O, nor closure beyond this battery.
