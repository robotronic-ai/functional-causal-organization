# T343 — Frozen performance-first protocol for Qwen3-8B

**Decision date:** 2026-10-18. **Status:** approved design, pending matched-arm acquisition; NOT a Qwen3-8B prediction, perturbational-response, or closure certificate.

## 0. Independent result tracks

The following may proceed and be reported independently:

- **P0: External performance scorecard.** Scores and confidence intervals per external target. No organizational-profile variation is required to publish a score.
- **P1: Out-of-sample predictive comparison, V45.1 §17.6.** Compare specified rivals against numeric C_O features on matched realizations and a sealed holdout.
- **P2: Signed intervention and dose-response test, V45.1 §17.6.** Preregister the manipulated route, predicted target direction, and matching controls before observing downstream targets.
- **P3: Formal informative graded closure, T341.** Direct within-cell pairwise comparisons and preregistered eta grid; never a gate for P0–P2.

Each initiated analysis is labeled EXPLORATORY or CONFIRMATORY before measuring its targets, with all initiated outcomes, nulls, invalidations and exclusions published. No successful track entails the others.

## 1. Measurement integrity and scorecard (P0)

**Matched unit:** one fully specified, independent system realization/arm with model weights, dtype, quantization, causal intervention, donor law, temporal horizon, gauge, task generator, prompts, seeds, decoding rule, and measurement protocol identified. The 4,096 donor masks in a single FULL12 cube are *not* 4,096 independent predictive realizations.

The FCA-Bench v0.2.3/v0.2.4/v0.2.7 values in V45.1 §17.9 were taken on **8-bit quantized** Qwen3-8B and must be displayed as **HISTORICAL_Q8_UNPAIRED**. The C_L24 organization experiments use bf16 and cannot be paired silently with those FCA results. Remeasure all selected external targets on the same bf16 realization/arm before association analyses. PCI_T's archived directional ratio must likewise be linked to its actual run configuration before a matched analysis; its existing status alone does not establish pairing.

**Primary independent targets (both remain separate):**

1. `FCA_E1`: strict four-arm counterfactual erasure performance. Bounded measured score with test-specific uncertainty, exact benchmark version and task/decoding contract.
2. `PCI_T_DS_L27`: L27 directional specificity, designated perturbation response divided by the median over 15 norm-matched orthogonal directions. Native *ratio* scale, including uncertainty; never convert to an arbitrary /100 score or use absolute PCI_T complexity as a substitute.

**Secondary targets:** `FCA_C3`, `FCA_T2`. **Descriptive audit targets:** `FCA_T1`, `FCA_T3`, `FCA_B1`, `FCA_R3`, optional further FCA dimensions, and PCI_T absolute complexity in its own panel. All FCA behavioral and causal panels remain distinct; `STRUCTURAL_ABSENCE`, `INVALID`, and `MISSING` are typed states, not numeric zero. Publish per-target 95% intervals and denominators, including valid sample counts. No composite consciousness score or averaged FCA index.

The primary-target designation is a *prospective choice for the next study*; it does not retroactively turn archived data into held-out evidence. Other targets can be prospectively promoted only by a NEW preregistration before data collection.

## 2. Precommitted theory-versus-rivals models (P1)

**Prediction unit:** independent matched *arm*, not tokens, repeated donors or individual tasks from the same arm. Individual tasks/episodes are measurement replicates; their within-arm dependence must be respected. Group all episodes, seeds and donors of a realization in the same split. Where possible, hold out entire intervention families for stronger generalization.

### Feature definitions — same contract across arms

All numeric features are measured on matching bf16 arms, fixed causal boundary, role typing, calibrated delta, q_donor and agreed readout/horizon. All normalization, missingness handling and transforms are learned within training folds only. Four `NONZERO_CERTIFIED` labels are not four predictive numeric features.

- `Gamma`: numeric declared finite-gauge min-cut score or fixed comparable conservative bound, not a binary certificate.
- `Delta`: **predeclared scalar projection `min(Delta_H, Delta_S)`**, using matched cut/role conventions. Retain the two component values for a secondary analysis, not to increase the main model dimension after inspection.
- `R`: numeric same-source finite state reentry score/bound, with identical k0 and denominator convention across arms; do not mix R_probe and R_state.
- `Lambda`: numeric L24 relation density on declared directed role-to-reader cells at **strict >delta_abs** (secondary >2delta_abs), keeping fixed denominator and distinction between retained roles and contextual reader. A mere positive status is not an adequate feature.

If these four comparable numeric features are unavailable, run P0 and P2 normally, and label a reduced P1 model as exploratory. Do not impute absent evidence as zero.

### Baseline M0: four predeclared simple features

1. Matched RMS of the intervention/damage magnitude (including a sham baseline).
2. Content-to-output information under the same decoding law, with no use of future target labels.
3. Effective/passive memory available to the realization, under a fixed estimator.
4. Pre-intervention access-task score measured on **independent development episodes** (never the held-out target itself).

Parameter count and compute are invariant in the fixed-weight arm study and are recorded as metadata, not inserted as zero-variance regressors. For cross-model scaling these factors enter a separately preregistered baseline; that is NOT the same experiment.

### The principal model and matched-size rivals

- `M0`: intercept plus four simple covariates.
- `M1_CO`: M0 plus four organizational coordinates (`Gamma`, `Delta`, `R`, `Lambda`).
- `MR_INFLUENCE4`: M0 plus four predeclared summaries of the same directed, role-aligned influence matrix on the same arms: normalized Frobenius norm, spectral norm, stable rank (`||J||_F^2 / ||J||_2^2`), and direction asymmetry (`||J-J^T||_F / ||J||_F`) when a square role-aligned matrix is identifiable. Specify a fixed zero-matrix convention before acquisition.
- `MR_COMPETING`: an additional M0 plus four competing internal descriptors (e.g. rank, dimension, observability and controllability) **ONLY IF** all four have prospectively specified estimators on the same input-output/time horizon. A single-step influence matrix does not automatically identify state-space observability or controllability: report them as not measured until identifiable. Their unavailability is not a failure of C_O or of P0/P2.
- Ablations `M1_CO_minus_Gamma`, `_minus_Delta`, `_minus_R`, `_minus_Lambda` are diagnostic and are to be fitted only when the corresponding numerical features have valid comparable definitions.

Avoid tuning/selecting the best rival after seeing held-out labels. Rival selection, if any, is within development folds only. Additional rival feature blocks must be registered before their confirmatory run.

### Fits, splits and metric

Predeclare arm/family IDs and seal target labels for the outer evaluation. If >=5 independent groups are available, use **5-fold group-held-out cross-validation** with one outer prediction per arm and an inner grouped CV for tuning. If <5 groups, the full four-coordinate model comparison is **EXPLORATORY/UNDERPOWERED**, not an automatic rejection of the theory; proceed with P0/P2. Avoid splitting donor masks across folds. Use identical splits and preprocessing pipelines for every model and rival.

For numeric outcomes use ridge regression with intercept, feature standardization fit in each training fold, and a fixed inner-CV candidate grid of penalties `[0.01, 0.1, 1, 10, 100]`. For each target report held-out RMSE, MSE, raw R^2 and the signed difference

`Delta_R2_j = R2_heldout(M1_CO) - R2_heldout(MR_best_development_selected)`.

Negative R^2 is allowed and must not be clipped. A target with negligible held-out variance is flagged `TARGET_UNRESOLVED_FOR_R2`; use its held-out MSE and signed difference descriptively rather than reporting a misleading R^2. Matched size is a fairness diagnostic; regularization and genuinely independent groups are still required.

**Confirmatory decision, per primary target:** `CO_INCREMENTAL_PREDICTION_SUPPORTED` only if the lower endpoint of a *group-aware* one-sided 97.5% confidence interval for held-out Delta_R2 is >0 versus the best prospectively eligible matched-dimension rival, and M1_CO also beats M0. This assigns 2.5% one-sided familywise allocation to each of the two primary targets. A CI crossing zero yields `INCONCLUSIVE`; a reliably negative result is `RIVAL_FAVORED`. The joint theory claim requires separate explicit evidence, not simply selecting whichever target happened to succeed. Report coverage, effective number of independent groups, number of covariates, arm/family IDs, and all tested contrasts. Group-aware uncertainty methods must be committed before opening held-out outcomes. If group resampling is unstable, use `UNCERTAIN_UNDERPOWERED` rather than fabricate a narrow CI.

P1 is not formal closure, and a result on 10–20 arms does not justify a general theory victory by fitting five to nine coefficients.

## 3. Preregistered same-weight ablation and augmentation (P2)

**Primary contrast:** bf16 Qwen3-8B, architecture and weights fixed, same task episodes, with `BASELINE`, `SHAM`, and a targeted `L24_ROLE_ABLATION` arm. Pair intervention strength, active tokens, computational budget and prompt seeds as much as possible. An optional `AUGMENTATION` arm is separately declared before acquisition and is not assumed to improve C_O by design.

**Manipulation family:** target a declared L24 propagation/role-sharing mechanism (Delta H/S) and record numerical changes in Gamma, Delta H/S, R and Lambda. The preregistered signed hypothesis is that a **valid downward movement of measured Delta**, with confounds and other component changes within declared equivalence margins or directionally compatible, will be accompanied by a **non-positive change** in independent FCA_E1 and PCI_T_DS_L27 (relative to sham); the opposite directional prediction applies only to a prospectively specified successful augmentation. Directional hypotheses are tested, not guaranteed. If the intervention fails to change the intended coordinate, label `MANIPULATION_INVALID` and still publish the external performance outcomes.

Primary effect reports: baseline/sham/ablation paired target changes, per-episode uncertainty respecting arm pairing, dose (measured Delta change), slope or rank-based dose-response if at least 3 fixed preregistered nonzero dose levels are actually measured, and simultaneous simple rival changes (intervention norm, output information, passive memory, compute). One baseline–ablation pair can test the signed contrast, **not** a general monotone dose-response. No post-hoc target, reader, dose or best-world selection to rescue an adverse result. `same-weight` does not imply no distribution shift; document it.

**Scaling track (separate exploratory/confirmatory experiment):** use explicitly identified smaller/larger model configurations with their own weights, quantization and compute/training controls. Parameter count is not manipulated while holding weights fixed. Scaling need not raise any C_O coordinate or access target monotonically; test this hypothesis, never assume it.

## 4. Independent graded-closure detector (P3)

Retain T341's direct pairwise per-target implication, cell diameters, resolved target-variation rule, eta grid, uncertainty and statuses. Permutation baselines, normalized mutual information and adversarial discrimination are secondary diagnostics. P0–P2 may proceed and be published if P3 remains `OPEN_NOT_ESTABLISHED`.

## 5. Version, status and publication

The accompanying JSON is a preregistration *template frozen at the level of variables, models, target priorities, statistical tests and decision rules*. Arm IDs, precise calibration tolerances, seeds, dataset generators and instrument implementations MUST be inserted and sealed **before** the corresponding new confirmatory data are acquired. A missing seal means that run is exploratory; it does not block other tracks.

Current organization register on the declared Qwen3-8B L24 contract: `Gamma=NONZERO_CERTIFIED`, `Delta=NONZERO_CERTIFIED`, `R=NONZERO_CERTIFIED` (user-confirmed), `Lambda0=NONZERO_CERTIFIED` at strict >delta_abs. No recertification is requested. Formal global graded closure is still open; the present files make no new claim that any rival has been beaten or that any perturbational dose effect exists.
