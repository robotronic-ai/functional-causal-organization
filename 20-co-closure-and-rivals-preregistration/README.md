# 19 — Graded functional closure and rival comparison: frozen protocols (Qwen3-8B)

This folder holds the protocols that define, before any outcome is opened, how the working paper tests two claims about the organizational profile C_O = (Γ, Δ, R, Λ):

1. **Functional closure** of C_O relative to external targets (manuscript **§17.7**, *graded functional closure*).
2. **Incremental predictive validity against preregistered rivals** (manuscript **§17.6**).

The component certificates on which both rest are in [`../18-co-qwen3-8b-c-l24/`](../18-co-qwen3-8b-c-l24/). The external targets are PCI_T directional specificity ([`../16-pci-t-directional-specificity/`](../16-pci-t-directional-specificity/)) and FCA-Bench E1 ([`../17-fca-bench/`](../17-fca-bench/)).

**No result of these tests is reported here.** The protocols are frozen; the joint dataset they require has not been acquired.

## Frozen files (byte-identical; do not edit)

| File | Content |
|---|---|
| `frozen/T341_FROZEN_GRADED_CLOSURE_SPEC.md` | Graded closure: fixed objects, direct within-cell pair criterion, simultaneous uncertainty over cells, targets and the full η grid, non-triviality conditions, five verdicts, secondary diagnostics. |
| `frozen/T343_FROZEN_PERFORMANCE_RIVALS_ABLATION_PROTOCOL.md` | Performance-first protocol: independent result tracks, measurement integrity, targets, models, decision rules. |
| `frozen/T343_PREREGISTRATION.json` | Machine-readable form of the previous protocol. |
| `frozen/T344_RIVAL_PREREGISTRATION_LOCK.json` | Rival models, features, folds, contrasts, multiplicity and bootstrap, fixed before coding. |
| `frozen/T345_INFLUENCE4_EXTRACTION_LOCK.json` | Extraction convention of the four influence-matrix descriptors of the structural rival. |

SHA-256:

```
f2c0772cb069281ff2dae0b83b93ab8dca4064769f212e7f4ddb1fc537c5ee05  T341_FROZEN_GRADED_CLOSURE_SPEC.md
370f1d645cffb5a8f0968406cd76e174c700b7c60e0c1b85abe72b6de7a06883  T343_FROZEN_PERFORMANCE_RIVALS_ABLATION_PROTOCOL.md
4dcdf74bca5355e9bd90ea81cb0a5eccab577637e65a9239b1dfccd89bcc78df  T343_PREREGISTRATION.json
6b888be9fe3ef47bebb10b5d3b625f1498363ee75013bd2e80be76af850c85d9  T344_RIVAL_PREREGISTRATION_LOCK.json
5fa6481c18acac62d234159e71ad3cf5e882f7aa516546db643febc3306044d3  T345_INFLUENCE4_EXTRACTION_LOCK.json
```

Frozen, hashed protocol files are never retyped or translated: doing so would change their hash. One wording note: `T341_FROZEN_GRADED_CLOSURE_SPEC.md` calls the graded closure "proposed §17.7bis". In manuscript v45.3 the graded closure was adopted as the definition of §17.7; the exact exhaustive-collision formulation is its limiting case (finite enumerable universe, target A†, zero tolerance). The frozen content is unchanged.

## Graded closure in brief

For each η, target *j* and organizational cell *c*, the closure condition is D<sub>η,c,j</sub> = max over pairs (s, t) in the cell of d<sub>j</sub>(Y<sub>j</sub>(s), Y<sub>j</sub>(t)) ≤ ε<sub>j</sub>. Pairs are compared directly (no chains), ε<sub>j</sub> is fixed from repeatability, and the uncertainty is simultaneous over all pairs, cells, targets and η values. A closure is **informative** only if the target is resolved (full-universe diameter > ε<sub>j</sub>), at least two cells are occupied, and at least one cell holds two independent realizations. Five verdicts, per target and per η, never merged: `GRADED_CLOSURE_REFUTED_WITH_PAIR`, `OPEN_INVALID_OR_MISSING_TARGET`, `OPEN_UNCERTAIN`, `FORMAL_CLOSURE_ONLY_NON_INFORMATIVE`, `GRADED_CLOSURE_INFORMATIVE_CERTIFIED`.

## Rival comparison in brief

| Model | Variables |
|---|---|
| `M0` | intervention RMS (damage magnitude); output information; passive effective memory; pre-intervention independent access score |
| `M1_CO` | `M0` + Γ, Δ<sub>min,HS</sub>, R (same source state), Λ (density of cells above δ at layer 24) |
| `MR_INFLUENCE4` | `M0` + normalized Frobenius norm, spectral norm, stable rank and directional asymmetry of the 11 × 11 role-aligned causal-influence matrix |

Observability and controllability rivals are **not identified** (a one-step influence matrix does not identify them) and are not tested; effective rank is exploratory.

- **Targets.** `FCA_E1` (native [0, 1]) and `PCI_T_DS_L27` (a directional-specificity ratio, not raw complexity). `FCA_C3` and `FCA_T2` are secondary.
- **Unit.** One independent matched realization (arm) with its own numerical C_O values and both targets. The 4,096 donor masks of one gauge cube are not 4,096 realizations.
- **Fit.** Ridge regression, 5-fold group cross-validation (identical folds for all models and targets), pooled out-of-fold R², negative values reported.
- **Contrasts.** D0 = R²(M1_CO) − R²(M0); D4 = R²(M1_CO) − R²(MR_INFLUENCE4).
- **Decision.** Advantage on a target requires both lower bounds strictly positive at one-sided 98.75 % (paired group bootstrap on fixed out-of-fold predictions, 2,000 resamples; 5 % familywise over four comparisons). Joint advantage requires both targets. At least five independent groups are needed.
- **FCA values on 8-bit weights** (manuscript Table 17.9) are historical and unpaired with the bf16 C_O measurements; targets must be re-measured on the same bf16 realization.

## Status (2026-10-21)

| Item | Status |
|---|---|
| Γ, Δ, R, Λ₀ on the C_L24 contract | `NONZERO_CERTIFIED` each (see folder 18) |
| Graded functional closure, Qwen3-8B | `OPEN_NOT_ESTABLISHED`: no table pairs, per independent realization, numerical C_O values with the two targets and repeated cells |
| Superiority over the preregistered rivals | `NOT_TESTED`: D0 and D4 are not computable for either target |
| Closure relative to A† | Formal and not informative for realizations whose A† verdict does not vary (coarse grain: the target is unresolved) |
| Perturbation pilot (mlp0 attenuation, one family) | Exploratory, descriptive; it does not measure the change of Δ and does not test the C_O bridge. Files in `../18-co-qwen3-8b-c-l24/exploratory_t346/` |

A confirmed predictive advantage of `M1_CO` would not by itself imply functional closure: the direct pair test stays distinct. Four non-nullity labels do not replace four continuous predictors: the tests need numerical Γ, Δ, R and Λ for each independent realization.

## Reading order

1. `frozen/T341_FROZEN_GRADED_CLOSURE_SPEC.md`
2. `frozen/T343_FROZEN_PERFORMANCE_RIVALS_ABLATION_PROTOCOL.md`
3. `frozen/T344_RIVAL_PREREGISTRATION_LOCK.json`, then `frozen/T345_INFLUENCE4_EXTRACTION_LOCK.json`
