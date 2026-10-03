# PCI_T v0.21: Directional Specificity in Qwen3-8B

## Reviewer summary

PCI_T was originally introduced as an internal perturbational complexity measure for transformers. The historical v0.20 protocol produced a reproducible Qwen3-8B score of 277.9609375, but adversarial testing later showed that absolute perturbational complexity alone is not sufficiently discriminant: random linear and chaotic uncoupled systems can also produce high raw PCI_T values.

The final v0.21 analysis therefore asks a more specific question: **does a designated intervention direction produce a richer downstream perturbational response than norm-matched orthogonal directions within the same system?** This within-system comparison is the directional-specificity derivative of PCI_T.

## Adversarial validation

The frozen synthetic benchmark compares one designated direction with 15 orthogonal RMS-matched control directions across 64 seeds. Two adversarial systems represent generic dynamical richness without a planted selective causal direction. Three positive controls contain explicitly directionally structured propagation.

| System family | Median designated PCI_T | Median control PCI_T | Median directional enrichment | Median win rate |
|---|---:|---:|---:|---:|
| Untrained random linear | 217.515625 | 219.765625 | 0.995383 | 0.400000 |
| Chaotic uncoupled | 155.312500 | 157.187500 | 1.000495 | 0.533333 |
| Feed-forward directionally structured | 107.250000 | 12.906250 | 8.398058 | 1.000000 |
| Branching directionally structured | 22.625000 | 2.687500 | 8.418605 | 1.000000 |
| Recurrent directionally structured | 98.796875 | 13.968750 | 7.077322 | 1.000000 |

All preregistered gates passed. The minimum structured enrichment divided by the maximum adversarial enrichment was 7.073821.

The important point is that the discriminator is **not the absolute PCI_T magnitude**. The random and chaotic systems can have large absolute scores. They fail to show a privileged intervention direction. The structured systems show a large within-system enrichment for the designated direction. The feed-forward positive control also separates, so the result should not be interpreted as a recurrence-specific effect.

## Prospective Qwen3-8B replication

A development analysis found that the directional effect was localized most consistently at source layer index 27. This localization was treated only as a development finding. The layer, token fractions, perturbation amplitude, direction construction, control construction, aggregation, and CAL3 gates were frozen before the independent CAL3 holdout was opened.

CAL3 passed all frozen gates:

- deepest-layer dominance in 5 of 6 episodes;
- L27 system enrichment = 1.329707064466;
- L27 fraction-0.20 median = 1.432915563202;
- L27 fraction-0.40 median = 1.277131071141;
- late-versus-lower-layer system ratio = 1.314874989059.

This pass authorized the already-frozen SCORE2 measurement. SCORE2 was opened only after the final X8 contract had been frozen, and no SCORE2 selection gate or post-SCORE2 tuning was used.

## Final SCORE2 result

The frozen primary scalar is `PCI_T_DS_L27`. For each episode, PCI_T is measured at source layer index 27 for two frozen token fractions. At each site, the endogenous residual direction is compared with 15 orthogonal RMS-matched controls. Site enrichment is aggregated across the two fractions and then across episodes by medians.

The final result is:

**PCI_T_DS_L27 = 1.381374456016**

95% episode-bootstrap interval, 5,000 replicates, seed 11011:

**[1.200589200476, 1.609210775101]**

Episode values were 1.433444, 1.190693, 1.329305, 1.481924, 1.210486, and 1.736497. The frozen fraction medians were 1.364170 for fraction 0.20 and 1.267708 for fraction 0.40.

The secondary absolute source-direction measure is:

**PCI_T_ST_L27_SOURCE = 343.9609375**

with 95% episode-bootstrap interval:

**[300.640625, 380.2578125]**

## What the final scalar means

`PCI_T_DS_L27 = 1.381` means that, under the frozen protocol, the endogenous residual direction at layer index 27 produces approximately 38% more state-transition perturbational complexity than the median of 15 orthogonal RMS-matched control directions.

The intervention modifies an activation state, not the model parameters. The model weights remain fixed. Layer index 27 is the source-layer index used by the code; under zero-based indexing it corresponds to the 28th transformer block, with downstream observations taken from later layers.

The secondary value 343.96 is not a replacement for, or a direct increase from, the historical v0.20 score of 277.96. The two values use different panels and aggregations. The primary interpretable comparison in v0.21 is the within-site directional ratio, not the cross-protocol difference between 343.96 and 277.96.

## Scientific scope

The synthetic adversarial result supports directional specificity as a way to distinguish selective, directionally organized propagation from generic random or chaotic perturbational richness. The independent CAL3 replication and final SCORE2 measurement show that a corresponding late-layer directional specificity is reproducibly present in Qwen3-8B under the frozen protocol.

This is an independent perturbational result. It does not by itself establish that C_O has incremental validity, that the privileged direction is task-useful, that recurrence is the causal source of the effect, or that the measurement indexes consciousness. Those questions require paired prospective comparisons against C_O, A-dagger, or independently defined functional targets.
