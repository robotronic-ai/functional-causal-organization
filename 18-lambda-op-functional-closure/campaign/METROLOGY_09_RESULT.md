# Metrology 09 Result

## Verdict

`REPEATABILITY_CLEAN`

Metrology 09 found no repeat-run variability in the tested Qwen3-8B CUDA/bfloat16 stack for the six sealed systems.

- repeated systems: 6
- independent process/model reloads per system: 3
- behavioral target-vector stability: 1.0000
- individual cell-outcome stability: 1.0000
- selected descriptor stability: 1.0000
- exact equality fraction across continuous internal measurements: 1.0000
- p95 within-system noise / between-system signal: 0.0000
- p95 relative internal range across repeats: 0.0000

The result is an engineering repeatability result, not a theorem. It does not estimate systematic bias relative to a float32 model execution.

## Compression-fidelity audit

The rich reference object was the 288-dimensional target-blind cross-target log trajectory from Cross-Target 07.

| Summary | Dimension | Pairwise-distance Spearman | kNN-3 overlap | LOOCV rich reconstruction R2 |
|---|---:|---:|---:|---:|
| `GLOBAL_LOG_MEAN_1D` | 1 | 0.7202 | 0.2778 | 0.2629 |
| `GLOBAL_LOG_MEAN_STD_2D` | 2 | 0.5956 | 0.4167 | 0.3173 |
| `OPERATION_MOMENTS_8D` | 8 | 0.7218 | 0.3750 | 0.2365 |
| `OP_SOURCE_MOMENTS_24D` | 24 | 0.7456 | 0.5278 | 0.2480 |
| `DIAG_OFFDIAG_TEMPORAL_48D` | 48 | 0.8882 | 0.6111 | 0.3930 |

The 48-dimensional temporal summary is the strongest tested continuous handcrafted summary, but it still reconstructs only 39.30% of standardized rich variation under the declared leave-one-out diagnostic.

The categorical topology quotient `ALL_FLOW_H2_FORWARD_CATEGORICAL` is strongly compressive but very low fidelity:

- fibers: 6
- nontrivial coverage: 83.33%
- pairwise-distance Spearman versus the rich proxy: -0.0267
- rich variance explained by fibers: 24.76%
- within-fiber rich distortion: 75.24%

## Interpretation

The current closure failures should not be attributed to repeat-run stochastic noise in the tested execution stack. The larger metrological problem is information loss introduced by the numerical summaries and quotients applied to the rich Lambda/active-kernel proxy.

This audit does not establish the fidelity of a scalar `C_O`. Complete `C_O` is not a default scalar, and the Cross-Target 07 panel does not contain certified Gamma, Delta, or R measurements. The audit applies only to the current cross-target Lambda/active-kernel proxy.

## Campaign consequence

Further closure search is now gated by representation fidelity. A quotient must be selected without the behavioral target and must pass a declared fidelity threshold relative to the 288-dimensional proxy before behavioral fiber homogeneity is inspected.
