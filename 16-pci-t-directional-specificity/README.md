# 16 — PCI_T directional specificity (Qwen3-8B)

PCI_T transposes the logic of the Perturbational Complexity Index to transformers: perturb an internal state, let the perturbation propagate causally, and measure the complexity of the downstream response. It is constructed independently of C_O and of the behavioral battery A†, so that it can serve as an external variable for both.

This folder contains the final v0.21 reviewer package. It supports manuscript **§17.8** (and the PCI_T row of Table 1, §2).

## Results in one paragraph

On a synthetic adversarial benchmark (64 seeds per family), absolute perturbational complexity is high for untrained random-linear and uncoupled chaotic systems as well as for directionally structured ones. What separates them is **directional specificity**: the ratio between the complexity evoked by a designated intervention direction and the median over 15 orthogonal, norm-matched control directions. Adversarial families stay at ≈ 1.0 (0.995 and 1.000); feed-forward, branching and recurrent structured systems reach 8.40, 8.42 and 7.08 (separation ratio 7.07; all preregistered gates pass). On Qwen3-8B, a late-layer localization (source block index 27) found on development data was frozen, prospectively replicated on an independent holdout (CAL3: L27 dominant in 5/6 episodes, enrichment 1.33), and then measured once under the frozen contract:

| Quantity | Value | 95 % bootstrap CI |
|---|---:|---|
| `PCI_T_DS_L27` (primary, ratio) | **1.381** | [1.201, 1.609] |
| `PCI_T_ST_L27_SOURCE` (secondary, absolute) | 343.96 | [300.64, 380.26] |
| `PCI_T_ST_ADAPTED` (v0.20 panel, retained) | 277.96 | [238.82, 309.29] |

Model weights stay frozen; interventions act on residual-stream activations. The two absolute scores use different panels and are read separately, not as a before/after pair.

## Contents

`PCI_T_v0.21_FINAL_REVIEWER_PACKAGE/` is byte-identical to the frozen package (SHA-256 of the original archive: `6dbcac7a134cecba11d56364f6117682cd148e69f48e49d8587c9b4f2bf5c9d2`). Its own `README.md` and `PCI_T_V021_REVIEWER_SUMMARY.md` give the full method and interpretation boundaries.

## Reproduce

```bash
cd PCI_T_v0.21_FINAL_REVIEWER_PACKAGE
python verify_pci_t_v021_final.py
```

Exit status 0; prints `PCI_T_DS_L27=1.381374456016` and `PCI_T_ST_L27_SOURCE=343.960937500000`. Standard library only.

## Scope

The result establishes a reproducible, prospectively confirmed late-layer directional specificity of perturbational complexity in Qwen3-8B under the frozen protocol. No consciousness scale is attached to these values. Their role in the programme is that of an external variable to be confronted prospectively with variations of C_O under paired architectural interventions (manuscript §17.6).
