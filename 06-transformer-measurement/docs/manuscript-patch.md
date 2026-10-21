# Proposed manuscript patch (point 4)

Source: `POINT4_MANUSCRIPT_PATCH_v0_1.md`. This is the exact diff proposed for the
manuscript's §18/§12.5 discussion of the transformer measurement, translated verbatim.

## 1. Replace the operational definition of K_eff

```
K_eff(q) = H_q(C|U)
```

where `C` is the source battery's effective causal quotient, defined independently of
the return channel. Then:

```
B_reentry,q = I_q(C; Y_ret | U)
R_q = I_q(C; Y_ret | U) / H_q(C|U)     for H_q(C|U) > 0, else R = NA
```

Add: `0 ≤ R ≤ 1`, by the conditional-mutual-information identity.

## 2. Primary recurrence profile

For consistency with the Δ construction, report

```
κ_R = H(C|U) / log2(|C*|)
R   = I(C; Y_ret | U) / H(C|U)
```

The primary profile becomes `(κ_R, R)`. The support-normalized index
`R_cap = κ_R · R` remains secondary.

## 3. First transformer campaign

Add as an empirical example, not as a universal calibration:

> On Pythia-70M-deduped, layer 2 / position 1, eight donor interventions, and the
> ordered top-2 source quotient, a candidate-fixed holdout reproduces the seven-class
> partition and measures a positive but weak re-entry.

Holdout v0.5 value: `R ≈ 0.004954`. State explicitly that this value is conditional on
the entire instrumentation.

## 4. Multi-round loop

Introduce the experimentally closed loop separately:

```
C_t → Y_t^ret → C_{t+1}
```

with declared transport `y_i ↦ q_i ↦ χ(q_i)`. Report the profile
`(R_1, R_2, R_4, R_8, R_16)`, not a single scalar. For campaign v0.6.1:

```
R_1 ≈ 3.96×10⁻³
R_2 ≈ 1.80×10⁻⁵
R_4 ≈ 5.09×10⁻¹⁰
```

then numerical zero.

## 5. Mandatory caveat

Add explicitly:

> This loop is closed experimentally by a declared external transport. It does not
> constitute a demonstration of an autonomous internal recurrent loop of the
> transformer.

## 6. Claim-status vocabulary

Use:

- `analytic theorem` for the bound `0 ≤ R ≤ 1`;
- `candidate-fixed finite/empirical non-refutation` for the top-2 quotient;
- `empirical measurement under frozen instrumentation` for the R values;
- `external closed-loop retention profile` for v0.6.1.

Avoid any phrasing suggesting an intrinsic or universal measurement of the model.
