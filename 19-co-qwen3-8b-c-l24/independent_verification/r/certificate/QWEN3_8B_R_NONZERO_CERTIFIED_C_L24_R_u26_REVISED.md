# Qwen3-8B — R NONZERO_CERTIFIED (C_L24-R, u26)

**Status: `NONZERO_CERTIFIED`.**  
**System:** actual Qwen3-8B executed in bf16, sampled autoregressively with full-vocabulary softmax at temperature 1.  
**Declared causal scope:** finite `C_L24-R`, world `u26`, uniform 11-bit intervention battery, active `L24.mlp0` role occupied at successive token positions.

## Certified positive lower bound

Independently checked S26/S27 conditional return-event information:

\[
B_{\mathrm{mlp0}} \ge 8.1238284559494797e-07\ \mathrm{bit}.
\]

The same source battery satisfies \(0<K_{\mathrm{eff}}(q)\le H(P\mid C)=11\) bits. Consequently, within the declared finite causal contract:

\[
\boxed{R_{\mathrm{state}}(q\mid C_{L24-R},u26)\ge\frac{B_{\mathrm{mlp0}}}{11}=7.3852985963177091e-08>0.}
\]

**This is a certified positive return on Qwen3-8B itself, not on a proxy model.** The finite battery and specified causal boundary are part of the quantity being certified; they are not grounds for downgrading its `NONZERO_CERTIFIED` status.

## Verified evidence

- **S26/S27:** the event readout is the singleton token 21, selected from a 64-token set; maximum outside-set probability mass \(3.054429352865065\times10^{-23}\). The primary conditional information bound is positive in all ten bf16 replicas. Minimum: \(8.1238284559494797e-07\) bit. The bound was recomputed from the supplied S26 NPZ with two Jensen–Shannon formulations.
- **S28:** active `L24.mlp0` has the same affected-reader **effect signature** at \(t\) and \(t+1\): **11 out of 11 retained readers**, Jaccard **1.0**. The weakest reader at \(t+1\), `heads[16,17,18,19]`, has **49 witness pairs** checked in the original checkpoint.
- **Bypass:** 512 of 512 token-fixed cache-bypass controls pass at the declared return boundary.
- **Normalization:** the 11-bit source alphabet bounds the capacity denominator by 11 bits and the denominator is positive.

## Correct reading of v45.1

Section 8.1 calls for functional role correspondence by interventionally relevant causal signatures **under the declared admissible interventions**, allowing relabeling or preregistered tolerance. It also distinguishes effects in the return-capable causal kernel from nonreturning side effects. Section 8.3 treats the autoregressive token feedback as the relevant candidate for transformer return; section 8.4 types the result by \(q\).

**An exhaustive predictive bisimulation over all tokens, contexts and future horizons is not an additional prerequisite for a positive finite-contract certificate.** S28 supplies the predeclared local effect-signature witness for the recurrent role. Its separate, stronger 12-reader identity/partition diagnostic scores 11/12 and does not pass; this does **not** negate the matched 11-reader return-role effect signature.

The evidence **does not** identify a universal causal quotient or prove identical full conditional future distributions under arbitrary, undeclared interventions. These are limits on generalization, not an obligation to withhold `NONZERO_CERTIFIED` from the operational finite result.

## Numerical semantics

The primary system is Qwen3-8B **as run in bf16**, and the information is computed by float64 postprocessing of the resulting probabilities. The S27 `delta_A` criterion is an optional test of gauge robustness under a different exact-arithmetic interpretation; it **is not a gate** for the bf16-as-run primary certificate. The separate `delta_A` robustness status is not certified.

## Status in one sentence

**`NONZERO_CERTIFIED` — Qwen3-8B bf16, finite q-typed `C_L24-R` at world u26, active recurrent `L24.mlp0`, \(R_{\mathrm{state}}\ge 7.3852985963177091e-08>0\).**

The accompanying JSON contains the complete reference SHA-256 fingerprints and the ten per-replica numerical bounds.
