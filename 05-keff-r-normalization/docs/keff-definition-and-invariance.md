# K_eff definition and invariance

## What this campaign forbids

The protocol opens by ruling out a list of substrate-specific quantities as candidate
definitions of `K_eff`, precisely because none of them is axiomatically tied to the
information-theoretic re-entry numerator `B_reentry`:

- residual-stream width,
- neuron count,
- nominal KV-cache size,
- covariance rank,
- participation ratio,
- floating-point precision.

These may be *descriptive*, but using any of them as the denominator of `R` would make `R`
depend on incidental implementation choices rather than on causal structure — and would make
`R` incomparable across systems with different nominal state dimensions.

## The causal quotient of the source battery

Let `Q` be the label of a source intervention under context `U`. Two labels `q, q'` are
declared equivalent iff, in the source causal quotient declared *before* the return channel is
examined, they realize the same effective causal-interventional state:

\[
q \sim_C q'.
\]

Write `C = χ(Q)` for the resulting causal-equivalence class. The quotient is declared
**without looking at the return channel** used to compute `B_reentry` — this ordering is what
prevents the normalization from being circular (choosing the quotient so as to make `R` come
out a particular way).

## Definition

\[
\boxed{K_{\rm eff}(q) = H_q(C \mid U)}
\]

the conditional Shannon entropy, under the source-battery distribution `q`, of the causal
quotient `C` given the context `U`. The matching numerator, defined in the same neutralized/
cut model as the causal quotient:

\[
\boxed{B_{\rm reentry,q} = I_q(C; Y^{ret} \mid U)}.
\]

## Axioms

- **K0 — same battery.** The same `q` and `U` are used in both the numerator and the
  denominator.
- **K1 — source-side only.** `K_eff` depends on the source battery and its causal quotient, not
  on the return channel.
- **K2 — invariance.** `K_eff` is invariant under bijective recoding of intervention labels,
  latent states, or causal classes.
- **K3 — causal duplication.** Duplicating an intervention label without creating a new causal
  class, and while preserving the total probability mass of that class, does not change
  `K_eff`.
- **K4 — informational bound.** Always `0 ≤ B_reentry ≤ K_eff` (proved in
  `r-normalization-theorem.md`).
- **K5 — perfect calibration.** If `Y^ret` bijectively determines `C` given `U`, then `R = 1`.
- **K6 — absence of return.** If `C ⟂ Y^ret | U`, then `R = 0`.
- **K7 — sensitivity to coverage.** At fixed causal quotient, making `q(C|U)` more
  concentrated can *reduce* `K_eff`. This is intentional: `K_eff` measures the capacity
  actually excited by the declared battery, not the substrate's maximal nominal capacity.

## Why this makes K_eff substrate-independent

The definition never asks for the differential entropy of a continuous activation vector; it
asks for the discrete entropy of the **causal quotient of the intervention battery**. Two
continuous representations related by a bijection can therefore have exactly the same `K_eff`
— and, per axiom K3, subdividing labels within one causal class (e.g. by measuring at higher
nominal resolution) does not inflate `K_eff` either. This is the property that the K7 control
in the finite audit (`docs/finite-audit-results.md`) checks directly: two substrates with
different nominal dimensions but the same causal quotient must produce identical `K_eff` and
identical `R`.

## Scope

Fixing this normalization does not by itself measure anything on a real transformer. The
empirical campaign (folder `06-transformer-measurement/`, out of scope here) still has to
identify (1) a source intervention algebra, (2) the causal quotient `C`, (3) a return channel
`Y^ret` consistent with the neutralized/cut model, and (4) `I(C;Y^ret|U)`. Activation patching
can supply the intervention *mechanism*, but does not by itself define the causal quotient or
the battery.
