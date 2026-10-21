# FCI-3: Causal Construct Validity Protocol

**Status: frozen preregistration template (v0.1). Never executed against
real architecture pairs.** Included as a specification, not as a result —
this is the protocol that would need to be run to advance rung L5 of the
claim ladder (`../14-four-family-closure-and-compression-limit/docs/claim-ladder.md`)
from "open" to a verdict.

**Notation note:** the source document uses `C_F = [(Γ, Δ_prop, R(k))]` and
`𝒜_F = (a_flex, a_delay, a_update, a_status[, a_coord])`. In this
manuscript's notation, `Δ_prop → Δ`, and the profile actually under test
would be the closure-repaired `C_F^++` / `C_O` rather than the bare
three-scalar `C_F` — the intervention targets below are unchanged in kind.

## 1. Purpose

Test whether the compact causal profile is a **causal predictor** of
independently measured functional-access consequences — not whether the
access-consequence battery structurally identifies the profile (that
direction is separately and already refuted; see
`docs/intervention-identification-safeguards.md` and
`results/CF_AF_IDENTIFICATION_LIMITS_RESULT_v0.1.json`).

Target relation:

```
do(architecture) → ΔC_F → Δ𝒜_F
```

## 2. Independent consequence outcomes

Primary frozen families: `𝒜_F = (a_flex, a_delay, a_update, a_status)`. An
additional coordination endpoint `a_coord` may be preregistered before any
model results, defined purely behaviorally (correct joint use of information
distributed across multiple typed task channels) and may not inspect
internal causal variables. No `𝒜_F` score may use `Γ, Δ, R` directly.

## 3. Experimental unit

A matched architecture pair `(A_i, B_i)` sharing, as closely as technically
possible: training data, objective, parameter budget, external interface,
task semantics, inference budget, context length, memory capacity,
knowledge-access mechanism, and functional orientation. The intended
manipulation is architectural causal organization specifically.

## 4. Three intervention families

- **Z_R (recurrent self-conditioning).** Manipulate an internal latent
  return path while keeping external interaction fixed. Target: `R(k)`.
  Endpoints: `a_delay`, `a_update`.
- **Z_D (propagated differentiation).** Manipulate the number/independence
  of certified downstream causal roles reached by a content perturbation,
  without changing task semantics or stored content. Target: `Δ`. Endpoint:
  `a_flex`.
- **Z_G (integration).** Manipulate causal coupling across otherwise matched
  modules while preserving their local input/output capabilities. Target:
  `Γ`. Endpoint: `a_coord`, if independently preregistered and validated
  before use.

## 5. Manipulation checks

For intervention family `j`: require a preregistered minimum target
movement `Δc_j ≥ δ_j^min > 0`. Non-target profile components are reported,
never assumed constant. Rival variables (at minimum: `I_G`, `S`, memory
capacity, bandwidth, compute, raw task performance outside the access
battery) each carry preregistered equivalence margins `|D_r| ≤ ε_r`. Every
pair receives one of three statuses: `MANIPULATION-VALID`,
`MANIPULATION-INVALID`, `MANIPULATION-UNRESOLVED`. Primary inference uses
only `MANIPULATION-VALID` pairs; all statuses are reported regardless.

## 6. Primary directional hypotheses (not definitions)

- **H-R-DELAY.** For valid Z_R pairs: `ΔR > 0 ⟹ E[Δa_delay] > 0`.
- **H-R-UPDATE.** For valid Z_R pairs: `ΔR > 0 ⟹ E[Δa_update] > 0`.
- **H-D-FLEX.** For valid Z_D pairs: `ΔΔ > 0 ⟹ E[Δa_flex] > 0`.
- **H-G-COORD.** If `a_coord` is independently frozen before testing: for
  valid Z_G pairs, `ΔΓ > 0 ⟹ E[Δa_coord] > 0`.

Cross-effects (e.g. a Z_R intervention affecting `a_flex`) are secondary and
exploratory unless separately preregistered.

## 7. Dose-response analysis

Primary explanatory object: observed component change
`Δc = (ΔΓ, ΔΔ, ΔR(1), …)`. For outcome vector `Δa`, estimate a preregistered
local sensitivity model `Δa = J·Δc + B·Δr + ε`, where `Δr` contains measured
rival-variable changes. The Jacobian-like matrix `J` is empirical — not
assumed diagonal or universal across architectures.

## 8. Negative controls

- **NC-1 (knowledge-only):** increase accessible stored knowledge while
  keeping measured `C_F`/`C_O` within equivalence margins — tests
  discriminant validity against `I_G`/S.
- **NC-2 (compute-only):** increase compute budget without an intended
  causal-organization change.
- **NC-3 (memory-capacity-only):** increase passive storage capacity
  without adding an active recurrent self-conditioning path.
- **NC-4 (bandwidth-only):** increase external/passive channel bandwidth
  without an intended change in the internal causal profile.

Negative controls are not required to leave every `𝒜_F` endpoint unchanged;
they test whether the *predicted component-specific pattern* can be
explained by simpler rivals.

## 9. Failure criteria (bans post-hoc rescue)

A component-link hypothesis is rejected for the preregistered architecture
class if **all** of: (1) the targeted manipulation is repeatedly
`MANIPULATION-VALID`; (2) the primary endpoint is measured with adequate
preregistered precision; (3) the pooled causal effect falls inside a
preregistered negligible-effect band or has the wrong sign; (4) the result
replicates across the required number of independent seeds/pairs. No
post-hoc appeal to an unspecified nonlinear downstream readout may rescue a
failed hypothesis; a new readout requires a new frozen protocol before new
data are examined.

## 10. Profile-level failure criterion

The access interpretation of the compact profile is rejected for the tested
architecture class if substantial, replicated manipulations spanning all
three coordinates produce no preregistered access-consequence pattern
beyond rival variables.

## 11. Positive construct-validity criterion

Requires all of: successful prospective manipulation of the profile;
preregistered directional prediction of independent `𝒜_F` outcomes;
replication across seeds and at least two architecture scales; discriminant
performance against negative controls; held-out task variants generated
*after* model freeze; and no dependence of `𝒜_F` scoring on profile
internals. A pass supports "the profile is a causal predictor of the frozen
functional-access consequence profile in the tested architecture class" —
**it does not establish phenomenal consciousness.**

## 12. Stronger future target (FCI-4)

Cross-system construct validation would additionally require transport of
causal roles, intervention families, resource budgets, `𝒜_F` task
semantics, and rival-variable controls across architecture families — left
for future work.
