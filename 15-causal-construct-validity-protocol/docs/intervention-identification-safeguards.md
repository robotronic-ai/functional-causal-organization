# Intervention Identification Safeguards for FCI-3

**Status: executed against synthetic adjudicator cases only.** Validates the
*protocol logic*, not construct validity of any causal profile. Result file:
`results/FCI3_INTERVENTION_IDENTIFICATION_RESULT_v0.2.json`; harness:
`scripts/fci3_intervention_identification_harness_v0_2.py`.

## 1. Why FCI-3 alone is not enough

Running the FCI-3 protocol (`fci3-protocol-specification.md`) and observing
both `do(architecture) → ΔC_F` and `do(architecture) → Δ𝒜_F` does **not**
establish `C_F → 𝒜_F`: the intervention may act through a rival
architectural pathway that happens to move both the profile and the
consequence without either causing the other. This document specifies the
additional safeguards needed to close that gap.

## 2. Stronger identification target

Rather than a bare before/after comparison, the identification target is a
profile-response relation under matched rivals:

```
m(c, r) = E[a | do(c), r]
```

i.e. the expected access-consequence vector as a function of the causal
profile, holding rival variables `r` fixed.

## 3. Primary empirical sufficiency stress test: same-profile invariance

```
c(e₁) ≈ c(e₂),  r(e₁) ≈ r(e₂)  ⟹  a(e₁) ≈ a(e₂)
```

A replicated violation of this returns:

```
FIBER-VIOLATION
```

— direct evidence that the compact profile is incomplete in the tested
domain (a fresh instance of the Compression Limit Theorem's logic, folder
`14`).

## 4. Convergent interventions

At least two mechanistically distinct interventions should target the same
profile component. If they produce equivalent profile changes but
incompatible access-consequence effects, return:

```
INSTRUMENT-DISAGREEMENT
```

This distinguishes genuine profile incompleteness from a single
implementation-specific side effect of one particular intervention.

## 5. Exclusion

An architectural manipulation is never assumed to satisfy the exclusion
restriction merely because compute, memory, and other known rivals were
matched. Direct-effect sentinels are used as falsification aids; a sentinel
failure returns:

```
EXCLUSION-UNSUPPORTED
```

## 6. Synthetic adjudicator validation (what was actually run)

Five frozen cases were used to validate the adjudication logic itself:
a valid convergent intervention; a direct-effect contamination case; a
rival-mismatch case; a same-profile access-fiber violation case; and an
unresolved profile-equivalence-interval case. All five were classified as
preregistered.

**Verdict: `FCI3-IDENTIFICATION-ADJUDICATOR-PASS`.**

This confirms the adjudication rules behave as specified on constructed
cases. **It is not evidence for the construct validity of any particular
causal profile** — no real architecture pair has been run through this
adjudication logic yet.

## 7. Manuscript consequence

The empirical bridge should never be stated merely as
`do(architecture) → ΔC_O → Δ𝒜_F`. It should state that construct validity
additionally requires: (1) successful manipulation of the causal profile;
(2) rival-variable equivalence; (3) same-profile outcome invariance across
distinct implementations; (4) convergent component-response effects; (5)
prospectively frozen failure criteria. Together with
`fci3-protocol-specification.md`, this converts the profile-to-access
relation into a genuinely falsifiable causal identification program, rather
than a single-shot before/after comparison.

## 8. Where the necessity-direction evidence for C_O's components comes from

A separate, already-executed result bears on why `Γ`, `Δ`, and `R(k)`
specifically were chosen as intervention targets in §4 of the protocol,
rather than being a purely engineering choice. See
`functional-access-necessity-bridge.md` for the independent construct
`A(e) = [K_A]_semantic` and its Breadth/Temporal/Unity necessity theorems,
which motivate — without presupposing — the specific component targets
`Z_D → Δ`, `Z_R → R(k)`, `Z_G → Γ` used above.
