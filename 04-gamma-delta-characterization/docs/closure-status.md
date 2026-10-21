# Closure status: CLOSE-CONDITIONAL

## Final status

\[
\boxed{\text{CLOSE-CONDITIONAL}}
\]

This is the status recorded in the campaign's true final closure note. It confirms that
independent reproduction succeeded on every required piece of evidence:

- the manifest and hashes are complete and reproduce;
- the analytic Γ/Δ audits (axiom non-uniqueness, marginal-independence uniqueness, Δ
  decomposition, independence witnesses) reproduce;
- the GD-SEM prospective campaign reproduces at **32/32**;
- the GD-SEM candidate-fixed holdout, run against the **same unmodified candidate**,
  reproduces at **32/32**.

An earlier closure note in this same work stream recorded the status
`PENDING INDEPENDENT REPRODUCTION` and explicitly stated that promotion to
`CLOSE-CONDITIONAL` was authorized only *after* independent reproduction of both the
prospective run and the holdout. That note is superseded by the result above — the
independent reproduction it was waiting on is exactly what happened, with matching 32/32
results in both runs.

## Semantic choices retained

1. Keep Γ as the arithmetic mean, under the `G-WEAK` additive interpretation.
2. Report the directional balance β separately (for the minimizing cut, or the full set of
   minimizing β values when several cuts tie).
3. Make `(κ_q, δ_q)` the primary reported profile for Δ.
4. Keep `Δ = κ_q·δ_q` as a historical scalar synthesis.
5. Provisionally keep `C_F = (Γ·Δ·R)^{1/3}` as a secondary scalar synthesis, with the richer
   primary profile `(Γ, {β_π*}, κ_q, δ_q, R)` reported ahead of it.

## Claim status, as closed

- **Γ**: characterized *conditionally* on the explicit additive axiom of independent
  directional contributions (marginal independence); finite non-refutation on both the
  prospective and candidate-fixed-holdout finite domains.
- **Δ**: exact analytic decomposition `Δ = κ_q·δ_q`; the support-only (D-CAP) and q-adaptive
  (D-DEC) normalizations are each characterized within their own MI-linear function class;
  finite non-refutation on both domains.
- **Independence**: exact finite witnesses of non-dependence in both directions
  (`Γ↛Δ`, `Δ↛Γ`) on the witness domain.

## Manuscript patch (normative text)

The companion patch note lays out the exact text changes the closure authorizes. Summary:

1. **Γ — normative text.** Replace any wording implying Γ requires strictly bidirectional
   influence with: Γ measures the additive mean of the two normalized directional
   interventional influences of a cut; a strictly one-directional influence therefore
   contributes positively. This corresponds to the axiom that each direction's marginal
   contribution is independent of the strength already present in the other direction.
   Immediately report `β_π = 2·min(x_π,y_π)/(x_π+y_π)` alongside Γ for the minimizing cut(s),
   noting that Γ alone does not carry directional-balance information, and that ties among
   minimizing cuts are reported as the full set of their β values, with no tie-break.

2. **Γ — axiomatic characterization proposition.** Add: for `F:[0,1]²→[0,1]` satisfying
   nullity, symmetry, diagonal calibration `F(t,t)=t`, and marginal independence
   (`F(x₂,y₁)−F(x₁,y₁) = F(x₂,y₂)−F(x₁,y₂)`), `F(x,y)=(x+y)/2` follows. Note explicitly that
   marginal independence is substantive: it excludes synergy/redundancy between directions at
   the aggregator level.

3. **Δ — primary profile.** Keep `Δ ≡ Δ_cap = I(P;Y)/log₂|P*|` for compatibility, but
   introduce `κ_q = H(P)/log₂|P*|` and `δ_q = I(P;Y)/H(P)` as the primary reported profile,
   with `Δ = κ_q·δ_q`. Recommended terminology: `κ_q` = entropic battery coverage; `δ_q` =
   conditional propagated discriminability; `Δ` = coverage-weighted propagated
   differentiation, a synthetic index.

4. **Uniform-battery case.** If `q` is uniform on the effective support, `H(P)=log₂|P*|`, so
   `κ_q=1` and `Δ=δ_q`.

5. **Non-uniform-battery warning.** If the battery is non-uniform, perfect decoding gives
   `δ_q=1` but not necessarily `Δ=1`. A reduced `Δ` in that case reflects incomplete/non-uniform
   battery coverage, not reduced system discriminability.

6. **Relation to `C_F`.** Keep `C_F = (Γ·Δ·R)^{1/3}` provisionally, but note that Δ's coverage
   factor means any `C_F` comparison requires a comparable battery `q`, effective support, and
   instrumentation. The minimum profile to report ahead of the scalar becomes
   `(Γ, {β_π*}, κ_q, δ_q, R)`; `C_F` remains a secondary conventional synthesis.

7. **Γ/Δ independence.** State the exact finite witnesses: `Γ(C1)=Γ(C2), Δ(C1)≠Δ(C2)` and
   `Δ(C3)=Δ(C4), Γ(C3)≠Γ(C4)` — on the witness domain, neither Γ nor Δ reconstructs as a
   universal function of the other.

8. **Recommended claim-status language.** For Γ: "formula characterized conditionally on the
   additive axiom of independent directional contributions; finite non-refutation, prospective
   and candidate-fixed holdout, on a finite domain." For Δ: "exact analytic decomposition
   `Δ=κδ`; support-only normalization characterized in the MI-linear class; finite
   non-refutation, prospective and candidate-fixed holdout." For independence: "exact finite
   witnesses of non-dependence in both directions."

## What comes next

With this work stream at `CLOSE-CONDITIONAL`, the next authorized step in the manuscript's
order of battle is the empirical measurement of `K_eff` and `R` on a transformer / latent
loop — see `05-keff-r-normalization/` for the normalization that measurement depends on, and
`06-transformer-measurement/` (out of scope for this folder) for the empirical run itself.
