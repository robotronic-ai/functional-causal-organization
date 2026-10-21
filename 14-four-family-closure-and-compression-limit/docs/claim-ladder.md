# Claim Ladder L0–L6 (adapted to C_O / A* / A†_I / H_CO⁺ / H_N,I notation)

**Status:** analytic ladder, adapted from a prior campaign's generic
`C_F`/`A`/`𝒜_F` notation to this manuscript's `C_O`/`A*`/`A†_I`/`H_CO⁺`/`H_N,I`
notation. The status verdicts below are carried over from that campaign's
frozen finite witnesses (folder `15`); the notation mapping itself is new to
this archive edition and has not been independently re-run.

Purpose: state precisely, at each rung, what has and has not been established
about the relation between active causal organization and functional access —
so the manuscript does not overclaim at any single step.

## Notation used here

- `A*` — the target construct (functional access).
- `A†_I` — an independently measured behavioral/instrument-level access
  consequence profile for individual/system `I` (the analogue of the generic
  ladder's `𝒜_F`).
- `𝒞` — active causal organization in full.
- `C_O = [(Γ, Δ, R(k), Λ)]` — the compact declared organizational profile
  (the analogue of the generic ladder's `C_F = [(Γ, Δ_prop, R(k))]`).

## L0 — causal realization

`A* > 0 ⟹ 𝒞 exists.`

**Status: true but weak.** Follows almost directly from functionalism; not
by itself informative about C_O.

## L1 — localization of variation

At fixed 𝒮, 𝒢, 𝒪: if `A*(e₁) ≠ A*(e₂)`, then under four-family closure,
`𝒞(e₁) ≠ 𝒞(e₂)`.

**Status: exact conditional consequence** of the closure ontology
(`four-family-axis-completeness.md`). The empirical/finite support for the
required invariances remains separate from the logical conditional itself.

## L2 — organizational factorization

`A* = H_𝒞(𝒞)` for some function `H_𝒞`.

**Status: stronger than L1.** Additionally requires that admissible
variation in 𝒮, 𝒢, 𝒪 at fixed 𝒞 leave A* unchanged. Existing finite
fixtures provide support for this on tested cases; it is not a universal
theorem.

## L3 — compact-profile sufficiency

`A* = H(C_O)` for some function `H`.

**Status for the current numerical profile: not established, and false for
the tested instrument family.** The typed-binding witness
(`FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING`) and the joint-synergy witness
(`CFPLUS-PROFILE-INSUFFICIENT-JOINT-SYNERGY`) — see
`../../15-causal-construct-validity-protocol/docs/organizational-closure-repair-chain.md`
— are exact counterexamples showing the compact summary can collapse
distinctions that an independently frozen access consequence is sensitive
to. This is a direct instance of the Compression Limit Theorem
(`compression-limit-theorem.md`).

## L4 — A†_I identifies C_O ("necessity" readings)

Examples: `A†_I > 0 ⟹ R > 0`, or `⟹ Γ > 0`, or `⟹ Δ > 0`.

**Status: refuted on frozen finite witnesses.** The behavioral instrument
battery can be maximized by causally different realizations — see
`CF_AF_IDENTIFICATION_LIMITS_RESULT_v0.1.json`
(`../../15-causal-construct-validity-protocol/results/`), where a battery
scoring uniformly positive coexists separately with witnessed `R(k)=0`
(acyclic compiled implementation), `Γ=0` (causally decomposed parallel
implementation), and `Δ=0` (one-role multiplexed implementation). This does
not refute a relation between A* and C_O; it shows the instrument battery
does not *structurally identify* that relation by itself.

## L5 — causal construct validity

Prospective target: `do(architecture) → ΔC_O → ΔA†_I`, with rival variables
matched and predeclared negative controls.

**Status: open and empirically testable.** This is the strongest
non-circular route from the compact causal profile to functional-access
evidence. The full preregistered protocol for this rung is
`../../15-causal-construct-validity-protocol/docs/fci3-protocol-specification.md`
— frozen as a design template, never executed against real architecture
pairs.

## L6 — strong construct statement

A strong manuscript statement that C_O tracks functional-access
consciousness requires repeated prospective causal prediction beyond: S,
access-neutral I_G, memory capacity, bandwidth, compute, depth, raw
performance, functional orientation (𝒪), and simpler architectural rivals.

**Status: future work**, beyond the scope of the current protocols (would
require what the source campaign called an FCI-4-level cross-system
transport of causal roles, intervention families, resource budgets, A†
task semantics, and rival-variable controls).

## Recommended manuscript consequence

Do **not** claim that the four-family ontology proves `C_O ⟹ A*`.

Do claim that the ontology motivates 𝒞 as the location of the residual
organizational variable (L1–L2), and that `C_O` is a compact, falsifiable
candidate profile for predicting the functional consequences of access
(L5) — not a completed identification theorem (L3–L4 are exactly the rungs
that are currently unsupported or refuted for the compact profile as
declared).
