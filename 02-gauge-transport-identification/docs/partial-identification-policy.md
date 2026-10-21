# FIBER-PARTIAL-ID: a formal policy for reporting an unidentified transport honestly

## Why a policy, not another candidate selector

`Protocole_FIBER_PARTIAL_ID_v0_1.md` is written explicitly *after* the failure of `JID-VERT` and
`JID-CONJ` (two earlier, abandoned attempts to derive the intervention lift `J` from vertical-
monoid conjugacy or equivariance — not shipped in this edition; the source repository keeps them
only "for genealogy"). Rather than trying another selection rule, it formalizes the distinction
between **point identification**, **partial identification**, and **non-identification /
inconsistency** of the surgery, given an enriched causal context `𝔠=(𝓜,q,𝓡)`.

## Failed simple fixes, recorded before the policy (`FIBER_STATUS_POST_AUT_v0_1.md`)

- **Elementary invariants don't select the oracle.** On P4-A and P4-B, every competing section
  shares exactly the same rank, fixed-point count, kernel signature, Green `L/R/J` ideal sizes,
  centralizer size, and commutation with the vertical submonoid. None of these standard
  semigroup-theoretic invariants distinguishes the oracle gauge from the others; the sections live
  in genuinely different automorphism orbits, so their difference lies in more global relational
  structure of the extension.
- **Naive gauge-free extrema both fail as a general rule.** Enumerating every orthogonal
  trivialization of a fixed `P`:

  | Control | `𝓑` (all gauge scores) | Oracle | Oracle = |
  |---|---|---:|---|
  | IR10 | `{0.5, 1}` | 1 | max |
  | P3-D | `{1, 1.125, 1.25, 1.5}` | 1.5 | max |
  | P4-A | `{0.5, 1}` | 0.5 | min |
  | P4-B | 8 values in `[0.392657719481…, 1]` | min | min |
  | P4-C | 8 values in `[0.379430105305…, 0.987692508896…]` | max | max |

  So neither `min_φ B_φ` nor `max_φ B_φ` is a valid general rule — each is refuted by some other
  control in the same table.
- **A conditional-mutual-information substitute also fails.** The gauge-free quantity
  `I(S;Y∣h(S),U)` reproduces the IR10 and P3-D oracles exactly (`1` and `1.5`) but not P4-A/B/C
  (`1` vs. oracle `0.5` on P4-A; `0.836653270753…` vs. oracle `0.392657719481…` on P4-B;
  `0.869565217391…` vs. oracle `0.987692508896…` on P4-C). Additionally conditioning on `h(Y)`
  does not repair these failures. The surgery cannot in general be replaced by a simple intra-fiber
  CMI.

## The formal policy

For a context `𝔠=(𝓜,q,𝓡)`, let `𝒥(𝔠)` be the set of intervention-lift families `J=(J_p)_{p∈P}`
satisfying, at minimum, I1–I3 (`docs/fiber-gauge-closure-and-transport.md`) plus any additional
constraint actually declared in `𝓡` before measurement. For each `J∈𝒥`, the surgery kernel is

\[
\widetilde K_J(y\mid s,u)=\sum_{s'\in[s]_J} q(s'\mid[s]_J)\,K(y\mid s',u),
\qquad
[s]_J=\{J_p(s):p\in P\},
\]

giving an identified set of kernels `𝓚̃={K̃_J : J∈𝒥}` and an identified set of scores
`𝓑={B_J : J∈𝒥}`, with bounds `B⁻=min 𝓑`, `B⁺=max 𝓑` on a finite domain. Statuses:

- **`PI-POINT`**: `|𝓑|=1` — the score is point-identified even if `J` itself is not.
- **`PI-PARTIAL`**: `1<|𝓑|<∞` — report the exact set `𝓑` if practical, otherwise at minimum the
  interval `[B⁻,B⁺]`. No element of `𝓑` is granted the status of "the" system value, and **no
  interval midpoint or other tie-break is authorized**.
- **`PI-EMPTY`**: `𝒥=∅` — the declared constraints are incompatible with the observed structure,
  or insufficiently specified. The score is `NA`.

If the effective causal denominator `K_eff` is already point-identified and independent of the
choice of `J`, the same policy lifts to the ratio `R`: `𝓡_set = {B/K_eff : B∈𝓑}`; otherwise the
uncertainty in `J` must be propagated jointly with the denominator's own uncertainty. Under any
causal-equivalence bijection transporting `K, q, π` and the constraints defining `𝒥`, the
identified set `𝓑` itself is invariant (a bijection may permute the individual lifts without
changing the set).

## Post-hoc illustration

`scripts/fiber_partial_identification_audit_v0_1.py` applies exactly this policy to five
controls, using as the admissible set all orthogonal trivializations of the fixed `P` (this
particular admissibility class is used to *illustrate* the formalism; it is not proposed as the
general definition of `𝒥`):

| Control | Status |
|---|---|
| IR10 | `PI-PARTIAL` (`𝓑={0.5,1}`) |
| P3-D | `PI-PARTIAL` (`𝓑={1, 1.125, 1.25, 1.5}`) |
| P4-A | `PI-PARTIAL` (`𝓑={0.5,1}`) |
| P4-B | `PI-PARTIAL` (8 distinct values) |
| P4-C | `PI-PARTIAL` (8 distinct values) |

Every one of these was previously reported as a clean point value by picking the historical
oracle's gauge; under the honest policy, all five are `PI-PARTIAL` given only "orthogonal
trivialization of a fixed `P`" as the admissibility constraint. Reaching `PI-POINT` therefore
requires genuinely new information not contained in `π` — exactly what `docs/jid-p5-p6-auxiliary-
signature.md` supplies conditionally, via a declared auxiliary causal signature.

## Scripts

- `scripts/fiber_partial_identification_audit_v0_1.py` — applies `PI-POINT`/`PI-PARTIAL`/
  `PI-EMPTY` to IR10, P3-D, P4-A, P4-B, P4-C.
- `scripts/fiber_full_gauge_min_diagnostic_v0_1_1.py` — included solely because the script above
  imports its helper functions (`complements`, `surgery`, `cmi`, ...) at run time; it is a code
  dependency, not a separately-documented result. Its own standalone diagnostic role (searching
  for a gauge-free `min`/CMI selector) is superseded by the analysis in this document.
