# 08 — Λ addressing theory (causal sufficiency of quotient representations)

Clean edition of the theoretical campaign backing manuscript §17.5 ("Pont d'adressage") and
the §7.3 Λ discussion. This repository removes exploratory history and keeps only the
stabilized conceptual framework, core theorems, concise proofs, finite witnesses, and
executable checks.

## Central question

Given a coupled quantity

\[
L_\tau(O,E,D)=\Lambda_{E,D}^{(\tau)},
\]

when does it factor through a compressed causal description

\[
L_\tau = G_\tau \circ F\,?
\]

The theory treats this as a problem of **causal sufficiency of quotient representations** rather than as a search for a preferred closed-form function `G`.

## Reproduction discipline

This campaign follows the same reproducibility discipline as the rest of this repository (see
the root `README.md` for the full statement), expressed in the vocabulary appropriate to a
deductive theory campaign rather than an empirical one — see
`09-lambda-addressing-empirical/README.md` for the same discipline applied to the companion
data-driven fiber searches. There is no candidate-vs-oracle scoring here: the object under
test is a set of theorems and finite witnesses, not a black-box predictor. Each theorem's
proof is written out in full in `docs/THEOREMS_AND_PROOFS.md`; each finite witness domain
(`D`, `E`, `ED`, `OED`) is an exact, hand-specifiable structure whose invariants are
*computed*, not fit, by `scripts/verify_theory.py`, and the frozen expected output is recorded
in `verification_output.json`. `MANIFEST_SHA256.json` fixes a SHA-256 hash for every file in
this archive, so any silent post-hoc edit to a theorem, a witness, or the verification script
is externally detectable. Where the companion empirical campaign reports verdicts as
`ESTABLISHED` / `NOT-ESTABLISHED` / `WITHHELD` / `OPEN`, and the other campaigns in this
repository report `V1/POINT` / `V3/PARTIAL` / `V4/NA`, this campaign reports each theorem as
proved (QED) or not claimed, and each witness as an exact-computed pass/fail against the
frozen output — no numeric identity here is asserted without either a closed-form proof or a
from-scratch exact recomputation.

## Main results

1. Fiber Factorization Theorem.
2. Refinement Monotonicity Theorem.
3. Minimal Admissible Sufficiency.
4. Symmetry Obstruction Theorem.
5. Causal-Groupoid Factorization Theorem.
6. Closed-Quotient Transport Theorem.
7. Certified-Quotient Composition Theorem.

## Finite witnesses

Four exact toy domains demonstrate distinct failure modes:

- `D`: information specific to a coupling can be necessary.
- `E`: instantaneous environmental variety can miss temporal structure.
- `ED`: separate quotients can lose relative alignment.
- `OED`: all pairwise relations can be insufficient; a ternary invariant can be necessary.

## Quick check

```bash
python scripts/verify_theory.py
```

The script uses only the Python standard library.

## Scope

`Lambda_O`, `V_E`, `Lambda_{E,D}`, `Xi`, and the sufficiency signature developed here are theoretical extensions. They are not claimed to be pre-existing definitions in the main manuscript. The framework is constrained by the manuscript's existing principles: fixed causal boundaries, typed perturbation-response channels, causal equivalence, and temporal closure.
