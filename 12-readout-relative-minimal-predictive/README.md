# 12 — Readout-relative minimal predictive realization

Clean English edition of the Readout-Relative Minimal Predictive Realization Protocol:
the current theoretical branch of the functional-consciousness campaign around a single
question:

> What is the smallest object that can generate every future observation available to
> a declared class of readouts, without storing the future target table itself?

Like campaign `10-continuous-predictive-geometry/`, this is a **mathematical /
protocol-design campaign**: a constructive theorem with proof sketches, exhaustively
verified on small finite systems. It is not a measurement on a trained model.

## What question this answers

The protocol distinguishes two different notions of minimality that are easy to
conflate:

1. **Behavioral quotient minimality** — the coarsest deterministic state quotient that
   preserves all future declared readouts.
2. **Linear predictive realization minimality** — the smallest reachable linear
   state-space realization that generates all future declared readout trajectories.

The central construction is the **Readout-Relative Minimal Predictive Realization**,
built from the Koopman closure of the declared readout space: given a deterministic
transition `F`, a source interface `E`, and a vector space of admissible readouts `D`,
the smallest `F`-invariant function space containing `D` is
`W = span{ d o F^k : d in D, k >= 0 }`. Each reachable state `x` gets an evaluation
functional `epsilon_x(f) = f(x)` on `W`; the predictive realization space is
`V = span{ epsilon_x }`, with transition `A_pred(epsilon_x) = epsilon_{F(x)}`. See
`docs/theorem-and-proofs.md`.

## Main results

1. **Exact predictive realization (Theorem 1)**: `(V, A_pred, source embedding, output
   functionals)` exactly generates every future trajectory observable by the declared
   readout class — `c_d A_pred^k eta(z) = d(F^k(E(z)))` for every source `z`, readout
   `d`, and delay `k`. Status: `THEORETICAL-DERIVATION`.
2. **Universal minimality and uniqueness (Theorem 2)**: every reachable exact linear
   realization surjects onto `V` via an intertwining map, so `dim(V)` is minimal among
   reachable exact linear realizations, and any two minimal realizations are
   isomorphic. Status: `THEORETICAL-DERIVATION`.
3. **Generalized Hankel rank equals minimal predictive dimension**, in finite-
   dimensional cases. Status: `THEORETICAL-DERIVATION + EXHAUSTIVE-FINITE-CHECK`
   (`0 / 512` failures on all binary `3x3` linear systems — see
   `docs/finite-verification-results.md`).
4. **The two minimal objects are genuinely different**: `Q_{E,D}` (behavioral quotient)
   is minimal among deterministic state quotients; `V_{E,D}` (linear realization) is
   minimal among reachable linear realizations, and can be strictly smaller because
   distinct behavioral states can be linearly dependent as predictive vectors. This is
   directly observed in the restricted-readout finite-state exhaustive check.
5. **Observable-semigroup generalization** covers controlled deterministic systems and
   finite-state Markov expectation dynamics as corollaries of one algebraic theorem —
   see `docs/semigroup-generalization.md`.
6. **Five exhaustive finite-universe checks, all `PASS`, zero failures across 13,538
   total map/source/kernel cases**: arbitrary-readout orbit corollary (3,840 cases),
   restricted-readout nonlinear predictive theorem (3,840 cases), U0-style binary
   linear corollary (512 systems), controlled deterministic extension (5,103 cases),
   and exact-rational finite Markov expectation corollary (243 cases). Full tables in
   `docs/finite-verification-results.md` and `results/*.json`.
7. **Open**: full stochastic path-law / probabilistic-bisimulation extension, and
   continuous/approximate realization. Explicitly not established here — see
   `docs/stochastic-extension-and-open-questions.md`.

## What this campaign explicitly does not claim

- Finite exhaustive checks validate the definitions and proof architecture on small
  universes; they are not a substitute for, or a proof of, the general algebraic
  theorem, which is `THEORETICAL-DERIVATION`, unreviewed outside this project.
- The finite-state Markov corollary preserves **expected** future declared readouts,
  not the full joint future path law; it must not be called probabilistic bisimulation.
- This theory characterizes predictive capacity, not realized causal use — the two are
  kept explicitly separate throughout (see `docs/rmt-bridge-and-status.md`, which also
  references this project's independent RMT causal-retargeting result as a different,
  complementary line of evidence).
- Every result carries an explicit anti-tautology gate: a candidate that merely
  re-encodes the target table under a different index is rejected as a substantive
  result unless it passes an independent held-out transport test — see
  `docs/protocol-and-guardrails.md`.

## Reproduction discipline

This campaign follows the same discipline as the rest of the repository (see the root
`README.md` for the full statement):

1. A **protocol** is written and frozen (hashed conceptually, by being committed as a
   `_FROZEN`-equivalent file — here, `PROTOCOL.md` together with the machine-readable
   `manifest.json`/`MANIFEST_SHA256.json`) *before* any candidate code is written or
   run.
2. A **generator + private oracle** pattern is not separately instantiated here in the
   RMT sense used by other campaigns — instead, the exhaustive small finite universes
   (all `4^4` maps on 4 states, all 512 binary `3x3` dynamics, all `729` controlled
   transition-map pairs, all `81` ordered Markov-kernel pairs) play the equivalent role
   of fixtures, and each verifier checks the theorem's own internal consistency
   conditions (closure, congruence, separation, well-definedness, Hankel-rank equality)
   exactly, with zero tolerance for failure.
3. Where this campaign would claim **"candidate-fixed holdout"**: the methodology
   guardrails (`docs/protocol-and-guardrails.md`) require any *learned or identified*
   predictive representation to be built on a strict calibration subset and tested on
   unseen horizons, contents, controls, or readout requests before being accepted —
   this is a protocol requirement for future empirical use of the theory, not exercised
   in this bundle, since every check here is an exact, exhaustive, closed-form
   computation rather than a learned representation.
4. Verdicts are never silently converted into a single point value when the protocol
   doesn't license one. Every claim in `results/claims_registry.json` carries an
   explicit status from a fixed vocabulary (`THEORETICAL-DERIVATION`,
   `EXHAUSTIVE-FINITE-CHECK`, `SAMPLED-CHECK`, `CONJECTURE`, `OPEN`) — the full
   stochastic path-law extension is honestly labeled `OPEN`, not silently assumed.

## How to verify / rerun

```bash
python3 scripts/run_all.py
```

This runs all five verifiers in sequence and writes their JSON output to
`results/*.json`; it fails loudly (`ALL VERIFIERS PASSED` is only printed if every
verifier exits 0). All scripts use the Python standard library only (plus
`fractions.Fraction` for exact rational arithmetic in the Markov corollary).

Individual verifiers can also be run directly, e.g.:

```bash
python3 scripts/verify_finite_predictive_theorem.py
python3 scripts/verify_u0_linear_corollary.py
python3 scripts/verify_controlled_finite_theorem.py
python3 scripts/verify_markov_expectation_corollary.py
python3 scripts/verify_all_readouts_orbit_corollary.py
```

## Contents

```
README.md
docs/
  theorem-and-proofs.md                 — deterministic Koopman-closure construction,
                                           exact realization, universal minimality,
                                           uniqueness, Hankel characterization, proofs
  semigroup-generalization.md           — observable-semigroup theorem covering
                                           controlled and Markov-expectation corollaries
  protocol-and-guardrails.md            — falsifiable claims (C1-C5), anti-tautology
                                           gates (G1-G4), validation ladder, reporting rule
  finite-verification-results.md        — all five exhaustive finite-universe checks
  stochastic-extension-and-open-questions.md — what the expectation-level extension
                                           does and does not preserve; open problems
  rmt-bridge-and-status.md              — capacity vs. realized causal binding, the A5
                                           retarget result, current status and handoff
scripts/
  common_gf2.py                         — shared GF(2) linear-algebra utilities
  verify_all_readouts_orbit_corollary.py
  verify_finite_predictive_theorem.py
  verify_u0_linear_corollary.py
  verify_controlled_finite_theorem.py
  verify_markov_expectation_corollary.py
  run_all.py                            — runs all five verifiers in sequence
results/
  all_readouts_orbit_corollary.json
  finite_predictive_theorem.json
  u0_linear_corollary.json
  controlled_finite_theorem.json
  markov_expectation_corollary.json
  claims_registry.json                  — every claim in this campaign with its status
  external_inputs_registry.json         — hashes/roles of external files this campaign
                                           refers to but does not itself contain
  STATUS.json                           — machine-readable summary of verdicts
```
