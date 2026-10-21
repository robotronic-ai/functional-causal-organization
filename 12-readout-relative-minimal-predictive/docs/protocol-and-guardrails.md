# Protocol, falsifiable claims, and methodology guardrails

Source material: `PROTOCOL.md` and `METHODOLOGY_GUARDRAILS.md`.

## Goal

Test and refine the Readout-Relative Minimal Predictive Realization without repeating
a previous failure mode in this project, where a candidate statistic secretly
re-encoded the target table rather than genuinely compressing it.

## Frozen definitions

Deterministic system `F : X -> X` (time-homogeneous); source interface `E : Z -> X`;
readout class `D`, a declared `K`-vector space of scalar readouts; predictive closure
`W = span{ d o F^k : d in D, k >= 0 }`; behavioral quotient `x ~_D y` iff all future
declared readouts agree; linear predictive realization
`V = span{ epsilon_x|_W : x reachable from E }` with transition induced by dual
Koopman evolution.

## Claims under test

- **C1 — Behavioral congruence.** The predictive equivalence relation is invariant
  under `F`. Falsifier: a pair `x ~ y` with `F(x) not~ F(y)`.
- **C2 — Behavioral coarseness.** Every deterministic quotient preserving all future
  declared readouts refines the predictive equivalence quotient. Falsifier: a strictly
  coarser quotient preserving every future declared readout.
- **C3 — Exact linear generation.** The predictive realization exactly reproduces
  every declared readout at every future delay. Falsifier: any `(z,d,k)` mismatch.
- **C4 — Linear minimality.** Every reachable exact linear realization surjects onto
  the predictive realization. Falsifier: an exact reachable linear realization of
  strictly smaller dimension.
- **C5 — Hankel identity.** For finite-dimensional cases, generalized Hankel rank
  equals predictive realization dimension. Falsifier: any exact finite example with
  unequal ranks.

## Mandatory anti-tautology gates

Before reporting any new zero-collision or exact-sufficiency result, apply all four:

- **Gate G1 — target-index overlap.** Ask whether the candidate is indexed by the same
  (content, role, horizon) cells as the target. If yes, the result requires an
  independent held-out transport test.
- **Gate G2 — bijection alarm.** If candidate class count equals target class count
  and both directions are collision-free, treat the candidate as a probable
  re-encoding until an independent generative derivation is shown.
- **Gate G3 — held-out transport.** Whenever the candidate is learned or identified
  from partial trajectories, construct it on a strict calibration subset and test it on
  unseen horizons, contents, controls, or readout requests.
- **Gate G4 — generative transition.** A claimed transport object must have a
  transition law that generates future outputs; a static table of future outputs is
  not sufficient.

## Validation ladder

1. Proof-level derivation under explicit assumptions.
2. Small exhaustive universes to catch definition mistakes.
3. Existing exact universes where computationally practical.
4. Controlled deterministic extension with multiple generators.
5. Stochastic expectation-level extension.
6. Full path-law / probabilistic-bisimulation extension, only after the
   expectation-level scope is made explicit.
7. Continuous approximate realization, with declared error metrics and held-out
   transport.

This campaign has completed rungs 1-5 of the ladder (see
`docs/finite-verification-results.md` for rungs 2-5); rungs 6-7 are `OPEN`
(`docs/stochastic-extension-and-open-questions.md`).

## Current finite checks

The included scripts perform: exhaustive four-state arbitrary-readout orbit corollary
checks; exhaustive four-state restricted-readout predictive realization checks; and
exhaustive U0-style `3x3` binary linear checks — plus, beyond what this document's
source material originally scoped, the controlled and Markov-expectation exhaustive
checks reported in `docs/finite-verification-results.md`.

## Reporting rule

Every result must be labeled as one of: `THEORETICAL-DERIVATION`,
`EXHAUSTIVE-FINITE-CHECK`, `SAMPLED-CHECK`, `CONJECTURE`, or `OPEN`. No finite
verification is described as proof of the general theorem — see
`results/claims_registry.json` for the actual label attached to every claim in this
campaign.

## Methodology guardrails

1. **Do not infer sufficiency from zero collisions alone.** A candidate can achieve
   zero collisions because it already contains the target table. Zero collisions are
   informative only when the candidate is independently generated or passes held-out
   transport.
2. **Apply the target-reencoding diagnostic first**: does the candidate share the
   target's index? Can the target be recovered by a system-independent unpacking
   operation? Are candidate and target partitions bijective on the tested universe? A
   positive answer warns that the result may be an identity rather than a substantive
   compression.
3. **Preserve the capacity / realized-use distinction.** `A_cap` means an admissible
   readout exists; `A_real` means the readout or causal binding actually installed in
   the system produces the behavior. A fresh external probe can establish capacity
   without establishing realized causal use (see `docs/rmt-bridge-and-status.md`).
4. **Keep organizational and task-relative objects typed separately.** `C_O` is an
   intrinsic organizational projection; source interface `E`, target task `L`, and
   readout class `D` are contextual/relational inputs. Do not silently append
   task-relative factors to `C_O` and call them new intrinsic organizational
   coordinates.
5. **Distinguish minimal deterministic quotient from minimal linear realization.** The
   behavioral quotient minimizes the number of deterministic predictive state classes;
   the predictive realization minimizes linear state-space dimension among exact
   reachable linear realizations. The latter can be smaller because distinct
   behavioral states can be linearly dependent as predictive vectors
   (`docs/theorem-and-proofs.md`, Section 6).
6. **Do not overclaim the stochastic extension.** Closure of expected readouts under a
   Markov Koopman operator preserves expected future observables, not full joint
   future path laws. Full probabilistic bisimulation requires a richer observable
   class or a direct path-law formulation.
7. **No new scalar metric by default.** The current line should not return to scalar
   combinations of `Gamma`, `Delta`, `R`, `q_A`, or similar summaries unless a specific
   falsifiable target requires them.
