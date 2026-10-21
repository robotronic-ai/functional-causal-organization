# RHO-CERT: certifying the causal-role equivalence ρ

## Position in the campaign

CAN-PN v0.2 established that N's canonicity is not needed for the estimand,
and that a P-free construction exists once two things are certified: a
causal-role equivalence ρ, and a boundary T_{t+1}. RHO-CERT treats **only**
ρ. T_{t+1} is audited separately in `target-boundary.md`.

## What ρ has to decide

For functional variables `v, w`, decide whether they occupy the same causal
role, written `v ~ρ w`. Equality of names, dimensions, marginal
distributions, or local degree is never sufficient — the protocol
(`Protocole_RHO_CERT_v0.2_FROZEN.md`) requires the verdict to be derived only
from causally accessible O1 data: finite alphabets, the minimal quotient
graph, structural/interventional kernels, the declared intervention algebra,
and — when a loop is unrolled — an explicit temporal transport. No hidden
"same role" annotation is ever available to the candidate.

## Finite interface (v0.1)

Each fixture supplies, per variable `v`: a finite alphabet `X_v`, a family of
incoming causal responses `Q^in_{v,p}(x)`, and a family of outgoing causal
responses `Q^out_{v,r}(o | do(v=x))`. Two variables are only compared when the
fixture supplies a `candidate_transport` between them (e.g. a declared time
shift) — the transport does not assert same-role, it only asserts that the
comparison is structurally admissible.

A certificate `v ~ρ w` exists iff there is a bijection `g : X_v -> X_w` such
that (1) every incoming distribution coincides after pushing through `g`,
and (2) for every outgoing port, some bijection of its observation alphabet
makes every conditional distribution coincide after transporting `x` by `g`.
Certificates form an undirected compatibility graph; a role partition is
**admissible** if every block is a clique of that graph, and **maximal** if
no two blocks can be merged and remain a clique. Ordinary negative fixtures
are not failures — they simply produce singleton blocks inside an `RHO-V1`
partition.

## Protocol obligations (RHO0-RHO7)

- **RHO0** (alphabet bijection): different cardinalities immediately force
  `v not~ρ w`.
- **RHO1** (incoming signature): under an admissible transport of causes,
  incoming kernels must coincide up to relabeling.
- **RHO2** (outgoing signature): under an intervention on the variable
  itself, its future functional effects must also coincide — a local match
  with different futures must be rejected.
- **RHO3** (temporality): in an unrolled homogeneous dynamics, instances
  like `A_t, A_{t+1}` can share a role despite different windows; the
  candidate must compare signatures modulo a declared shift δ or on an
  already-folded temporal quotient — naively comparing in/out degree within
  a single window is disallowed.
- **RHO4** (compositional coherence): if `v~ρ w` and `w~ρ x`, the bijections
  and transports must compose coherently (`g_vx = g_wx . g_vw` up to an
  admissible internal automorphism); the final relation must be an
  equivalence.
- **RHO5** (cautious maximality): the maximal relation is generated from all
  compatible exact certificates, but if several non-isomorphic maximal
  partitions survive, or distinct temporal transports yield incompatible
  partitions, the verdict is `RHO-PARTIAL` — no tie-break is allowed.
- **RHO6** (representation invariance): invariant under bijective alphabet
  recoding, variable renaming, insertion of reconstructible mediators
  removed at the minimal quotient, and causally-equivalent
  unrolling/folding.
- **RHO7** (independence of q): ρ is structural; changing the measurement
  battery must not change it.

## Verdicts

- **RHO-V1** — point-identified: a unique maximal relation satisfies
  RHO0-RHO7.
- **RHO-V2** — refuted: the candidate declares same-role for a pair that
  violates a required interventional kernel.
- **RHO-V3** — partially identified: several non-isomorphic maximal
  relations remain compatible.
- **RHO-V4** — interface insufficient: the window/intervention algebra does
  not contain enough information to test the role obligations. Status `NA`,
  not false.

## Prospective control families (frozen before the candidate)

`R0` unrolled positive loop, `R1` negative pipeline, `R2` false historical
same-role, `R3` same local law but different future, `R4` reconstructible
mediator, `R5` >=32 deterministic alphabet/identifier recodings, `R6`
rational stochasticity in `{0, 1/4, 1/2, 3/4, 1}`, `R7` internal automorphism
invariance, `R8` genuine ambiguity (expected `RHO-PARTIAL`, to keep "several
certificates" distinct from "unique canonical relation").

## First run: the R2/R3 oracle defect (kept, not hidden)

Candidate `RHO-EXACT-CERT v0.1`
(SHA-256 `90bbbbc4cd0958a31ed70d3cbd5d6c71c4a825495c5e6fef8832a010b85a785d`)
was run against the frozen v0.1 fixtures/oracles.

    EXACT_PASSES: 45/49
    R2: 0/2
    R3: 0/2
    RHO-CERT VERDICT: RHO-CANDIDATE REFUTED

`RHO_CERT_v0.1_FIRST_RUN_ADJUDICATION.md` diagnoses the four failures
(`R2.01, R2.02, R3.01, R3.02`) exactly: in every case, a binary bijection
correctly certifies a transport between the two profiles that the oracle had
labeled different-role. The "future" difference the generator had tried to
introduce was, in fact, only a recoding (a permutation of the source value
and/or of the outgoing observation alphabet) — admissible under RHO0-RHO2 and
the frozen interface. The verdict:

    boxed: FIXTURE/ORACLE DEFECT - NOT A SCIENTIFIC REFUTATION

The candidate had correctly detected a causal equivalence that the oracle had
mislabeled negative. The v0.1 generator and its oracles were left immutable;
a v0.1.1 fixture set corrects only the negative profile C so that it is truly
non-isomorphic. This repair is explicitly **post-hoc** and does not count as
prospective validation:

    EXACT_PASSES: 49/49   (post-hoc consistency check only)

To recover an independent evaluation, the v0.1 candidate itself was left
unchanged and was instead tested on a new holdout created after it was
frozen.

## Candidate-fixed adversarial holdout — the load-bearing result

`Protocole_RHO_CERT_HOLDOUT_CF_v0.1_FROZEN.md` fixes the same candidate
(`RHO-EXACT-CERT v0.1`, same SHA-256) **before** generating a new,
independent fixture set — 8 families (`H0`-`H7`) x 4 deterministic recodings
= 32 fixtures, covering mismatched alphabet cardinalities, positive ternary
roles, matching-in/differing-future and differing-in/matching-future cases,
a 3-role transitive class, genuine structural ambiguity, two distinct
positive classes with negative cross-comparisons, and an explicitly
insufficient interface.

Result (`RHO_CERT_HOLDOUT_CF_RESULT_v0.1.md`):

    CANDIDATE: RHO-EXACT-CERT 0.1
    EXACT_PASSES: 32/32
    H0: 4/4  H1: 4/4  H2: 4/4  H3: 4/4
    H4: 4/4  H5: 4/4  H6: 4/4  H7: 4/4
    FIRST_FAILURE: None
    RHO-CF-HOLDOUT VERDICT: CF-HOLDOUT-PASS

    boxed: RHO candidate-fixed finite non-refutation

This candidate-fixed holdout — no modification of the candidate between
freezing and running, fixtures generated only after the freeze — is what
carries the campaign's actual reproducibility claim for ρ, with explicit
handling of point-identified roles (`RHO-V1`), several maximal partitions
(`RHO-V3`), and interface insufficiency (`RHO-V4`).

## Files in this edition

- Final candidate: `scripts/rho_candidate_exact_certificate_v0_1.py`.
- Harness (fixed profiles v0.1.1): `scripts/rho_cert_harness_v0_1_1.py`.
- Holdout harness: `scripts/rho_cert_holdout_cf_harness_v0_1.py`.
- Fixtures: `results/RHO_CERT_FIXTURES_v0_1_1.json`,
  `results/RHO_CERT_HOLDOUT_CF_FIXTURES_v0_1.json`.
- Recorded numbers: `results/STATUS.json`.

This note is a clean-English digest of the original frozen protocol and
result documents (`Protocole_RHO_CERT_v0.2_FROZEN.md`,
`RHO_CERT_v0.1_FIRST_RUN_ADJUDICATION.md`,
`Protocole_RHO_CERT_HOLDOUT_CF_v0.1_FROZEN.md`,
`RHO_CERT_HOLDOUT_CF_RESULT_v0.1.md`); those source documents are not
themselves included in this edition.
