# TARGET-BOUNDARY: identifying the canonical next-macro-state boundary T_{t+1}

## Position in the campaign

With ρ certified (`rho-certification.md`), the last lock in the P-free
construction is the untyped boundary of the next macro-state, T_{t+1}: the
subset of the next-state variables that must be neutralized by intervention
so that no information from the current source can reach the declared
future by any other route.

## Setup

A finite, deterministic, acyclic unrolled SCM with a current source variable
`Z_t*`, optional background inputs `U_t` fixed before intervention, a
declared set of variables eligible for the next macro-state `E_{t+1}`, and a
declared set of future observables `F_{>t+1}`. The interface is already
assumed to be at the **minimal macro quotient** — purely reconstructible
mediators have been eliminated upstream.

## v0.1: interventional sufficiency alone

A subset `B ⊆ E_{t+1}` is **sufficient** if, for every background value `u`,
every assignment `b` of `B`, and every pair of source values `z, z'`:

    L(F_{>t+1} | do(Z_t*=z), do(B=b), U_t=u)
      = L(F_{>t+1} | do(Z_t*=z'), do(B=b), U_t=u).

(v0.1 fixtures are deterministic, so this is exact-output equality.) `B` is
**minimal by inclusion** if no strict subset is sufficient; `T` is the set of
all such minimal boundaries. Verdicts: `TB-V1` (`|T|=1`, including the empty
boundary), `TB-V3` (`|T|>1`, report all of them, no tie-break by cardinality,
depth, entropy, or graph proximity), `TB-V4` (no eligible subset is
sufficient — a route bypasses the eligible interface entirely; `NA`, not a
forced choice).

## The v0.1 refutation: functional cancellation hides a bypass

Candidate `TARGET-BOUNDARY-EXACT v0.1`
(SHA-256 `17ce4c2cda156751deab551d344dcf3ac90d42b0f63134a22eac5eaccde7bb32`),
frozen after the pre-candidate bundle, scored:

    28/32

All failures belong to family `T3` (bypass). The counterexample
(`TARGET_BOUNDARY_v0.1_REFUTATION.md`):

    A := Z,   Y := Z xor A.

`Y` is identically 0 for every value of `Z`. The v0.1 criterion, which looks
only at `L(F | do(Z=z), do(B=b))`, therefore classifies `B = empty` as
sufficient — the *law* of the future doesn't depend on `Z` once nothing is
fixed. But a direct causal edge `Z -> Y` bypasses every eligible variable of
the next state: the functional cancellation masks a real structural
bypass. Verdict:

    boxed: TB-v0.1 SUFFICIENCY-ONLY CRITERION REFUTED

This was neither a candidate bug nor an oracle defect: the candidate
correctly implements the v0.1 criterion, `T3` was deliberately designed to
require a genuine notion of causal boundary, and the run demonstrates that
non-robust interventional independence alone is too weak in the presence of
functional cancellations. The v0.1 fixture bundle was left immutable; the
repair required is explicit: **v0.2 must require a structural cut in
addition to interventional sufficiency.**

## v0.2: structural cut ∧ interventional sufficiency

`Protocole_TARGET_BOUNDARY_v0.2_FROZEN.md` adds a structural requirement.
For `B ⊆ E_{t+1}`, form the graph mutilated by `do(B=b)` (incoming edges into
`B`'s nodes removed, outgoing edges kept), and require

    boxed: F_{>t+1} ∩ Desc_{M^do(B)}(Z_t*) = empty,

independent of the value `b` — no future node remains a descendant of the
source once `B` is fixed by intervention. This is combined with the exact
v0.1 interventional-sufficiency test. A boundary is **admissible** iff it
satisfies `STRUCTURAL-CUT ∧ INTERVENTIONAL-SUFFICIENCY`, and the reported set
is the inclusion-minimal admissible boundaries. Verdicts as before
(`TB-V1`/`TB-V3`/`TB-V4`); no tie-break by minimal cardinality between
several inclusion-minimal boundaries (control `S4`, a diamond with a
bottleneck, exists specifically to forbid that shortcut: `{C}` and `{A,B}`
are both minimal, and the correct verdict is `TB-V3`, not "pick the smaller
one"). Parent lists are required to be *effective* causal edges of the
minimal macro quotient — a mechanism that never actually depends on a
declared parent must be eliminated first, so that no syntactic-only
dependency manufactures a spurious boundary.

Prospective control families: `S0` reinforced cancellation-with-bypass
(expects `TB-V4`), `S1` unique singleton series, `S2` parallel branches that
must be cut together, `S3` ambiguous chain (`TB-V3`), `S4` diamond with
bottleneck (`TB-V3`, blocks the cardinality tie-break), `S5` direct edge
bypassing the eligible interface (`TB-V4`), `S6` genuinely empty boundary
(`TB-V1`), `S7` two futures plus an irrelevant eligible variable.

## v0.2 result: prospective pass, then holdout pass

Candidate `TARGET-BOUNDARY-EXACT v0.2`
(SHA-256 `28019ac36cf9f38a97526e378405a13ddd0d49109fb981d4346918c2f866e768`),
frozen before the v0.2 fixtures. Prospective result
(`TARGET_BOUNDARY_v0.2_RESULT.md`):

    32/32
    S0: 4/4  S1: 4/4  S2: 4/4  S3: 4/4
    S4: 4/4  S5: 4/4  S6: 4/4  S7: 4/4

    boxed: TB-PROSPECTIVE FINITE NON-REFUTATION

A candidate-fixed adversarial holdout was then generated after freezing this
same candidate (`Protocole_TARGET_BOUNDARY_HOLDOUT_CF_v0.1_FROZEN.md`),
adding ternary series (`H0`), parallel ternary branches (`H1`), a diamond
with differently-sized minimal cuts (`H2`), bypass-with-cancellation (`H3`),
a genuinely empty boundary with two background inputs (`H4`), three parallel
routes with a unique triple boundary (`H5`), two competing bottlenecks after
a merge (`H6`), and a misleading eligible variable alongside an uncuttable
direct route (`H7`) — 8 families x 4 recodings = 32 fixtures. Result
(`TARGET_BOUNDARY_HOLDOUT_CF_RESULT.md`):

    32/32

    boxed: TB-CF-HOLDOUT-PASS

## Files in this edition

- Final candidate: `scripts/target_boundary_candidate_exact_v0_2.py`.
- Harness: `scripts/target_boundary_harness_v0_2.py`.
- Holdout harness: `scripts/target_boundary_holdout_cf_harness_v0_1.py`.
- Fixtures: `results/TARGET_BOUNDARY_FIXTURES_v0_2.json`,
  `results/TARGET_BOUNDARY_HOLDOUT_CF_FIXTURES_v0_1.json`.
- Recorded numbers: `results/STATUS.json`.

This note is a clean-English digest of the original frozen protocol and
result documents (`Protocole_TARGET_BOUNDARY_v0.1_FROZEN.md`,
`TARGET_BOUNDARY_v0.1_REFUTATION.md`,
`Protocole_TARGET_BOUNDARY_v0.2_FROZEN.md`, `TARGET_BOUNDARY_v0.2_RESULT.md`,
`Protocole_TARGET_BOUNDARY_HOLDOUT_CF_v0.1_FROZEN.md`,
`TARGET_BOUNDARY_HOLDOUT_CF_RESULT.md`); those source documents are not
themselves included in this edition. The v0.1 fixtures/oracles and the v0.1
harness/generator scripts are also not included — they are superseded by
v0.2 after the refutation above; only the refutation *record* (which
motivates v0.2) is digested here.
