# The PN theorem: N's canonicity is not needed for the estimand

## The problem this removes

The campaign starts from the historical decomposition

    Z_{t+1}^ret ~= (P_{t+1}, N_{t+1})

and the question of whether the persistent/non-return split P/N must itself be
canonical in order to define re-entry. `Protocole_CAN_PN_v0_2_FROZEN.md`
splits this into three questions that earlier work had been conflating:

1. is the **estimand** canonical (does it depend on which N you pick)?
2. is the **boundary to neutralize** canonical?
3. is the inter-fiber **transport J** identified?

This note is about question 1 only.

## Statement

Under hypotheses M1-M3 (the persistent factor is fixed and the cut is
complete — `Z_t* not in An(P_{t+1})` in the mutilated model), the PN theorem
gives:

    I_q^do(Z_t* ; P_{t+1} | U_t) = 0.

By the chain rule, for *any* explanatory bijection `Z_{t+1}^ret <-> (P_{t+1}, N_{t+1})`:

    I_q^do(Z_t* ; N_{t+1} | P_{t+1}, U_t) = I_q^do(Z_t* ; Z_{t+1}^ret | U_t).

So the mutual information the estimand actually needs is already fully
determined by the return-complex `Z_{t+1}^ret` and `U_t`; it does not depend
on which particular N was used to write the bijection. This is not an extra
assumption layered on top of PN — it is a direct consequence of the chain
rule once P is fixed.

    boxed: the canonicity of N is not required for the final estimand.

The `PN-MIX` control checks this directly: it re-runs the identity under all
24 bijective mixings of the historical P/N coordinates and requires
`B_full` (the full boundary functional) to stay invariant even when the new
first coordinate carries all the information. This blocks any silent
reintroduction of a canonical N through the back door.

## What is *not* closed by this alone

- The identification of the inter-fiber transport J is a separate problem,
  conditionally closed elsewhere (by P5/P6, campaign `02-`), not by PN.
- What remains open after PN is the canonicity of the *boundary to
  neutralize itself* — that is the subject of `rho-certification.md` and
  `target-boundary.md`.

## Historical regression audit (CAN-PN v0.2)

`CAN_PN_v0_2_HISTORICAL_RESULT.md` re-validates every prior witness under the
frozen v0.2 protocol. This is an audit, not a new extractor: no new candidate
factor-extraction code is exercised. All mandatory historical obligations
pass:

- **PN**: exact chain-rule identity; neutralization holds on the recorded
  oracle cuts.
- **PN-MIX**: `B_full` invariant under the 24 bijective mixings of the old
  P/N coordinates.
- **F8** (empty-cut rejection): the empty cut is correctly rejected on
  `F0.4`, `F1.1`, `F4P.1`, `F6.1`, `F6.2`.
- **IR11**: no strict G-Set factor (B1/B2) is accepted — the historical
  append-only false positive stays rejected.
- **IR10**: a non-trivial strict factor is present; the finite OPF
  diagnostic yields a unique append-only factor.
- **P3-D**: the finite OPF diagnostic yields a unique append-only factor.
- **P4-A/P4-B**: `P-VALID` for the persistent factor together with
  `J-PARTIAL` for the transport — a declared-valid P is never rejected
  merely because J is ambiguous.
- **IR9/IR12** remain explicitly `OPEN`/`PARTIAL`; they are not silently
  reclassified as positive or negative canonicity controls.

Verdict:

    boxed: CAN-PN-HIST-V1 - ALL MANDATORY HISTORICAL OBLIGATIONS PASS

### Consequence: the P/N question splits into three tracks

1. **Canonicity of N** — resolved: not needed for the final estimand
   (this document).
2. **Canonicity of the boundary to neutralize** — still open at this point;
   addressed by `rho-certification.md` and `target-boundary.md`.
3. **Identification of J** — a separate work stream, conditionally closed by
   P5/P6 (campaign `02-`).

The next authorized campaign is explicitly **not** a new append-only
extractor. It is the canonical certification of ρ (a causal-role equivalence)
and of T_{t+1} (the boundary of the next macro-state) from the O1 interface —
see `rho-certification.md` and `target-boundary.md`.

## Source protocol requirements (CAN-PN v0.2), for reference

- **CAN0**: elimination of the complement N (the theorem above).
- **CAN1**: invariance under bijective state recoding, context relabeling,
  insertion/removal of reconstructible representations, and admissible
  causal equivalences.
- **CAN2**: the persistence/return split does not depend on the measurement
  battery q (q only enters the *estimation* of the boundary, never the
  decision of which factor is persistent).
- **CAN3**: strict persistence semantics — irreversible retention of
  functionally relevant information, an accumulation/lumping dynamics
  compatible with interventions, not mere succession, and not an artificial
  factor created by partitioning the current code (IR11-like factors are
  rejected).
- **CAN4**: complete neutralization — a declared persistent boundary must
  satisfy `Z_t* not in An(P_{t+1})` in the cut model; a cut that still leaves
  a path `Z_t* -> P_{t+1}` is rejected even if the old conditional score was
  already zero (this is obligation F8/M1).
- **CAN5**: separation between P and J — identifying P does not imply
  identifying the transport `J_p : S -> pi^{-1}(p)`; a case can be
  simultaneously `P-VALID` and `J-PARTIAL`.
- **CAN6**: abstention — if several non-isomorphic separations satisfy every
  available structural constraint, the output is `PN-PARTIAL`/`NA`; no
  post-hoc numerical tie-break, arbitrary maximality, or gauge choice is
  allowed.
- **CAN7**: the P-free branch — if the untyped boundary of the next
  macro-state T_{t+1} is given at the minimal quotient, and a causal-role
  relation ρ is causally certified, then the return factor can be defined as
  `R_{t+1}^{ρ*} = {X in T_{t+1} : ρ(X) = ρ(Z_t*)}` and the boundary to
  neutralize as `H_{t+1} = T_{t+1} \ R_{t+1}^{ρ*}`. The P/N split is then no
  longer a primitive; the remaining lock is the canonicity/certification of
  ρ and T_{t+1}. This is exactly the construction closed in
  `composition-and-closure.md`.

## PN-CLOSE-A / PN-CLOSE-B

Point 2 (the P/N canonicity question) is declared closed once both hold:

- **PN-CLOSE-A** (estimand): N's canonicity is eliminated analytically by
  PN under the neutralization hypotheses — established here, no extractor
  required.
- **PN-CLOSE-B** (boundary): the return/persistence boundary is either
  canonically derived from O1, or derived from a causally certified ρ and a
  canonical T_{t+1}, or explicitly declared unidentified
  (`PN-PARTIAL`/`NA`) — this requires an extractor/certificate, delivered by
  the RHO-CERT and TARGET-BOUNDARY campaigns.

## Files in this edition

This is an analytic result (a chain-rule consequence of M1-M3): there is no
dedicated candidate/harness/fixture bundle for the PN theorem itself. This
note is a clean-English digest of the original frozen protocol and result
documents (`Protocole_CAN_PN_v0_2_FROZEN.md`,
`CAN_PN_v0_2_HISTORICAL_RESULT.md`); those source documents are not
themselves included in this edition. See `results/STATUS.json` for the
recorded verdict (`CAN-PN-HIST-V1`).
