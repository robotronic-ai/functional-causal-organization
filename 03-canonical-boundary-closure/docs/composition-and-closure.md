# Composition of ρ and T, and the final CLOSE-CONDITIONAL status

## The P-free construction

With ρ certified (`rho-certification.md`) and T_{t+1} certified
(`target-boundary.md`), the campaign composes them into a definition of the
return boundary that never needs a primitive P/N split
(`POINT2_CLOSURE_v0_1.md`, section 7):

    T_{t+1} = B                         (the certified boundary)
    r* = ρ(Z_t*)                        (the source's certified role)
    R_{t+1}^{ρ*} = { X in T_{t+1} : ρ(X) = r* }
    H_{t+1} = T_{t+1} \ R_{t+1}^{ρ*}

The cut neutralizes every downstream route from the current role into
`H_{t+1}`, and the final estimand is measured on the return factor
`R_{t+1}^{ρ*}`. The P/N primitive is gone from the main definition.

## Composition rule

For each minimal boundary `B` produced by TARGET-BOUNDARY and each maximal
role partition `Π_ρ` produced by RHO-CERT, let `C_{ρ*}` be the partition
block containing the source's role. Define:

    R = B ∩ C_{ρ*},   H = B \ R.

All **distinct** triples `(B, R, H)` across every `(B, Π_ρ)` combination are
kept. Verdicts: a single triple -> `PN-V1`; several distinct triples ->
`PN-V3`; a `TB-V4` from TARGET-BOUNDARY or a `RHO-V4` from RHO-CERT ->
`PN-V4`. Note this means a ρ-ambiguity that lies entirely outside the
boundary can collapse harmlessly to `PN-V1` — only ambiguity that actually
changes which nodes of `B` are classified as return vs. non-return survives
into `PN-V3`.

## First integrated run: the I3 fixture defect (kept, not hidden)

`Protocole_POINT2_INTEGRATED_HOLDOUT_v0.1_FROZEN.md` fixes both candidates
(TARGET-BOUNDARY-EXACT v0.2,
SHA-256 `28019ac36cf9f38a97526e378405a13ddd0d49109fb981d4346918c2f866e768`;
RHO-EXACT-CERT v0.1,
SHA-256 `90bbbbc4cd0958a31ed70d3cbd5d6c71c4a825495c5e6fef8832a010b85a785d`)
and defines 9 families (`I0`-`I8`, 2 recodings each = 18 fixtures) covering:
a mixed return/non-return unique boundary, an all-return unique boundary, an
all-non-return unique boundary, `TARGET-BOUNDARY` partial, `RHO` partial
affecting the boundary, `RHO` partial only outside the boundary (final
classification still unique), `TARGET-BOUNDARY` `TB-V4`, `RHO` `RHO-V4`, and
a genuinely empty boundary.

First run (`POINT2_INTEGRATED_v0.1_ADJUDICATION.md`):

    16/18

Failures: `I3.01`, `I3.02`. Diagnosis: TARGET-BOUNDARY correctly returns two
minimal boundaries `{a}` and `{b}`. The fixture intended to also assert
`ρ(src) = ρ(a) = ρ(b)`, but its RHO interface only supplied the transports
`src <-> a` and `src <-> b` — **not** a certificate `a <-> b`. RHO-CERT
defines a role class as a clique of certified compatibilities and then takes
maximal partitions, so with that interface it legitimately produces two
maximal partitions, `{src,a} | {b}` and `{src,b} | {a}`, not one. Composition
then correctly yields four classifications instead of the two the oracle
expected. Verdict:

    boxed: INTEGRATED FIXTURE/ORACLE DEFECT

Neither TARGET-BOUNDARY nor RHO-CERT was refuted, and the composition rule
itself was not refuted either — it correctly propagated the RHO ambiguity
that the oracle had failed to account for. The first fixture lot was left
immutable, and a new integrated holdout v0.2 was built with both candidates
and the composition rule already frozen.

## Corrected integrated holdout (v0.2): 18/18

`Protocole_POINT2_INTEGRATED_HOLDOUT_v0.2_FROZEN.md` is created **after**
the I3 adjudication, without modifying either candidate or the composition
rule (composition-rule/harness SHA-256
`cddaf4fbdb9f6cf2a867e5dcababbb8cb71690bfc83017152f4445938143e165`). New
families `J0`-`J8` (2 recodings each = 18 fixtures): a triple unique
boundary mixing return/non-return (`J0`), `TARGET-BOUNDARY` partial on a
diamond with `RHO` unique (`J1`), `RHO` partial affecting the boundary
(`J2`), `RHO` partial only outside the boundary (`J3`), a unique
all-non-return boundary (`J4`), `TARGET-BOUNDARY` `TB-V4` (`J5`), `RHO`
`RHO-V4` (`J6`), an empty boundary (`J7`), and simultaneous TB and RHO
ambiguity (`J8`).

Result:

    boxed: 18/18

    boxed: POINT2-INTEGRATED-PASS

A control inside this holdout also confirms that a ρ-ambiguity located
*outside* the boundary can resolve harmlessly: whenever every maximal
partition induces the same `(B, R, H)`, the composed verdict still collapses
to a single classification.

## Final scientific status: CLOSE-CONDITIONAL

Point 2 is declared

    boxed: CLOSE-CONDITIONAL

and not a "universal factorization theorem." The closure is explicitly
conditional on:

1. an SCM/macro-graph reduced to the minimal causal quotient;
2. an intervention algebra rich enough for the RHO-CERT and TARGET-BOUNDARY
   obligations;
3. the PN hypotheses M1-M3, used to eliminate the complement N from the
   estimand;
4. the abstention policy — `PN-V3` whenever several boundaries/roles change
   the classification, `PN-V4`/`NA` whenever the interface is insufficient.

No hidden tie-break is permitted anywhere in the chain.

### Order of battle

With the factorization-identifiability work stream (campaigns `01-`/`02-`)
and this P/N-canonicity/return-boundary work stream both now
`CLOSE-CONDITIONAL`, the next authorized campaign is the true
characterization and axiomatic independence of Γ and Δ
(`04-gamma-delta-characterization/`). The empirical measurement of K_eff and
R on a transformer (`05-`, `06-`) is deliberately sequenced after that.

## Files in this edition

- Harness (composition rule + both fixed candidates):
  `scripts/point2_integrated_holdout_harness_v0_2.py`.
- Fixtures: `results/POINT2_INTEGRATED_HOLDOUT_FIXTURES_v0_2.json`.
- Recorded numbers and final status: `results/STATUS.json`.

This note is a clean-English digest of the original frozen protocol and
result documents (`Protocole_POINT2_INTEGRATED_HOLDOUT_v0.1_FROZEN.md`,
`POINT2_INTEGRATED_v0.1_ADJUDICATION.md`,
`Protocole_POINT2_INTEGRATED_HOLDOUT_v0.2_FROZEN.md`, and the closure
statement `POINT2_CLOSURE_v0_1.md`); those source documents are not
themselves included in this edition. The v0.1 integrated-holdout
fixtures/oracles/manifests are also not included — they are superseded by
v0.2 after the I3 defect; only the adjudication note's content (which
describes the defect) is digested here.
