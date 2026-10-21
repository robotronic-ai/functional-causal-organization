# 03 — Canonicity of the P/N split and closure of the return boundary

Clean English edition of campaign `03-canonical-boundary-closure`, part of
the reproduction package for the causal-organization framework (Γ, Δ, R,
Λ, C_O) behind functional access capability. It removes debugging/hotfix
history and internal process notes; it keeps every stabilized theorem,
every candidate that ended up canonical, both refutations that shaped the
next candidate, and the final result data needed to check the claims.

## Reproduction discipline (recap)

Every campaign in the source repository follows the same protocol, and the
discipline itself is part of what is being reproduced:

1. A **protocol** is written and hashed before any candidate is coded.
2. A **generator** produces fixtures; a private **oracle** file records the
   expected verdict for each fixture, and is never edited after a candidate
   exists except through a documented **adjudication** note when a fixture
   or oracle itself turns out to be wrong.
3. A **candidate** is written against the frozen fixtures; its hash is
   recorded.
4. Where a campaign claims a **candidate-fixed holdout**, a second,
   independent fixture set is generated *after* the candidate's hash is
   frozen, and the same, unmodified candidate is re-run against it. This is
   the strongest reproducibility claim made anywhere in the source
   repository.
5. Verdicts are never forced to a single value when the data does not
   determine one:
   - **V1 / POINT** — a unique value is identified.
   - **V3 / PARTIAL** — several values remain compatible with the data; the
     *set* is reported, not a chosen element of it.
   - **V4 / NA** — the available intervention algebra is insufficient to
     certify even the factorization; the framework abstains.

No bundle anywhere in the source repository silently converts a V3/NA case
into a V1 number. This folder inherits that discipline and reports two real
mid-campaign defects and one real refutation exactly as they happened —
see below.

## The question this campaign answers

The starting point is an analytic result that already removes one half of
the historical problem: under hypotheses M1–M3, `I_q^do(Z*_t; P_{t+1} | U_t)
= 0`, so — by the chain rule — the canonicity of the non-return complement N
is *not* required for the final re-entry estimand; it falls out once the
persistent factor P is fixed, it is not an extra assumption. What remains is
whether the **boundary to neutralize** (which variables of the next state
must be cut off to measure re-entry cleanly) is itself canonical, absent a
canonical N. The campaign answers this with a P-free construction:

    R_{t+1}^{ρ*} = { X in T_{t+1} : ρ(X) = ρ(Z_t*) },   H_{t+1} = T_{t+1} \ R_{t+1}^{ρ*}

built from a certified causal-role equivalence **ρ** (do two variables play
the same causal role?) and a canonically derived minimal **target boundary**
**T_{t+1}** (which next-state variables must be cut to block every route
from the source into the declared future).

## Result narrative

### 1. N's canonicity is eliminated analytically (the PN theorem)

Under M1–M3, `I_q^do(Z*;P|U) = 0`; by the chain rule this forces
`I_q^do(Z*;N|P,U) = I_q^do(Z*;Z_ret|U)` for *any* explanatory bijection
`Z_ret <-> (P,N)` — the estimand cannot depend on which N was used to write
it. A historical regression audit (no new extractor) re-validates every
prior witness under this reading. Verdict: **CAN-PN-HIST-V1**, all mandatory
historical obligations pass. Details: `docs/pn-theorem.md`.

### 2. ρ-certification, including an honestly-reported first-run defect

`RHO-EXACT-CERT v0.1` was first run against frozen oracles and scored
`45/49`, failing 4 fixtures (`R2.01, R2.02, R3.01, R3.02`). The adjudication
found this was a **fixture/oracle defect, not a scientific refutation**: the
two "different-role" profiles the oracle expected were, on inspection,
genuinely causally isomorphic under an admissible recoding. That defect is
documented in full in `docs/rho-certification.md`, not silently dropped.
The same, unmodified candidate was then evaluated on a fresh
**candidate-fixed adversarial holdout** (32 fixtures across 8 families,
generated only after the candidate's hash was frozen) and scored
`32/32` — **CF-HOLDOUT-PASS**.

### 3. Target-boundary derivation, including the v0.1 refutation that motivated v0.2

`TARGET-BOUNDARY-EXACT v0.1`, based on interventional sufficiency alone,
was prospectively refuted `28/32` by a genuine counterexample: a functional
cancellation `A := Z, Y := Z ⊕ A` makes `Y` constant, so the v0.1 criterion
calls the empty boundary "sufficient" even though the causal edge `Z -> Y`
bypasses every eligible next-state variable. This refutation is kept in
full — it is the reason v0.2 requires a **structural cut** in addition to
interventional sufficiency. `TARGET-BOUNDARY-EXACT v0.2` then passed
`32/32` on its frozen prospective fixtures and `32/32` again on an
independent candidate-fixed adversarial holdout. Details, including both
candidate SHA-256 hashes: `docs/target-boundary.md`.

### 4. Composition of ρ and T, including an honestly-reported I3 fixture defect

Composing the two certified candidates (`R = B ∩ C_ρ*`, `H = B \ R` for
every minimal boundary and every maximal role partition), the first
integrated holdout scored `16/18`, failing `I3.01`/`I3.02`. Adjudication:
the fixture's RHO interface omitted the `a <-> b` transport certificate
needed to certify a genuine same-role clique between the two candidate
boundary nodes — a **fixture/oracle defect**, not a defect in either
candidate or in the composition rule, which in fact correctly propagated the
ambiguity the oracle had missed. A corrected integrated holdout, built with
both candidates and the composition rule already frozen, then scored
`18/18` — **POINT2-INTEGRATED-PASS**. Details: `docs/composition-and-closure.md`.

### Final status: CLOSE-CONDITIONAL

Point 2 — the canonicity of the P/N split and closure of the return
boundary — is **CLOSE-CONDITIONAL**, not a universal factorization theorem.
The closure depends explicitly on: an SCM/macro-graph reduced to the
minimal causal quotient; an intervention algebra rich enough for the
RHO-CERT/TARGET-BOUNDARY obligations; the PN hypotheses M1–M3; and the
abstention policy (`PN-V3` for multiple compatible boundaries/roles,
`PN-V4`/`NA` for an insufficient interface). No hidden tie-break is used
anywhere in the chain. Primary source: `docs/composition-and-closure.md`,
digesting `POINT2_CLOSURE_v0_1.md`.

## Layout

```
README.md                      this file
docs/
  pn-theorem.md                 the PN theorem and the CAN-PN v0.2 historical audit
  rho-certification.md          RHO-CERT protocol, the R2/R3 defect, the CF holdout
  target-boundary.md            TARGET-BOUNDARY v0.1 refutation and v0.2 result/holdout
  composition-and-closure.md    ρ+T composition, the I3 defect, and the CLOSE-CONDITIONAL verdict
scripts/
  rho_candidate_exact_certificate_v0_1.py      final ρ candidate (RHO-EXACT-CERT v0.1)
  rho_cert_harness_v0_1_1.py                   prospective harness (fixed v0.1.1 fixtures)
  rho_cert_holdout_cf_harness_v0_1.py          candidate-fixed holdout harness
  target_boundary_candidate_exact_v0_2.py      final T candidate (TARGET-BOUNDARY-EXACT v0.2)
  target_boundary_harness_v0_2.py              prospective harness (v0.2 fixtures)
  target_boundary_holdout_cf_harness_v0_1.py   candidate-fixed holdout harness
  point2_integrated_holdout_harness_v0_2.py    composition harness (loads both candidates above)
results/
  STATUS.json                                  final verdicts, exact pass counts, candidate hashes
  RHO_CERT_FIXTURES_v0_1_1.json
  RHO_CERT_HOLDOUT_CF_FIXTURES_v0_1.json
  TARGET_BOUNDARY_FIXTURES_v0_2.json
  TARGET_BOUNDARY_HOLDOUT_CF_FIXTURES_v0_1.json
  POINT2_INTEGRATED_HOLDOUT_FIXTURES_v0_2.json
```

## What was intentionally left out

- All `*_PRIVATE.json` oracle answer-key files. They record, per fixture,
  the expected verdict used for automatic scoring, and are withheld from
  this edition; the fixtures, the candidate code, the harness code, and the
  exact recorded pass/fail numbers (in the docs above and in
  `results/STATUS.json`) are all included, so every claim here can be
  checked against the actual candidate logic even though the harnesses
  cannot be run end-to-end without those private files.
- The **superseded** v0.1 TARGET-BOUNDARY fixtures/oracles and its
  harness/generator — dropped as stale duplicates once v0.2 replaced them;
  the refutation record that motivated v0.2 is kept in full
  (`docs/target-boundary.md`).
- The pre-fix `rho_cert_harness_v0_1.py`/`rho_cert_generator_v0_1.py` —
  superseded by the `_v0_1_1` versions kept here.
- The **superseded** v0.1 POINT2-INTEGRATED-HOLDOUT fixtures, oracles, and
  manifests — dropped once v0.2 replaced them after the I3 defect; the
  adjudication note describing that defect is kept in full
  (`docs/composition-and-closure.md`).
- Process bookkeeping: `MANIFEST.json` and the various pre-run/pre-candidate
  hash-freeze manifests. Their content (candidate SHA-256 hashes) is
  preserved where it matters, inline in the docs and in `results/STATUS.json`.
- `pn_neutralization_minimal_assumptions_v1_0.py` — shared with campaign
  `01-`; not duplicated here since no harness in this folder calls it
  directly.

Nothing that was refuted or that documents a defect was hidden: per the
shared reproduction discipline, only silently-superseded intermediate file
*duplicates* are dropped, never a refutation or an adjudicated defect.

## How to verify

Each harness in `scripts/` loads its candidate from the same directory and
its fixtures from `../results/`. With the fixtures included here but the
private oracles withheld (see above), a harness will get as far as scoring
against the fixtures and then fail to find the oracle file — this is
expected in this edition. What you can check directly:

```bash
# ρ candidate against the fixed v0.1.1 fixtures (needs RHO_CERT_ORACLES_v0_1_1_PRIVATE.json to score)
python3 scripts/rho_cert_harness_v0_1_1.py scripts/rho_candidate_exact_certificate_v0_1.py

# ρ candidate-fixed holdout (needs RHO_CERT_HOLDOUT_CF_ORACLES_v0_1_PRIVATE.json to score)
python3 scripts/rho_cert_holdout_cf_harness_v0_1.py scripts/rho_candidate_exact_certificate_v0_1.py

# T candidate against the v0.2 prospective fixtures (needs TARGET_BOUNDARY_ORACLES_v0_2_PRIVATE.json)
python3 scripts/target_boundary_harness_v0_2.py scripts/target_boundary_candidate_exact_v0_2.py

# T candidate-fixed holdout (needs TARGET_BOUNDARY_HOLDOUT_CF_ORACLES_v0_1_PRIVATE.json)
python3 scripts/target_boundary_holdout_cf_harness_v0_1.py scripts/target_boundary_candidate_exact_v0_2.py

# composition of both fixed candidates (needs POINT2_INTEGRATED_HOLDOUT_ORACLES_v0_2_PRIVATE.json)
python3 scripts/point2_integrated_holdout_harness_v0_2.py
```

Every exact pass count these commands were reported to produce when they
were run against the full (oracle-included) archive is recorded verbatim in
`docs/*.md` and in `results/STATUS.json`, together with the SHA-256 of every
frozen candidate, so the reported numbers can be checked against the visible
candidate logic without re-running the harness.

The scripts use only the Python standard library (`fractions`, `itertools`,
`json`, `pathlib`, `importlib`).

See the root `README.md` of the full reproduction package for the complete
protocol → manuscript-section table and further campaigns.
