# 01 — Persistent vs. active factorization

Clean edition of the campaign answering manuscript §8.2–§8.3: given a causal model at the
macro-grain O1 alone (no hand-labeled "active"/"persistent" coordinates), is there a general,
coordinate-free extractor that splits a same-role state transition into a persistent factor `P`
(information that accumulates, never succeeding its own past) and an active residual (information
that succeeds it)?

Six increasingly careful candidates are tried in sequence. Each is retired by a specific,
preregistered counterexample, and the next candidate is a direct response to that counterexample.
The campaign does not end in a working extractor — it ends in an exact, preregistered collision
between two control families that a plain state-transition audit cannot resolve.

## Reproduction discipline

This campaign follows the same discipline as every other campaign in this repository (see the
root `README.md` for the full statement): a protocol is frozen and hashed *before* any candidate
is coded; a generator produces fixtures and a private oracle records the expected verdict for
each one; a candidate is written against the frozen fixtures; and where a result claims a
**prospective** test, the fixtures/oracles are frozen strictly after the candidate's hash, never
adjusted afterward. Verdicts are reported as `V1/POINT` (a unique value), `V3/PARTIAL` (several
values remain compatible, reported as a set) or `V4/NA` (the intervention algebra abstains) — no
result here silently turns a partial or refuted finding into a clean single number.

## The discovery chain

### 0. Full-reset and sync-factor (not included in this edition — refuted, superseded drafts)

The first two candidates against the IRP v0.1/v0.2 protocol drafts classified persistence by
automaton **synchronizability**: if some finite context word collapses the whole source
repertoire to one state (or a connected component of pairwise-synchronizable states), that
component's dependence on the current source is declared persistent and replaced by a
`q`-mixture surrogate. Both were refuted by the frozen controls IR9 (stochastic accumulator) and
IR10 (a single same-role variable mixing one active and one persistent bit after a coordinate
recoding) — synchronizability is neither necessary nor sufficient for accumulation. Their protocol
drafts (`Protocole_IRP_v0.1`, `v0.2`) and candidate code are dropped from this edition; the frozen
`v0.3` protocol below supersedes them and is what every later candidate is tested against.

### 1. AF v0.1 — append-only factor recognition

Splits the problem into *recognition* (which factor of the state is append-only?) before
*surgery*. A factor `h : S → L` is append-only if `L` carries a finite join-semilattice with a
bottom element such that the quotient transition is exactly `Lₜ₊₁ = Lₜ ∨ Aᵤ` for a context-indexed
increment law `Aᵤ` independent of the current value of `L`. AF v0.1 enumerates all such factors
exhaustively on alphabets ≤ 4. It correctly finds a full accumulator on IR7 and a non-trivial
append-only factor on IR10, but it **false-positives on IR11** (an active, many-to-one, non-
invertible — but not accumulating — transformation) and it **over-absorbs on IR10**: its maximal
append-only factor there carries more information than the intended single persistent bit. See
`docs/af-ts-ipf-factor-recognition.md`.

### 2. TS v0.1 — transformation-semigroup pivot

Moves the object of study from state factors to the semigroup of context-indexed transformations
itself, and extracts its largest semilattice quotient (`T_SL`) — the "irreversible, commutative,
idempotent-at-the-action-level" component of the dynamics. This is a diagnostic only; it does not
yet define a surgery or claim that a non-trivial `T_SL` means persistence.

### 3. IPF v0.1 — image-profile factor: 228/228 equivariant

Builds a canonical **state** factor from `T_SL` by taking, for each semilattice action class, the
union of its image, then partitioning states by their vector of memberships in these unions. This
construction needs no hidden decomposition and no choice. Result: it recovers the historical
persistent bit on IR10 exactly (`{0,2}∣{1,3}`), corrects the AF false positive on IR11 down to the
trivial partition, and is confirmed to be **exactly equivariant under all 228 exhaustive
state/context relabelings** of the deterministic controls IR7, IR8, IR10, IR11, IR12
(`irp_IPF_exhaustive_relabeling_audit_v0_1.py`). It still does not define a surgery. See
`docs/af-ts-ipf-factor-recognition.md`.

### 4. TSRS v0.1 — residual saturation surrogate: 220/220 regression, then refuted 24/40 prospectively

Defines a full surgery: saturate the state by the idempotent representatives of `T_SL`'s top
class, then randomize within the resulting residual class under `q` before reapplying the
original channel. Reproduces **220/220** historical deterministic oracles exactly (IR7, IR10,
IR11, IR12) and gives a new, non-trivial partial surgery on IR8 — but this is a post-hoc
regression, discovered by inspecting IR7–IR12, with no confirmatory value. A dedicated prospective
campaign, **TSRS-P1**, frozen after TSRS's own hash, refutes it: **24/40**, with the entire
failure concentrated in P1-A (**0/16**) — a pure cyclic-plus-reset active automaton with no
accumulating memory at all. TSRS misreads the reset's irreversible semilattice class as
accumulation. Verdict: **`P1-V3 — TSRS v0.1 refuted`**. See
`docs/tsrs-tsf-surrogate-and-automorphisms.md`.

### 5. TSF / TSF-AUT — the semilattice quotient alone does not select a unique state factor

TSF v0.1 asks whether `T_SL` canonically selects a *unique* faithful, TS-compatible state
partition. On IR10 it does not: **4 faithful partitions** exist (1 finest with 3 classes, 3
coarsest with 2 classes each), and the historical persistent bit `P` is only one of the three
coarsest. TSF-AUT then asks whether this is a real O1 symmetry problem: it enumerates the exact
automorphisms of IR10.01 (2 of them) and finds `P` is fixed by both — verdict **`AUT-U`**. So the
TSF ambiguity is not an O1 symmetry that would make `P` inherently unidentifiable; IPF's
construction happens to break the tie correctly on this control, but by a choice (image-profile
membership) that is not forced by `T_SL` compatibility alone. See
`docs/tsrs-tsf-surrogate-and-automorphisms.md`.

### 6. IR13-S1 — an exact O1 collision between two preregistered control families

The campaign's final, adversarial step searches exhaustively for a pair of O1-isomorphic systems
— one from a **stochastic two-slot storage** family, one from a **stochastic active automaton**
family driven purely by a context-indexed causal factor of the current state — whose frozen
oracles disagree under every isomorphism between them. The very first pair checked (out of 6
storage channels × 576 active constructions) is already such a witness: **8 exact O1
isomorphisms**, all **8 oracle-conflicting** (`B_storage_oracle = 0` vs. `B_active_oracle = 1`
bit). Verdict: **`S1-V1 — exact O1 collision with oracle conflict`**, conditional on both control
families being jointly legitimate functional controls. `ADJ-IR13` defines the adjudication
framework needed to resolve which oracle should yield (or whether the framework needs a typed
extension), but the campaign is explicitly left **blocked** at this point
(`"candidate3_status": "DO_NOT_CONSTRUCT_UNTIL_SEMANTIC_FORK_RESOLVED"` in
`results/IR13_S1_RESULT_MANIFEST_v0_1.json`) — no adjudication verdict was reached. See
`docs/ir13-o1-collision.md`.

## Final status and open gap

No single extractor tested is general. The last confirmed, non-refuted construction is IPF v0.1
(state-factor recognition only, no surgery); the last attempted surgery (TSRS v0.1) is refuted.
IR13-S1 shows the obstruction is not just "no candidate found yet" but an **exact** collision
between two preregistered, individually-motivated control families. The question is carried
forward to `02-gauge-transport-identification/` reframed away from *"find P"* and toward *"given
a declared P, what can still be said about the residual/transport structure."*

**Open gap, flagged explicitly:** the result manifest `IR13_S1_RESULT_MANIFEST_v0_1.json`
references a witness write-up, `IR13_S1_exact_witness_v0_1.md` (SHA-256
`f28b5d48691a8ed22aba68859434e0ea1f6f978368d692f4cad8ebeaf499f7c7`), that is **not present** in
the source repository. Its script counterpart, `irp_ir13_S1_first_witness_audit_v0_1.py`, and the
raw numbers in the manifest survive, and this edition's narrative above is built from those, but
the prose write-up of the witness itself is missing and has not been reconstructed here.

## How to verify

```bash
# AF -> TS -> IPF chain (all use the frozen public fixtures IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json)
python3 scripts/irp_append_only_factor_exhaustive_audit_v0_1.py
python3 scripts/irp_transformation_semigroup_audit_v0_1.py
python3 scripts/irp_IPF_exhaustive_relabeling_audit_v0_1.py
# Expected: TOTAL_RELABELLINGS_CHECKED: 228 ; exact finite equivariance on IR7/IR8/IR10/IR11/IR12

# TSF / TSF-AUT
python3 scripts/irp_TSF_action_to_state_factor_audit_v0_1.py
python3 scripts/irp_TSF_IR10_automorphism_audit_v0_1.py
# Expected on IR10.01: O1_self_automorphism_count: 2 ; VERDICT: AUT-U

# IR13-S1 exhaustive collision search (fully self-contained, no external fixtures)
python3 scripts/irp_ir13_stochastic_family_search_v0_1.py
# Expected: VERDICT: S1-V1 - exact O1 collision ...
```

The TSRS-P1 prospective harness (`scripts/tsrs_P1_harness_v0_1.py`) and the TSRS candidate
(`scripts/irp_candidate_TSRS_v0_1.py`) are included for inspection of the algorithm, but its
private oracle file (`TSRS_P1_ORACLES_v0_1_PRIVATE.json`) is intentionally **not** redistributed
in this edition — the same withholding the original campaign used to keep the candidate from
seeing its own scoring key. The exact reported result (24/40, 0/16 on P1-A) is recorded with
hashes in `docs/tsrs-tsf-surrogate-and-automorphisms.md`. All scripts use only the Python standard
library (`fractions`, `itertools`, `json`, `importlib`).

See the root `README.md` for the full protocol → manuscript-section table.
