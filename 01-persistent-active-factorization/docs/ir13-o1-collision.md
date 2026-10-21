# IR13-S1: an exact O1 collision between two control families

## The search

`Protocole_IR13_S1_v0_1_stochastic_collision_search.md` (frozen 2026-09-11, before execution)
searches exhaustively for a pair `(I_S, I_A)` such that: `I_S` belongs to a stochastic
**storage** family, `I_A` belongs to a stochastic **active automaton** family, their O1 reducts
are isomorphic under state/context relabeling, and their frozen surgery oracles are incompatible
under *every* such isomorphism. Common domain: `S,Y∈{0,1,2,3}`, `U∈{0,1}`, `q(S=s)=1/4`,
`q(U=u)=1/2`, all exact rationals.

**Family S — two-slot storage.** State `(b_0,b_1)∈{0,1}²`; each context refreshes one slot with a
fresh uniform bit while the other slot is conserved exactly (canonical case:
`U=0: Y=(R,b_1)`, `U=1: Y=(b_0,R)`), closed under context swap, slot swap, and all 24 global
state relabelings — **6 storage channels**. Oracle: neutralize dependence on the old storage,
`K̃_S(y∣s,u) = ∑_s̃ q(s̃) K_S(y∣s̃,u)`, so `I(S;Y∣U)_K̃=0` in this family.

**Family A — active automaton via contextual causal factor.** No "memory" coordinate is ever
supplied to the construction: for each context `u`, choose a balanced 2+2 partition `π_u` of the
source alphabet and a balanced 2+2 partition of the target alphabet, constrained so the pair
`(π_0(s),π_1(s))` distinguishes all four source states (and likewise for the two target
partitions — "transversality"), and set `K_A(y∣s,u)=1/2` for `y` in the target block selected by
`π_u(s)`, else 0 — **576 active constructions**. Oracle: preserve the channel exactly,
`K̃_A = K_A` (a genuine active same-role transformation, no primitive of persistent storage in its
specification).

Two members are **O1-isomorphic** if there exist bijections `σ` (states), `τ` (contexts) with
`K_B(σy∣σs,τu)=K_A(y∣s,u)` exactly for all `s,y,u` (the uniform `q`'s transport automatically). A
pair is a genuine **IR13-S1 witness** only if **every** O1 isomorphism between the two members
transports the oracles into conflict, `σ_*K̃_S ≠ K̃_A` — this rules out a false conflict from
picking a bad isomorphism.

## The result

`scripts/irp_ir13_stochastic_family_search_v0_1.py` runs the fully exhaustive, PRNG-free search
(no fixtures required — the families are generated in code). The **very first pair checked**
(index 0 of 6 storage channels against index 0 of 576 active constructions) is already a witness:

| Quantity | Value |
|---|---:|
| Storage channels enumerated | 6 |
| Active constructions enumerated | 576 |
| Pairs checked before first witness | 1 |
| Exact O1 isomorphisms of the witness pair | 8 |
| Isomorphisms with conflicting oracles | 8 (all of them) |
| `B` under the storage oracle | 0.0 bits |
| `B` under the storage's *original* (unsurgered) channel | 1.0 bits |
| `B` under the active oracle | 1.0 bits |

Verdict, recorded in `results/IR13_S1_RESULT_MANIFEST_v0_1.json`:

> **`S1-V1 — exact O1 collision with oracle conflict`**

The claim status is stated precisely in the manifest: *"exact local non-identifiability
conditional on joint legitimacy of both preregistered control families"*. That is, if both the
storage family's "neutralize old storage" oracle and the active family's "preserve the channel"
oracle are accepted as simultaneously legitimate functional controls, then no O1-readable
surgery function can satisfy both on this pair — one instance of the same O1 system is being
asked, by two independently-motivated control families, to receive two different, exactly
conflicting surgeries.

## ADJ-IR13: the adjudication framework (not yet resolved)

`Protocole_ADJ_IR13_v0_1.md` sets out how such a collision should be adjudicated *within the
current framework*, without proposing a new IRP candidate. It ranks manuscript commitments —
definitions (e.g. the causal-equivalence principle) outrank estimator-level operationalizations,
which outrank illustrations — and states four engagements:

- **E1 (causal equivalence)** — O1-isomorphic contexts cannot receive surgeries that differ in
  `R` unless a definitional-level structure is explicitly added.
- **E2 (functional identity)** — a distinction that disappears under an O1 isomorphism cannot be
  reintroduced by internal-coordinate naming alone.
- **E3 (the current definition of persistent accumulation)** — succession (new value replaces
  old in the same active role) versus accumulation (new value is added without replacing; all
  prior values remain simultaneously active members of a growing memory) — a stronger clause than
  "some old information is still predictive" or "the channel is non-invertible".
- **E4 (open status of the factorization)** — §18 already concedes P/N is not yet derived from
  dynamics alone; this permits an identification lock to exist, but does not license a declared
  factorization that violates E1 without announcing a revision to causal equivalence.

Verdict rules: **`ADJ-V1`** (true internal no-go: both diagnostics are independently required by
definitions of equal rank, no priority resolves it), **`ADJ-V2`** (an estimator-level oracle is
rejected/reclassified because a definition-level principle — causal equivalence — forces
equality), **`ADJ-V3`** (keeping both diagnoses requires an explicit typed extension to the
functional context and a restriction of causal equivalence — a revision, not a derivation).

**No ADJ verdict is recorded for the S1 witness.** The result manifest explicitly marks the next
step as blocked: `"candidate3_status": "DO_NOT_CONSTRUCT_UNTIL_SEMANTIC_FORK_RESOLVED"`. This is
the deliberate stopping point of the `01-` campaign, not an accidental gap — see the folder
`README.md`'s "Final status and open gap" section for what *is* missing from the source repository
(the witness write-up `IR13_S1_exact_witness_v0_1.md`) versus what is simply left open by design
(the ADJ-IR13 adjudication itself).

## Scripts

- `scripts/irp_ir13_stochastic_family_search_v0_1.py` — the exhaustive search; self-contained,
  no fixtures needed.

## Results

- `results/IR13_S1_RESULT_MANIFEST_v0_1.json` — frozen manifest with the exact numbers above,
  file hashes, and the blocked `candidate3_status`.
