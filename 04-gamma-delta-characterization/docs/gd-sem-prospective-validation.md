# GD-SEM: prospective campaign and candidate-fixed holdout

## Frozen semantic choice

Before any candidate was written, `Protocole_GAMMA_DELTA_SEM_v0.1_FROZEN.md` froze the
semantic choice that the axiom audits (see `gamma-axiom-characterization.md`,
`delta-decomposition.md`) left open:

- **Γ** keeps the current aggregator, `F_Γ(x,y) = (x+y)/2`, justified by the explicit
  marginal-independence axiom, under the **G-WEAK** interpretation (a purely one-directional
  influence still contributes positively to Γ). The directional balance
  `β = 2·min(x,y)/(x+y)` (`NA` if `x+y=0`) is reported separately for the minimizing cut(s);
  if several bipartitions tie for the minimum with different β, **all** minimizing β values
  are reported, with no tie-break.
- **Δ**'s primary object becomes the profile `(κ_q, δ_q)`, with the historical scalar
  retained as `Δ_cap = κ_q·δ_q`.

## Interface

Each fixture carries two independent blocks. **Gamma** — one or more admissible bipartitions,
each with two directions, each direction an exact input distribution and an exact
interventional kernel:

```json
"gamma_partitions": [
  {
    "id": "...",
    "ab": {"input_q": {"a0": "1/2", "a1": "1/2"},
           "kernel": {"a0": {"b0": "1", "b1": "0"}, "a1": {"b0": "0", "b1": "1"}}},
    "ba": { "...": "..." }
  }
]
```

The candidate computes `I(X;Y)` for each direction, then `Γ(π) = (x+y)/2`; the system score is
`Γ = min_π Γ(π)`. **Delta** — a declared support, exact `q(P)`, and exact kernel `K(Y|P)`:

```json
"delta": {
  "support": ["p0", "p1", "..."],
  "q": {"p0": "...", "...": "..."},
  "kernel": {"p0": {"y0": "...", "...": "..."}}
}
```

The candidate computes `H(P)`, `I(P;Y)`, `κ = H(P)/log₂|P*|`, `δ = I/H(P)`, `Δ = I/log₂|P*|`.
All labels are opaque strings; no oracle of role or formula is exposed. Scalars are compared
at `1e-12`; sets of minimizing β are sorted and compared elementwise at `1e-12`.

## Prospective families (16 families × 2 relabelings = 32 fixtures)

| Family | Construction | Expected |
|---|---|---|
| G0 | perfect bidirectional | `Γ=1, β=1` |
| G1 | perfect unidirectional | `Γ=1/2, β=0` |
| G2 | two 1/2-erasure channels (balanced, half information) | `Γ=1/2, β=1` (same Γ as G1, different β) |
| G3 | one strong cut, one weak cut | system Γ takes the weak cut exactly |
| G4 | two minimizing cuts, equal Γ, different β | both β values reported |
| G5 | bijective relabeling of a prior case | Γ, β invariant |
| D0 | uniform battery, perfect decoding | `(κ,δ,Δ)=(1,1,1)` |
| D1 | non-uniform battery, perfect decoding | `δ=1, κ<1, Δ=κ` |
| D2 | uniform battery, independent channel | `δ=0, Δ=0` |
| D3 | uniform 1/2-erasure | `δ=1/2, Δ=1/2` |
| D4 | non-uniform 1/2-erasure | `δ=1/2, Δ=κ/2` |
| D5 | bijective relabeling | profile invariant |
| IND1A/IND1B | equal Γ, unequal Δ (IND-1 witness) | — |
| IND2A/IND2B | equal Δ, unequal Γ (IND-2 witness) | — |

Actual run (`gamma_delta_sem_generator_v0_1.py` then
`gamma_delta_sem_harness_v0_1.py`):

```
CANDIDATE: GAMMA-DELTA-SEM 0.1
EXACT_PASSES: 32/32
G0: 2/2  G1: 2/2  G2: 2/2  G3: 2/2  G4: 2/2  G5: 2/2
D0: 2/2  D1: 2/2  D2: 2/2  D3: 2/2  D4: 2/2  D5: 2/2
IND1A: 2/2  IND1B: 2/2  IND2A: 2/2  IND2B: 2/2
FIRST_FAILURE: None
GD-SEM VERDICT: GD-V4 FINITE NON-REFUTATION
```

Sample confirmed values: `G1` → `Γ=0.5, β=[0.0]`; `G2` → `Γ=0.5, β=[1.0]`; `D1` →
`κ=0.875, δ=1.0, Δ=0.875`.

Candidate identity: `GAMMA-DELTA-SEM v0.1`,
SHA-256 `7681fe8c17c3bfd9e94d0e4b566e14254f718821dbea3ba2dae66334c21f26c6`.

**Verdict:** `GD-V4 FINITE NON-REFUTATION` — every fixture passes. This does not establish a
universal characterization beyond the frozen class and axioms; it is a finite non-refutation
of the frozen semantic choice.

## Candidate-fixed holdout (8 families × 4 relabelings = 32 fixtures)

The candidate was frozen (hash above) **before** the holdout fixtures existed; no modification
to the candidate is permitted between the two runs (`Protocole_GAMMA_DELTA_SEM_HOLDOUT_CF_v0.1_FROZEN.md`).
The holdout deliberately covers cases the prospective set did not:

| Family | What it adds |
|---|---|
| H0 | **three** minimizing cuts of equal Γ, distinct directional profiles |
| H1 | asymmetric rational binary channels |
| H2 | non-uniform directional input distributions |
| H3 | complete absence of directional influence |
| H4 | Delta on a non-uniform ternary support, perfect decoding |
| H5 | Delta with 1/3-erasure on a five-perturbation support |
| H6 | deterministic many-to-one Delta channel |
| H7 | combined independence witness outside the original families |

Actual run (`gamma_delta_sem_holdout_cf_generator_v0_1.py` then
`gamma_delta_sem_holdout_cf_harness_v0_1.py`, same unmodified candidate):

```
CANDIDATE: GAMMA-DELTA-SEM 0.1
EXACT_PASSES: 32/32
H0: 4/4  H1: 4/4  H2: 4/4  H3: 4/4  H4: 4/4  H5: 4/4  H6: 4/4  H7: 4/4
FIRST_FAILURE: None
GD-SEM VERDICT: GD-CF-HOLDOUT-PASS
```

**H0, the three-minimizing-cut witness**, confirmed exactly: three bipartitions —
one purely one-directional (`x=1,y=0`), one perfectly balanced 1/2-erasure
(`x=y=1/2`), and one asymmetric mix (`x=1/4,y=3/4`) — all achieve the identical
system-level `Γ = 0.5`, while their directional balances are the three distinct values

\[
\beta \in \{0.0,\; 0.5,\; 1.0\}.
\]

This is the concrete reason β is kept as a separate reported profile in the manuscript rather
than folded into the Γ scalar: at equal Γ, β can still take on every value its definition
allows.

**Verdict:** `GD-CF-HOLDOUT-PASS` — the frozen candidate reproduces exactly against an
independently generated, adversarially-designed fixture set.

## Scripts

- `scripts/gamma_delta_sem_generator_v0_1.py` — builds the 32 prospective fixtures + private
  oracle from the 16 declared families.
- `scripts/gamma_delta_sem_candidate_v0_1.py` — the frozen candidate (`GAMMA-DELTA-SEM v0.1`).
- `scripts/gamma_delta_sem_harness_v0_1.py` — scores a candidate against the prospective
  fixtures/oracle.
- `scripts/gamma_delta_sem_holdout_cf_generator_v0_1.py` — builds the 32 holdout fixtures +
  private oracle from the 8 declared H-families.
- `scripts/gamma_delta_sem_holdout_cf_harness_v0_1.py` — scores a candidate against the
  holdout fixtures/oracle.
