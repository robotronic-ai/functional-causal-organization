# AF → TS → IPF: from factor recognition to an equivariant construction

Three protocols, read in sequence. None of them yet defines a surgery on their own (that is
TSRS, see `docs/tsrs-tsf-surrogate-and-automorphisms.md`) — this is the *recognition* half of the
problem: which factor of the macro-state is even a candidate for "the persistent part"?

## AF v0.1 — append-only factor recognition

A factor `h : S → L` is **causally admissible** if the induced process is strongly lumpable for
every context (the quotiented transition kernel `∑_{y:h(y)=ℓ'} K_u(y∣s)` depends on `s` only
through `ℓ=h(s)`), giving an exact quotient kernel `Q_u(ℓ'∣ℓ)`.

It is **append-only** if `L` carries a finite join-semilattice `(L,∨)` with bottom `⊥`, and for
each context `u` a distribution `μ_u` on `L`, such that

\[
Q_u(\ell'\mid\ell) = \Pr_{A\sim\mu_u}[\ell\vee A=\ell'],
\]

i.e. `L_{t+1} = L_t ∨ A_u` for a context-indexed, possibly stochastic increment whose law does not
read the current value of `L`. Since `⊥∨A=A`, `μ_u` is pinned down exactly by the row leaving
`⊥`: `μ_u(a) = Q_u(a∣⊥)` — no free parameter remains once the semilattice is chosen.

A factor is **informative** for IRP if `I_q(h(S); h(Y) ∣ U) > 0` (a factor that saturates
instantly to a constant transports no measurable persistence). Factors are ordered by refinement,
and `h_1 ⪰ h_2` if `h_2 = φ∘h_1`.

Verdicts: `AF-U1` (unique maximal informative factor), `AF-MULTI` (several incomparable maximal
informative factors), `AF-NONE` (no non-trivial append-only factor), `AF-SAT` (an append-only
factor exists but none is source-informative). The audit
(`scripts/irp_append_only_factor_exhaustive_audit_v0_1.py`) is exhaustive on alphabets ≤ 4:
every partition of the source alphabet is enumerated, every binary relation on ≤4 elements is
tested for being a partial order with bottom and pairwise joins, and increment laws are checked
exactly in `Fraction` arithmetic — no PRNG.

**Two failure modes**, both found by inspecting IR7–IR12 rather than a blind holdout:

- **False positive on IR11.** IR11's two active transformations are many-to-one and
  non-invertible but genuinely active (no accumulating memory). AF still finds a non-trivial,
  informative append-only factor there — non-invertibility alone can satisfy the append-only
  definition without any real accumulation. (Confirmed corrected in `IRP_IPF_v0_1_result_note.md`
  §4: "the AF v0.1 false positive disappears entirely" once IPF's `T_SL`-derived construction
  gives IR11 the trivial partition and `I_IPF = 0`.)
- **Over-absorption on IR10.** IR10's maximal append-only factor carries *more* information than
  the single intended persistent bit — a static invariant can look append-only without being the
  right grain of "the" persistent factor.

Both failures motivate moving the object of analysis from state factors to the dynamics itself.

## TS v0.1 — the transformation-semigroup pivot

For a deterministic control, every context `u` defines a transformation `f_u : S → S`. Their
closure under composition is a finite transformation semigroup `T = ⟨f_u : u∈U⟩`
(`scripts/irp_transformation_semigroup_audit_v0_1.py` builds the multiplication table by
saturating closure exactly).

Define `θ_SL` as the smallest semigroup congruence containing `x ~ x²` and `xy ~ yx` for all
`x,y∈T`. The quotient `T_SL = T/θ_SL` is then the largest homomorphic image of `T` satisfying the
semilattice identities `z²=z`, `zw=wz` — the audit computes it by disjoint-set closure over the
multiplication table. This construction is purely about the **transformations**, not the state
coordinates, and TS v0.1 is explicitly only a structural diagnostic: a non-trivial `T_SL` is *not*
yet claimed to mean "this is the persistent memory to neutralize". It records, for each of IR7,
IR8, IR10, IR11, IR12: `|T|`, `|T_SL|`, which class each context generator falls in, and whether
`T` contains non-trivial permutation elements.

## IPF v0.1 — image-profile factor: an equivariant, canonical state factor

For each `T_SL` class, take the union of the images of its member transformations,
`J_c = ⋃_{i∈c} image(f_i)`. Profile each state `s` by its membership vector
`(𝟙[s∈J_c])_c` across classes, and partition states by identical profile
(`scripts/irp_image_profile_factor_audit_v0_1.py`). This needs no hidden decomposition, no
threshold, and no PRNG.

**Exact results** (`IRP_IPF_v0_1_result_note.md`):

| Control | IPF partition | `I_q(χ(S); χ(Y) ∣ U)` |
|---|---|---:|
| IR7 | `{0}\|{1}\|{2}\|{3}` | 1 |
| IR8 | `{0,2}\|{1,3}` | 0.5 |
| IR10 | `{0,2}\|{1,3}` | 0.5 |
| IR11 | `{0,1,2,3}` | 0 |
| IR12 | `{0}\|{1,2}` | 0.306098611351… |

- **IR10** recovers the historical persistent bit `{0,2}∣{1,3}` exactly, without ever reading the
  hidden `(A,P)` decomposition — this is the central positive result of the chain.
- **IR11** collapses to the single trivial class (`|T_SL|=1`), giving `I_IPF=0`: the AF false
  positive is gone.
- **IR12** recovers `{0}∣{1,2}` — two different concrete capacity-1 overwrite patterns land in the
  same abstract irreversible-action class, giving exactly "a write has occurred" as the candidate
  partial-surgery factor.
- **IR8** is only partially absorbed (`{0,2}∣{1,3}`, not the whole state) — correcting AF's
  earlier over-broad "the whole G-set is accumulation" reading, but also showing the coarser
  claim "nothing is accumulation" would have been equally wrong.
- **IR7** recovers the full cumulative state exactly: the three action classes' image unions,
  `{1,3}`, `{2,3}`, `{3}`, distinguish all four states `000, 100, 010, 111`.

**Exact equivariance.** `scripts/irp_IPF_exhaustive_relabeling_audit_v0_1.py` recomputes
`TS → T_SL → IPF` under **every** state and context relabeling of IR7, IR8, IR10, IR11, IR12 and
checks that the recovered partition equals the transported original partition exactly:

| Family | Relabelings checked | Result |
|---|---:|---|
| IR7 | 24 × 4-state perms × 2-ctx perms = 48 | PASS |
| IR8 | 48 | PASS |
| IR10 | 48 | PASS |
| IR11 | 48 | PASS |
| IR12 | 6-state perms × 6-ctx perms = 36 | PASS |
| **Total** | **228** | exact finite equivariance |

IPF v0.1 still does not define a surgery — it fixes the canonical persistent-factor grain,
`χ : S → {0,1}^{|T_SL|}`, and leaves `K̃(Y∣S,U)` (a construction that neutralizes only `χ(S)`'s
accumulated information while preserving the active residual) to the next step. See
`docs/tsrs-tsf-surrogate-and-automorphisms.md`.
