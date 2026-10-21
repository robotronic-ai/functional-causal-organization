# FIBER-AUT: exact O1 symmetry cannot change the score

## Question

After a persistent factor `π:S→P` and a monoid projection `ρ:T→B` (from the transition monoid `T`
onto its action on `P`) are fixed, several homomorphic sections `σ:B→T` can induce different
surgeries. `Protocole_FIBER_AUT_v0_1.md` tests a sharper no-go: **can two sections with different
scores be exchanged by an exact automorphism of the full O1 system that preserves `q` and `π`?**
If so, the ambiguity would be a genuine O1 symmetry problem, not just a matter of an
under-specified rule.

The protocol records an analytic expectation *before* running the audit: if an automorphism
transports section `σ_1` to `σ_2`, the equivariantly-constructed surgery should be conjugated by
the same automorphism, and since conditional mutual information is invariant under variable
bijections, `B_σ1 = B_σ2` is expected. The exhaustive audit's role is to test the
*implementation* of this consequence and map out section orbits — not to look for a violation of
the theorem itself.

## Exhaustive domain

Minimal case: `S={0,1,2,3}` with the fixed binary factor `π⁻¹(0)={0,2}`, `π⁻¹(1)={1,3}`, two
contexts `U={0,1}` where context 0 acts as the identity on `P` and context 1 as `p↦1`. For each
source state, the coordinate internal to its target fiber can be chosen in 2 ways per context —
`2⁴×2⁴=256` deterministic controlled extensions in total, uniform `q` throughout. For each system:
build the transition monoid `T`, its quotient `B` on `P`, enumerate every homomorphic section
`σ:B→T` (retaining those whose transport from the low fiber is bijective), and for each admissible
section construct the surgery (`P̃∼q(P∣A_σ)`, apply the original channel) and compute
`B_σ=I_q(S;Ỹ∣U)` exactly.

## Theorem (`FIBER_AUT_invariance_theorem_v0_1.md`)

Given `q_S`- and `q_U`-preserving bijections `β:S→S`, `κ:U→U` with `π(βs)=π(s)`, the original
channel invariant (`K(βy∣βs,κu)=K(y∣s,u)`), and `σ_2` the `β`-conjugate transport of `σ_1`, then

\[
\widetilde K_{\sigma_2}(\beta y\mid\beta s,\kappa u) = \widetilde K_{\sigma_1}(y\mid s,u)
\quad\Longrightarrow\quad
\boxed{B_{\sigma_1}=B_{\sigma_2}.}
\]

*Proof sketch:* an admissible section defines a fibral trivialization; conjugation by `β`
transports it to the trivialization of the transported section, and the conditional distribution
used to randomize `P` at fixed fiber coordinate is transported by `β` since `q_S` and `π` are
preserved. The original channel commutes with `(β,κ)` by hypothesis, so the surgery kernels are
conjugate; the simultaneous relabeling `S↦β(S)`, `Y↦β(Y)`, `U↦κ(U)` is bijective and law-preserving,
so conditional mutual information is unchanged. **Consequence:** a search for "an exact
automorphism exchanges two sections with different `B`" cannot succeed while these hypotheses
hold — the real locus of under-determination must be sought *between* distinct section orbits, or
in extra structure not contained in O1.

## Exhaustive audit result (`FIBER_STATUS_POST_AUT_v0_1.md`)

`scripts/fiber_aut_exhaustive_v0_1_1.py` (self-contained, no external fixtures):

| Quantity | Count |
|---|---:|
| Systems examined | 256 |
| Systems with ≥2 admissible sections | 8 |
| Of those, with differing section scores `B` | 8 (all of them) |
| Of those, with two such sections inside one automorphism orbit | 0 |
| ⇒ systems exhibiting an in-orbit `B` difference | 0 |

Verdict: **`AUT-B-INV`**. The no-go sought by exact symmetry at the scalar level is impossible
under the theorem's hypotheses, confirming it exactly on the exhaustive minimal domain.

*Implementation note preserved from the source:* the first frozen `v0.1` script failed before any
scientific result on a scope bug in the binary-quotient composition; the `v0.1.1` correction
changes only the composition to iterate over the transformation's proper domain — protocol and
theorem are unchanged, only the code defect is fixed.

## Scope and what remains open

`AUT-B-INV` does **not** resolve the FIBER problem. Sections in *different* automorphism orbits
can still produce different `B` — exactly what P4-A/P4-B exhibit
(`docs/opf-p4-gauge-ambiguity.md`). O1 can, in principle, distinguish sections structurally across
orbits; what is still missing is either a selection axiom or an explicit declaration that the
measure is not identified. `FIBER-STATUS-POST-AUT` additionally records that a related but
distinct diagnostic, **FIBER-SECTION**, is useful but not a general definition of fiber transport:
on control P3-D (a many-to-one active dynamics that can preserve a coordinate identity between
fibers without that transport being realizable as an internal bijective transformation belonging
to `T`), no homomorphic, fibrally-bijective section exists in the transition monoid at all — so a
section-based formalism cannot even be stated there, even though the underlying transport question
is still meaningful.

## Scripts

- `scripts/fiber_aut_exhaustive_v0_1_1.py` — the exhaustive 256-system automorphism-orbit audit.
