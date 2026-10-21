# FIBER-GAUGE-CLOSURE and FIBER-TRANSPORT: three equivalent faces of the missing structure

## The gauge group

Given a trivialization `φ:S→A×P` with `pr_P∘φ=π`, two trivializations of the same `π` differ by a
fiber-dependent gauge `φ_g(s)=(g_{π(s)}(a_φ(s)), π(s))` with `g_p∈Sym(A)` — the full gauge group
is `𝒢=∏_{p∈P} Sym(A)`. This transformation changes neither `S`, `K`, `q` nor `π`; it only changes
the correspondence declared between fibers.

## Theorem: the semantic class is closed under the gauge group

`Protocole_FIBER_GAUGE_CLOSURE_v0_1.md` asks whether the semantics already in use — `P` a
causally autonomous, append-only factor; `A` a successive (not accumulating) active coordinate;
arbitrary coupling of `P` into `A`'s update — pins down a trivialization by itself. Suppose
`p_{t+1}=G_u(p_t)`, `a_{t+1}=F_u(a_t,p_t)`. Under an arbitrary fiber-dependent gauge `g`, the
dynamics in the new gauge is

\[
p_{t+1}=G_u(p_t),
\qquad
a_{t+1}^g = g_{G_u(p_t)}\Big[F_u\big(g_{p_t}^{-1}(a_t^g),\, p_t\big)\Big].
\]

`FIBER_GAUGE_CLOSURE_theorem_v0_1.md` checks each defining property survives this substitution
unchanged: the factor `π` is literally the same map; `P`'s causal autonomy is preserved (its law
still depends only on `(P_t,U_t)`); the append-only regime on `P` is unchanged (the `G_u` are
identical); the active coordinate is still a pure *succession* (each transition replaces `a_t^g`
by a single new value, nothing is added to a growing store); and the coupling of `P` into `A`'s
update remains an arbitrary function of `(a_t^g,p_t,u_t)`. So:

\[
\boxed{\text{the class "successive active + autonomous append-only factor" is closed under }
\textstyle\prod_{p\in P}\mathrm{Sym}(A).}
\]

**Corollary.** If two gauges `φ,φ_g` satisfying the *same* declared semantic axioms give
`B_φ≠B_{φ_g}` on some system, those axioms do not determine a unique scalar:
`current axioms ⇏ B identified`. This is a no-go **relative to the declared semantics**, not
against every conceivable function of the full O1 data.

**Finite confirmation.** `scripts/fiber_gauge_closure_audit_v0_1_1.py` enumerates every orthogonal
gauge of the fixed `P` on the P4 fixtures, checks the semantic conditions exactly (product
bijectivity, `P`'s autonomy, well-definedness of the skew-product update), computes `B_g` for each
passing gauge, and reports a per-family verdict: `GC-U` (a single semantically valid gauge),
`GC-K` (several gauges, same kernel), `GC-B` (different kernels, same scalar `B`), or `GC-NONID`
(different kernels *and* different `B`). Running it on P4-A/P4-B/P4-C — all three families reach
**`GC-NONID`**, confirming the analytic result exactly, including on P4-C, the case OPF was
originally built to handle correctly: its 8 orthogonal gauges still give 8 different `B` values
(the OPF oracle happens to select the correct one, but nothing in the declared semantics forces
that choice).

## Theorem: trivialization ⟺ coherent fiber transport

`FIBER_TRANSPORT_equivalence_theorem_v0.1.md` proves, for a surjection `π:S→P` with equal-size
fibers, that two ways of supplying the missing structure are exactly equivalent (up to a global
relabeling of `A`):

- **(A) a trivialization** `φ:S→A×P` with `π=pr_P∘φ`;
- **(B) a coherent fiber transport**, a family of bijections `τ_{p→p'}:π⁻¹(p)→π⁻¹(p')` with
  `τ_{p→p}=id` and `τ_{p'→p''}∘τ_{p→p'}=τ_{p→p''}`.

`φ→τ`: set `τ_{p→p'}=φ⁻¹∘((a,p)↦(a,p'))∘φ`, identity and composition are immediate. `τ→φ`: fix a
reference fiber `p_0`, set `A=π⁻¹(p_0)`, `a(s)=τ_{π(s)→p_0}(s)`, `φ(s)=(a(s),π(s))`; coherence of
`τ` makes `φ` bijective with `φ⁻¹(a,p)=τ_{p_0→p}(a)`, and changing the reference fiber only
permutes the residual coordinate globally — a benign freedom the surgery kernel is invariant to
(a global permutation of `A` only renames conditional classes, never changes the kernel).

**What `π` alone gives:** the partition `{π⁻¹(p):p∈P}` — no privileged bijection between fibers.
P4-A and P4-B are exact finite witnesses where several coherent transports satisfy exactly the
same declared conditions and induce different `B`, so `π seul ne suffit pas`: transport must
either be derived by an independent rule, declared as typed structure in the causal context, or
the measure must be left unidentified.

## Theorem: fiber transport ⟺ intervention lift (the operational meaning)

`Protocole_FIBER_INTERVENTION_v0_1.md` / `FIBER_INTERVENTION_LIFT_theorem_v0_1.md` give the
missing structure its operational reading: "`do(P=p)` while keeping the same functional residual"
is exactly a family `(J_p)_{p∈P}`, `J_p:S→S`, satisfying

- **I1** (fiber fixing) `π(J_p(s))=p`;
- **I2** (identity on the already-fixed fiber) `π(s)=p ⟹ J_p(s)=s`;
- **I3** (coherent reassignment) `J_{p'}∘J_p=J_{p'}`.

This is provably equivalent to a coherent fiber transport (`τ_{p→p'}=J_{p'}|_{π⁻¹(p)}`) and hence
to a trivialization up to global relabeling. The intervention-free surgery kernel then has a
coordinate-free form: with horizontal class `[s]_J={J_p(s):p∈P}` and
`q_J(s'∣[s]_J)=q(s')/∑_{x∈[s]_J}q(x)`,

\[
\widetilde K_J(y\mid s,u) = \sum_{s'\in[s]_J} q_J(s'\mid[s]_J)\, K(y\mid s',u),
\]

which reduces exactly to "conserve `A`, randomize `P∣A`" inside any trivialization, with no
explicit labeling of `A` required. **Conceptual consequence:** "intervening on the memory `P` while
holding the rest fixed" is not defined by the quotient `π` alone — it requires precisely a
*lift of the intervention on the quotient to the full macro-state*. Fiber transport is not an
external accessory to the intervention algebra; it *is* its operational content whenever an
intervention targets a quotiented factor. Three levels must now be kept distinct: the persistent
factor `π:S→P`; the intervention lift `(J_p)_{p∈P}`; and the surgery/score `K̃_J, B_J`. If `π` is
identified but `J` is not, the scalar score is not identified — `J` must be derived from
independent causal structure, declared explicitly in the typed context, or the measure reported as
`NA`/partially identified. See `docs/partial-identification-policy.md` for the resulting formal
policy.

## Scripts

- `scripts/fiber_gauge_closure_audit_v0_1_1.py` — per-family `GC-*` verdicts on the P4 fixtures
  (depends on `opf_P4_fiber_complement_ambiguity_v0_1.py` in the same folder).
- FIBER-TRANSPORT and FIBER-INTERVENTION-LIFT are proved analytically; no separate audit script
  is shipped for them beyond the shared P4/IR10/P3-D fixtures used throughout this folder.
