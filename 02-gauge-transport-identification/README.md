# 02 — Gauge / transport identification

Clean edition of the campaign answering manuscript §8.2–§8.3, §18: even when the persistent
factor `P` is correctly and uniquely identified, a second, distinct question remains — which
active residual in one fiber of `P` corresponds to which active residual in another (i.e. "same
active state, different persistent value")? This folder proves `P` alone does not determine this
correspondence, characterizes exactly what kind of extra structure is needed, and develops a
formal policy for reporting the resulting identification honestly (point, partial, or NA) rather
than picking an arbitrary answer.

## Reproduction discipline

This campaign follows the same discipline as every other campaign in this repository (see the
root `README.md` for the full statement): a protocol is frozen and hashed *before* any candidate
is coded; a generator produces fixtures and a private oracle records the expected verdict for
each one; a candidate is written against the frozen fixtures; and where a result claims a
**prospective** test, the fixtures/oracles are frozen strictly after the candidate's hash. Verdicts
are reported as `V1/POINT` (a unique value), `V3/PARTIAL` (several values remain compatible,
reported as a set) or `V4/NA` (the intervention algebra abstains) — no result here silently turns
a partial or refuted finding into a clean single number.

## The discovery chain

### 1. OPF-P4 — the naive "orthogonal complement" gauge is refuted: 24/48

OPF v0.1 required a persistent factor `π:S→P` to have a causal complement `A` that is *itself* a
congruence, `S ≅ A×P`. P4 attacks this by testing a case where a persistent, causally-autonomous,
append-only `P` is nonetheless *read* by the active update (`A'` depends on `(A,P,U)`, not just
`(A,U)`) — so the "complement" projection is not itself a congruence. Result: **24/48** —
`P4-A` (bit read by the active update) 0/12, `P4-B` (2-bit G-set read by the active update) 0/12,
`P4-C` (the direct-product case OPF was built for) 12/12, `P4-D` (overwritable, non-append-only
register, negative control) 12/12. Verdict: **`P4-V3 — OPF v0.1 refuted prospectively`**
(`OPF_P4_RESULT_v0_1.md`). Relaxing "causal complement" to a merely *set-theoretic* orthogonal
complement does not fix this: P4-A has **2** orthogonal complements, P4-B has **8**, each inducing
a genuinely different surgery, with only one reproducing the frozen oracle in each case. See
`docs/opf-p4-gauge-ambiguity.md`.

### 2. FIBER-AUT — exact O1 symmetry never changes the score `B`

A sharper question: could an exact automorphism of the full O1 system exchange two gauge
trivializations that induce different `B`? An analytic theorem says no — two trivializations
related by an exact O1 automorphism preserving `q` and `π` are conjugate, and conditional mutual
information is bijection-invariant, so `B_σ1 = B_σ2` necessarily. An exhaustive audit over **256**
minimal 4-state extensions confirms this exactly: 8 systems have ≥2 admissible sections with
different `B`, and 0 of those 8 have two such sections inside one automorphism orbit. Verdict:
**`AUT-B-INV`**. So the real ambiguity lives **between** distinct O1 orbits of gauges, not inside
one. See `docs/fiber-aut-invariance.md`.

### 3. FIBER-GAUGE-CLOSURE / FIBER-TRANSPORT — what's missing is a fiber-transport structure

FIBER-GAUGE-CLOSURE proves analytically that the entire semantic class "active succession +
autonomous append-only factor" is closed under the full fiber gauge group
`∏_{p∈P} Sym(A)` — none of the axioms that are supposed to characterize the surgery rule it out,
so two gauges satisfying exactly the same declared semantics can still give different `B`
(confirmed on the P4 witnesses: **`GC-NONID`**). FIBER-TRANSPORT then proves an exact equivalence
theorem: a trivialization `S≅A×P` is formally equivalent (up to a global relabeling of `A`) to a
**coherent fiber-transport family** `τ_{p→p'}`. FIBER-INTERVENTION-LIFT adds a third equivalent
formulation as an intervention-lift family `(J_p)_{p∈P}` — this is the precise sense in which
"intervene on `P`, holding the rest fixed" is *not* defined by `π` alone; it requires exactly this
extra transport structure. See `docs/fiber-gauge-closure-and-transport.md`.

### 4. FIBER-PARTIAL-ID — a formal reporting policy, not a fix

`FIBER-STATUS-POST-AUT` documents that no elementary invariant (rank, fixed points, Green's
`L/R/J`, centralizer size, commutation with the vertical submonoid) selects the oracle gauge on
the P4 witnesses, that naive gauge-free scalarizations both fail (neither `min_φ B_φ` nor
`max_φ B_φ` is a general rule — the oracle is the min on some witnesses and the max on others),
and that a simple conditional-mutual-information substitute reproduces the IR10/P3-D oracles but
not P4-A/B/C. `FIBER-PARTIAL-ID` responds by formalizing three honest reporting statuses instead
of picking a winner: **`PI-POINT`** (`|B|=1`), **`PI-PARTIAL`** (report the exact set, or at
minimum `[B⁻,B⁺]` — no interval midpoint, no tie-break), and **`PI-EMPTY`** (`B=NA`). See
`docs/partial-identification-policy.md`.

### 5. JID-P5 / JID-P6 — transport IS constructible, conditional on a declared auxiliary signature

`JID-VERT` and `JID-CONJ` — two earlier attempts to derive the transport from vertical-monoid
conjugacy or equivariance — are abandoned without a general selection rule (not shipped in this
edition; kept in the source repository "for genealogy" only). `JID-P5` instead tests a
fundamentally different kind of input: a declared **auxiliary causal signature**
`c:S→C`, understood as the exact response of the system to a perturbation battery external to the
channel used to compute `B`, meant to identify "the same residual" across fibers. A theorem shows
that if `c` is injective within every fiber and every fiber shares the same signature set, it
induces a unique, well-defined intervention-lift family; partial-signature and
signature-incompatible cases degrade gracefully to `PI-PARTIAL` / `PI-EMPTY` respectively. The
prospective harness reaches **10/10** exactly across P5-A through P5-E. Verdict:
**`P5-V4 — finite non-refutation`**. `JID-P6` extends the same signature-identification test to
six families (`P6A`–`P6F`, 12 fixtures). See `docs/jid-p5-p6-auxiliary-signature.md` — **this
result is explicitly conditional**: it validates the identification *algorithm* given that such a
signature exists and is supplied; it does not show such a signature is derivable from O1 alone in
general.

## Final status

`IRP_FIBER_CURRENT_STATUS_v0_1.md` and `FIBER_STATUS_POST_AUT_v0_1.md` state the bottom line this
campaign establishes: **identifying `P` ≠ identifying transport ≠ computing the surgery.** These
are three genuinely separate problems that earlier work conflated. The minimal missing structure
is now characterized exactly (an intervention-lift family `(J_p)_{p∈P}`, equivalently a coherent
fiber-transport family, equivalently a trivialization up to global relabeling), and the honest
policy going forward is:

\[
B =
\begin{cases}
b & \text{if } J \text{ is point-identified (PI-POINT)} \\
\mathcal B \text{ or } [B^-,B^+] & \text{if partially identified (PI-PARTIAL)} \\
\mathrm{NA} & \text{if } \mathcal J=\varnothing \text{ (PI-EMPTY)}
\end{cases}
\]

`J` derived from a declared auxiliary causal signature is the one route shown here to reach
`PI-POINT` in a non-trivial case (JID-P5/P6) — but only conditionally on that signature being
available. Net conclusion carried forward to `03-canonical-boundary-closure/`: transport is either
derived from a declared auxiliary intervention algebra, or the framework reports the admissible
set (`PI-PARTIAL`) rather than choosing an element of it.

## How to verify

```bash
# OPF-P4: orthogonal-complement ambiguity (P4-A: 2 complements, P4-B: 8 complements)
python3 scripts/opf_P4_fiber_complement_ambiguity_v0_1.py

# FIBER-AUT: exhaustive 256-system automorphism-orbit audit
python3 scripts/fiber_aut_exhaustive_v0_1_1.py
# Expected: 256 systems; 8 with >=2 admissible sections; 0 with differing B inside one orbit -> AUT-B-INV

# FIBER-GAUGE-CLOSURE: per-family GC-* verdict on P4-A/B/C
python3 scripts/fiber_gauge_closure_audit_v0_1_1.py

# FIBER-PARTIAL-ID: PI-POINT/PARTIAL/EMPTY policy applied to IR10, P3-D, P4-A/B/C
python3 scripts/fiber_partial_identification_audit_v0_1.py

# JID-P5 prospective harness (private oracle not redistributed; see note below)
python3 scripts/jid_P5_harness_v0_1.py scripts/jid_candidate_signature_v0_1.py
```

The JID-P5 and JID-P6 private oracle files (`JID_P5_ORACLES_v0_1_PRIVATE.json`,
`JID_P6_FINITE_ORACLES_v0_1_PRIVATE.json`) and the OPF-P4 / historical private oracles are
intentionally **not** redistributed in this edition, for the same reason the source campaign kept
them private from the candidate under test. The exact recorded results (10/10 for P5, and the
P4/24/48 breakdown) are reproduced with hashes in the docs above. All scripts use only the Python
standard library (`fractions`, `itertools`, `json`, `importlib`).

See the root `README.md` for the full protocol → manuscript-section table.
