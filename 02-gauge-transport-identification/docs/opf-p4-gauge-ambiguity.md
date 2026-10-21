# OPF-P4: the naive complement requirement is refuted, and gauge ambiguity is real

## The attacked hypothesis

OPF v0.1 (candidate `IRP-OPF v0.1`, SHA-256
`a8e7b940a687a761628a3d8c357f519b4be9a0d7e1bf2b2835d8bc67eeb4b65a`) requires a persistent
append-only factor `P` to have a causal complement `A` that is *itself* a congruence, giving a
clean product `S ≅ A×P`. This may be too strong: a persistent memory can be a fully autonomous
causal factor while the active computation *reads* that memory — then `P'` depends only on
`(P,U)`, but `A'` depends on `(A,P,U)`, so the naive projection onto `A` is not an autonomous
quotient. `Protocole_OPF_P4_v0_1_PROSPECTIVE.md` (frozen 2026-09-11, before execution) tests
exactly this.

## The four families

| Family | Construction | Oracle |
|---|---|---|
| **P4-A** | `S=(A,P)∈{0,1}²`; `P'=P∨U`; `A'=A⊕P⊕U` (P strictly append-only and autonomous, but read by the active update) | Preserve `A`, draw `P̃∼q(P∣A)`, reconstruct `(A,P̃)`, apply the original channel |
| **P4-B** | `S=(A,H)`, `A∈{0,1}`, `H⊆{0,1}` (a 2-bit G-set); `H'=H∪m_U`; `A'=A⊕parity(H)⊕[U≠0]`; full-support `q` correlating `A` and `H` | Same recipe: preserve `A`, draw `H̃∼q(H∣A)`, apply the original channel |
| **P4-C** | Same `(A,H)` and G-set as P4-B, but `A'=A⊕1` independent of `H` — the direct-product case OPF is explicitly designed to recognize | Same recipe (here the complement genuinely is a congruence) |
| **P4-D** | `S=(A,R)∈{0,1}²`, overwritable (not append-only) register: `u=0: R'=R`; `u=1: R'=0`; `u=2: R'=1`; always `A'=A⊕R⊕(U mod 2)` — negative control | Preserve the original channel unchanged |

Each family is submitted to 3 preregistered state recodings (identity, reversal, one-step
rotation) and scored under 4 frozen interface anonymization seeds (`1729, 2718, 31415, 65537`) by
exact full-kernel equality — 12 fixtures × 4 seeds = 48 runs total.

## Result: 24/48

From `OPF_P4_RESULT_v0_1.md`:

| Family | Passes | Interpretation |
|---|---:|---|
| P4-A | 0/12 | append-only `P` exists with no orthogonal causal complement |
| P4-B | 0/12 | same failure mode at larger scale |
| P4-C | 12/12 | the direct-product case OPF was built for still works |
| P4-D | 12/12 | an overwritable register is correctly *not* classified append-only |
| **Total** | **24/48** | |

Verdict: **`P4-V3 — OPF v0.1 refuted prospectively`**. P4-A/B specifically refute "the active
complement must itself be a congruence" — not the existence of the append-only factor itself; P4-C
confirms the campaign does not destroy the case OPF handles correctly, and P4-D confirms an
overwritable register is not falsely classified as append-only.

## A stronger post-P4 result: relaxing to a set-theoretic complement doesn't help

`OPF_fiber_gauge_no_go_v0_1.md` shows that weakening "causal complement" (a congruence) to a
merely *set-theoretic orthogonal complement* — any partition `A` of `S` such that `(a,p)↦s` is a
bijection with the fixed persistent partition `P` — does not resolve the ambiguity; it reveals a
deeper one. `scripts/opf_P4_fiber_complement_ambiguity_v0_1.py` enumerates every orthogonal
complement of the fixed `P` exhaustively and computes the surgery kernel each induces:

| Family | `P` | Orthogonal complements | Distinct induced surgeries |
|---|---|---:|---:|
| P4-A | `{0,2}∣{1,3}` | 2 | 2 |
| P4-B | `{0,4}∣{1,5}∣{2,6}∣{3,7}` | 8 | 8 |

In both cases exactly **one** complement reproduces the frozen oracle; every other orthogonal
complement — an equally valid trivialization of the *same* fixed `P` — gives a genuinely
different surgery kernel. This is a **fiber gauge non-identifiability theorem**, proved
analytically:

> If two admissible trivializations `φ, ψ` of the same factor `π` give `K̃_φ ≠ K̃_ψ`, no
> deterministic functional `F=F(K,q,π)` invariant under the gauge transformation can equal both
> simultaneously. *Proof:* `φ,ψ` describe exactly the same data `(K,q,π)`, differing only by a
> fiber-dependent internal permutation, so any functional of `(K,q,π)` must agree on both — but
> the requested outputs differ. Contradiction.

So the problem is not that OPF's congruence requirement was too strict; it is that **a canonical
persistent factor, even perfectly identified, does not by itself fix "which residual value in one
fiber is the same active state as which residual value in another."** This correspondence is not
contained in `π`. Three theoretical options remain open at this point: enrich the causal context
with a fiber transport / typed active–persistent decomposition; find a new surgery genuinely
invariant under the whole gauge group `∏_p Sym(A)`; or abstain from partial neutralization when no
canonical connection is derivable. These are developed in
`docs/fiber-gauge-closure-and-transport.md` and `docs/partial-identification-policy.md`.

## Scripts

- `scripts/opf_P4_fiber_complement_ambiguity_v0_1.py` — post-P4 exact audit of orthogonal-
  complement ambiguity (also a dependency of the FIBER-GAUGE-CLOSURE and FIBER-PARTIAL-ID audits).
