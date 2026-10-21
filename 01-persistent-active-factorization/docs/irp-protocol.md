# The IRP protocol (frozen v0.3)

`Protocole_IRP_v0_3_FROZEN.md` is the frozen specification every candidate extractor in this
campaign is tested against. It went through three freeze passes before the first candidate was
written; each pass is preserved below because the later ones correct real interface gaps found
*before* any candidate existed, not after a refutation.

## 1. What a candidate must return

A same-role transition is described, without ever naming a "persistent" or "active" component, by
a context

\[
\mathfrak I = (\mathcal M^*, q, \rho, T_t, T_{t+1}, U_t)
\]

where `M*` is the SCM at the minimal causal macro-grain, `q` is the intervention battery, `ρ` is a
certified role relation, `T_t`/`T_{t+1}` are the untyped boundaries of successive macro-states, and
`U_t` is the declared conditioning context.

A candidate `Ψ_IRP` does **not** return a P/N split or even necessarily a set of edges to cut. A
single binary edge-cut can only keep or discard *all* information crossing a transition, which is
too coarse once persistent and active information can share one same-role variable (control IR10,
below). Instead, a candidate returns a **channel surgery** — an emulation kernel

\[
\Psi_{\rm IRP}(\mathfrak I) = \widetilde K(T_{t+1} \mid T_t, U_t)
\]

as an exact rational table over the declared repertoires. The post-surgery return quantity is

\[
B_{\rm IRP} = I_q^{do}\big(Z_t^*;\, R_{t+1}^{\rho^*} \mid U_t\big)_{\widetilde{\mathcal M}},
\qquad
R_{t+1}^{\rho^*} = \{X \in T_{t+1} : \rho(X) = \rho(Z_t^*)\}.
\]

## 2. Axioms IR-A1–IR-A7 (constraints on any surgery)

- **IR-A1 O1-readability** — `Ψ` may depend only on ordinary kernels, the quotient graph, the
  temporal boundary, `q`, certified `ρ`, and `U_t`. No dependence on a constructor label such as
  "active", "persistent", "replacement" or "accumulation" is allowed.
- **IR-A2 q-preservation** — any surrogate/copy introduced by the surgery uses a joint kernel
  independent of the current label, in the sense of control F9.
- **IR-A3 coordinate invariance** — for every admissible bijection `g` of the macro-state,
  `Ψ(gM) ~ gΨ(M)` (equality of emulation kernels after transport). No order, distance, bit
  significance or set-inclusion structure is available unless it is itself causally identifiable.
- **IR-A4 representation invariance** — faithful mediator insertion, deterministic duplication,
  causally-inert readers, and reconstructible representations must not change `B_IRP`.
- **IR-A5 no penalty on active identity** — a genuine active succession can be the identity
  (`X_{t+1} = X_t`); literal bit-for-bit conservation is not by itself evidence of persistence.
- **IR-A6 no purely marginal criterion** — a candidate that decides per-coordinate from
  `I(Z; X_i ∣ U)` alone is incomplete; the relevant dependence can be synergistic.
- **IR-A7 temporality** — classification concerns a declared functional transition and must be
  compatible with unrolling/folding and with control F12 for multi-step extrapolation.

## 3. The frozen control suite

**Regression controls carried over from earlier campaigns** (not blind holdouts, but not to be
silently patched either): F13 (same-role cumulative memory `H_{t+1}=H_t ∨ (1≪U_t)`; requires
`B_IRP = 0` invariantly under all 24 bijections of the 4-state alphabet), PN-MIX, XOR/PN3, F2, F5.

**IR0–IR6 — basic requirements**, in order: F13 as IR0 (`B_IRP=0`); IR1 active identity
(`B_IRP = H_q(X_t∣U_t)` in the noiseless case — copying is not automatically persistence); IR2
active bijective transformation; IR3 XOR/synergy (marginals alone must not license "no
persistence"); IR4 PN-MIX coordinate-mixing bijections (invariance of `B_IRP` and the abstract
emulation kernel); IR5 mediator insertion/removal; IR6 duplication/reader/reconstructible
representation.

**IR7–IR12 — adversarial families, frozen before any candidate, 59 fixtures total** (generator
`irp_generators_IR7_IR12_v0_1.py`, public fixtures
`IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json`, private oracles not redistributed in this edition):

| Family | Count | What it tests | Frozen control value |
|---|---:|---|---|
| IR7 | 24 | F13 transported under all 24 exhaustive 4-state relabelings — no bitwise/numeric/ordinal reading of "monotonicity" is allowed | `B_original=1`, `B_oracle=0` |
| IR8 | 1 | Active idempotent same-role semigroup (projection vs. identity, by context) — idempotence alone is not persistence | `B_original=B_oracle=1.5` |
| IR9 | 3 | Stochastic accumulator, each old bit independently survives with `p ∈ {1/4,1/2,3/4}` — no ad hoc threshold behavior | `B_original ∈ [0.1379…, 0.5488…]`, `B_oracle=0` |
| IR10 | 24 | A hidden `(A,P)` state — `A'=A⊕U`, `P'=P∨U` — recoded under all 24 relabelings; the central test: persistent and active information share **one** same-role variable, so a total edge-cut must fail | `B_original=1.5`, `B_oracle=1` |
| IR11 | 1 | Two non-idempotent, many-to-one active transformations — non-invertibility is not persistence | `B_original=B_oracle=1.5` |
| IR12 | 6 | Capacity-1 memory (`H'=H` if `U=0`, else `H'=U`) under all 6 relabelings — no reliance on strict monotone growth of the memorized set | `B_original=0.5283…`, `B_oracle=0` |

**IR13 — the non-identifiability target**: search for two contexts `I_A, I_B` with identical
minimal macro-state, `q`, certified `ρ`, accessible O1 kernels and state boundary, but whose
functional controls impose *different* intra-role surgeries. An exact such pair means IRP is not
identifiable from O1 alone at that grain. See `docs/ir13-o1-collision.md` for the S1 witness.

## 4. Verdicts

- **IRP-V1** — construction theorem: a surgery is defined with an analytic proof it satisfies the
  axioms on the declared domain.
- **IRP-V2** — exact non-identifiability: an IR13 witness shows two O1-indistinguishable contexts
  requiring different surgeries.
- **IRP-V3** — candidate refuted: a frozen control fails.
- **IRP-V4** — finite non-refutation: the candidate passes the finite campaign. Must be stated as
  "not refuted on the declared IRP domain", never as "intra-role persistence identified".
- **IRP-V5** — reduction to an existing primitive: the surgery is proved fully derivable from `ρ`,
  the minimal quotient, `q`, or another already-justified primitive (F13 rules out naive reduction
  to `ρ` alone).

## 5. The v0.2 and v0.3 freeze addenda

**v0.2** (2026-09-11) froze the executable objects before any candidate existed: the candidate
interface `irp_interface_v0_1.py`, its self-audit, the IR7–IR12 generator and fixtures, and the
private oracles — all hashed, with the interface exposing `source_state()`, `target_state()`,
`context_state()`, `roles()`, a fraction-weighted `battery()`, and an ordinary `kernel(do,
targets)`, but never a `pn_split()` and never any family/oracle metadata.

**v0.3 erratum** (same day, before the first candidate): a genuine interface debt was found —
`v0.1` exposed no certified bijection carrying a `target_state()` value at `t+1` to the
corresponding `source_state()` value for the next transition, so any multi-step method risked
illicitly assuming the two codes coincide (forbidden by IR-A3). The corrected interface,
`irp_interface_v0_1_1.py`, adds

```python
def temporal_transport(self) -> Mapping[ValueTuple, ValueTuple]: ...
```

a certified bijection between the full target and source repertoires — structural O1/`ρ`
information, not a P/N label. Its validator (`validate_temporal_transport`, shipped in the same
file) checks full coverage, injectivity and surjectivity. The runner
(`irp_harness_v0_1.py`, not shipped separately in this edition — its role is folded into each
audit script below) anonymizes variable ids, source/target/context value codes, and role ids
independently under four frozen seeds `(1729, 2718, 31415, 65537)`; a fixture passes only under
**exact equality of every rational row** of the emulation kernel, not just equality of the scalar
`B`. This v0.1.1 interface (`scripts/irp_interface_v0_1_1.py`) is what every candidate downstream
of TSRS is actually built and scored against; the superseded `v0.1` interface is not included in
this edition.
