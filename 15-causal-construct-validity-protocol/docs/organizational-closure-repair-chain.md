# Organizational Closure Repair Chain: Typed Binding → Joint Synergy → Static Sufficiency

**Status: executed.** Frozen protocols, candidate implementations, and
result files are included under `scripts/` and `results/`. This document
narrates the chain; the JSON files are the primary record.

**Reproducibility note.** The v0.3 witness (`fci2_access_role_binding_harness_v0_3.py`)
and the v0.4 repair (`typed_propagation_repair_harness_v0_4.py`) were
re-executed in this archive edition from the supplied files alone and
reproduced their frozen verdicts exactly
(`FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING`,
`FCI2-TYPED-PROPAGATION-REPAIR-PASS`). The v0.5 witness and v0.6 repair
harnesses each verify a source-hash manifest referencing a
`*_SOURCE_REFS_*.json` file that was **not present** in the uploaded source
bundle; this is a gap in the material supplied to this archive edition, not
a withheld file. Their frozen `results/*.json` are included as the
record of the original run and are reported here as such — see
`results/STATUS.json` for the itemized reproducibility status of every
script in this folder.

**Notation note.** The source campaign used `C_F = [(Γ, Δ_prop, R(k))]`,
predating this manuscript's fourth component Λ. Below, `Δ_prop` corresponds
to this manuscript's `Δ`. The chain never used Λ — it is reproduced here
under its original three-component notation, and the reader should
understand the repaired profile `C_F^++` as a strict subset of the
information later folded into the manuscript's `C_O`, not as a claim about
Λ itself.

## 1. Starting point: the localization theorem

Conditional on four-family reachability closure (folder `14`) and invariance
of functional access to admissible changes in 𝒮, 𝒢, 𝒪 at fixed 𝒞,
functional access factors through active causal organization:

```
A = H_𝒞(𝒞)
```

This is an exact conditional theorem. It says nothing yet about whether the
*compact* profile `C_F = [(Γ, Δ_prop, R(k))]` — as opposed to 𝒞 in full —
is enough. That is what the rest of this chain tests.

## 2. v0.3 adversarial result: typed-binding insufficiency

A prospectively frozen exact witness (`scripts/fci2_access_role_binding_harness_v0_3.py`,
result in `results/FCI2_ACCESS_ROLE_BINDING_RESULT_v0.3.json`) constructs two
systems A and B with:

```
C_F^compressed(A) = C_F^compressed(B)      where C_F^compressed = [(Γ, Δ_prop, R(k))]
```

while an independently frozen access consequence differs:

```
a_flex(A) = 1,   a_flex(B) = 0
```

The witness uses only two preregistered access roles, avoiding an earlier
decoy-role protocol-scope error from a prior draft of the same test. The
information the compact profile loses is **typed content-to-role binding** —
*which* role a piece of content ends up controlling, not merely how many
roles are reachable.

**Verdict: `FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING`.**

This is a concrete instance of the Compression Limit Theorem (folder `14`):
`p = C_F^compressed` is non-injective, and `a_flex` distinguishes the two
preimages it collapses.

## 3. v0.4 repair: typed marginal response channel

A repair was frozen only *after* the v0.3 witness was recorded (no
retroactive tuning). It keeps the existing scalar `Δ_prop` and adds a typed
intervention–response signature per certified role:

```
B_q = [(r_j, K_{j,q})]_{j=1}^{m}
D_q = (Δ_prop, B_q)
C_F^+ = [(Γ, D_q, R(k))]
```

(`scripts/typed_propagation_repair_candidate_v0_4.py`,
`scripts/typed_propagation_repair_harness_v0_4.py`, result in
`results/FCI2_TYPED_PROPAGATION_REPAIR_RESULT_v0.4.json`.)

**Verdict: `FCI2-TYPED-PROPAGATION-REPAIR-PASS`** on the frozen witness and
frozen invariance controls. This is not universal sufficiency — it repairs
one exactly identified information-loss mode without introducing a new
primitive family.

## 4. v0.5 adversarial result: joint-synergy insufficiency

`C_F^+` was attacked again. Two systems were built with identical `Γ=1`,
identical `R(1)=R(2)=R(4)=1`, identical `Δ_cap=1/2`, identical
`Π_role,q=1`, identical `Δ_prop=1/2`, and identical typed marginal channels
`B_q` — differing only in how two roles' outputs combine:

- `CONTENT-SYNERGY`: the target content exists only in the cross-role
  relation `X = Y_L ⊕ Y_R`.
- `NUISANCE-SYNERGY`: the same amount of joint information carries an
  independent nuisance bit `S = Y_L ⊕ Y_R` instead.

The independent `AF-FLEX` consequence is `1` in one case and `0` in the
other, while every *marginal* typed channel is identical.

(`scripts/cfplus_joint_role_synergy_harness_v0_5.py`, result in
`results/CFPLUS_JOINT_ROLE_SYNERGY_RESULT_v0.5.json`.)

**Verdict: `CFPLUS-PROFILE-INSUFFICIENT-JOINT-SYNERGY`.** Typed role
*marginals* are not sufficient; joint cross-role structure can be load-bearing
for access and is invisible to a profile built only from per-role marginal
channels.

## 5. v0.6 repair: joint typed response channel

The final repair in this chain retains the full **joint** typed
intervention–response channel instead of per-role marginals:

```
K^joint_{q,R} : p ↦ P_q(Y_R | do(P = p))
D_q^++ = (Δ_prop, [K^joint_{q,R}]_adm)
C_F^++ = [(Γ, D_q^++, R(k))]
```

(`scripts/joint_typed_propagation_candidate_v0_6.py`,
`scripts/joint_typed_propagation_harness_v0_6.py`, result in
`results/CFPLUSPLUS_JOINT_TYPED_PROPAGATION_RESULT_v0.6.json`.) This repairs
both the v0.3 typed-binding witness and the v0.5 joint-synergy witness
simultaneously.

### Control adjudication (v0.6 → v0.6.1)

The initial v0.6 harness returned a `REVIEW` status because one cross-type
swap control happened to use a channel that was exactly symmetric under that
swap — an inconclusive control, not a candidate failure. The candidate was
**not modified** to work around this. Instead, a new, candidate-fixed
asymmetric holdout was frozen independently and run.

**Verdict: `CFPLUSPLUS-CROSS-TYPE-HOLDOUT-PASS`** (result in
`results/CFPLUSPLUS_CROSS_TYPE_ASYMMETRY_HOLDOUT_RESULT_v0.6.1.json`). The
candidate is retained on this basis.

## 6. Static-access sufficiency theorem

At fixed source distribution, target semantics, loss function, and
admissible decision-rule class, equal joint typed response channels
`K^joint_{q,R}` imply equal expected loss for every fixed static decision
rule, hence equal optimal achievable loss. Therefore `C_F^++`'s joint typed
channel is sufficient for **every static access consequence that reads only
the certified joint response**.

This is stated and proved as an **exact conditional theorem**, not a finite
non-refutation — it follows from the definitions once the channel is held
fixed, not from exhausting a witness set.

## 7. Current boundary — what remains open

This entire chain concerns *static* access consequences. Explicitly **not**
addressed:

- `AF-DELAY`, `AF-UPDATE` (temporal consequences)
- temporal order
- path-specific gating
- transition-specific content binding

The next object to test is not another static scalar but a typed
trajectory/intervention-process extension of the causal response kernel.
This is precisely the gap that the FCI-3 causal construct validity protocol
(`docs/fci3-protocol-specification.md`) is designed to probe empirically,
since it targets `R(k)` via temporal endpoints (`a_delay`, `a_update`)
directly.

## 8. Manuscript consequence

The manuscript should not claim general sufficiency of the *originally
declared* compact profile `[(Γ, Δ_prop, R(k))]` for functional access — this
is the L3 verdict in `../14-four-family-closure-and-compression-limit/docs/claim-ladder.md`.
It may claim the narrower, exact result: the enriched joint typed profile
`C_F^++` is provably sufficient for the class of *static* access
consequences tested, and the repair chain that reached it is itself evidence
that naively compact profiles need not be trusted without adversarial
testing.
