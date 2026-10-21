# Latent closed-loop result (v0.6 execution error → v0.6.1 fixed run)

This document covers the multi-round closed-loop measurement that follows the re-entry
holdout confirmation (see `reentry-holdout-confirmation.md`): the frozen scientific
protocol v0.6, its first runner's execution failure (archived openly, not hidden, and
explicitly *not* a refutation), and the bug-fixed v0.6.1 re-run that produced the
reported result.

## Protocol v0.6 — closing the loop explicitly

Protocol: `Protocole_TRANSFORMER_LATENT_CLOSED_LOOP_v0_6_FROZEN.md` (frozen after v0.5,
before any v0.6 execution).

**Objective.** Build the program's first multi-round loop without introducing any
hidden cross-layer latent transport. The loop is closed *experimentally*:

```
C_t → Y_t^ret → C_{t+1}
```

`Y_t^ret` is a symbol among the eight historical candidates. Each candidate symbol
`y_i` is transported explicitly to its historical donor intervention `q_i`, then to its
top-2 class: `y_i ↦ q_i ↦ χ(q_i) = C_{t+1}`. This transport is declared by
construction, not inferred after measurement — the protocol does not claim to identify
an autonomous internal loop of the transformer. It measures an **externally closed
latent loop** built from the already-validated latent interventions.

**Fixed objects.** Model/revision, source layer 2 / position 1, the eight historical
donors and candidate tokens, and the validated top-2 quotient
`Π_2 = {0}|{1}|{2}|{3,7}|{4}|{5}|{6}`. Initial mass on `C_0` is induced by uniform `Q`:
`(1/8, 1/8, 1/8, 2/8, 1/8, 1/8, 1/8)`, so `H(C_0) = 2.75 bit`.

**New loop contexts.** After all tokens used through v0.5, four new eligible tokens are
selected: two loop keys `L_0, L_1` and two loop fillers `G_0, G_1`. The four contexts
are `(L_a, A, G_b, L_a)` for `a, b ∈ {0,1}` — never used for the top-2 quotient or the
v0.4/v0.5 holdouts.

**One-round kernel.** For each context `u` and source class `c`: run every donor
`q_i ∈ c` with the historical source patch; read the return distribution over the 8
candidates; average uniformly over the donors in the class; push the distribution
forward by `y_i ↦ χ(q_i)`. This yields a Markov kernel `T_u(c'|c)` over the seven top-2
classes.

**Multi-round loop.** For a context `u` held fixed for an entire trajectory,
`T_u^(n) = T_u^n`. Measured for `n ∈ {1,2,4,8,16}`. With the initial distribution
`p_0(C)` above, `B_n = I(C_0; C_n | U)` and `R_n = B_n / H(C_0)`. Primary profile:
`(R_1, R_2, R_4, R_8, R_16)`.

**Required monotonicity.** Each context's chain is Markovian
(`C_0 → C_n → C_{n+1}`), so by data processing `I(C_0; C_{n+1}|U=u) ≤ I(C_0; C_n|U=u)`
for every `u`, hence `B_{n+1} ≤ B_n` after averaging over `U`. The runner invalidates
the run if `B_2 > B_1 + 1e-12` (or the analogous check at 4, 8, 16).

**Controls declared:** L0 stochasticity (`T_u` rows sum to 1 at `1e-12`); L1 explicit
transport table `y_i ↦ q_i ↦ C` published in the result; L2 temporal control
`I(C; Y_pre|U) < 1e-8 bit`; L3 duplication (duplicating donor labels at constant class
mass must not change `T_u` or the `R_n` profile); L4 one-round consistency —
`I(C_0; Y_ret|U)` before projection to `C_1` must dominate `I(C_0; C_1|U)` after
projection, by data processing.

**No numeric target pre-declared.** Immediate extinction, slow decay, a non-zero
plateau, or strong context-dependence were all pre-declared as scientifically
acceptable outcomes.

**Verdicts.** `LOOP-V1` (all controls pass, profile computed); `LOOP-V2` (valid kernel
but `H(C_0)=0`, impossible here since the frozen partition is non-trivial);
`LOOP-V3` (structural/causal control violation); `LOOP-V4` (model/tokenizer/interface
incompatibility).

**Claim status.** Even `LOOP-V1` does not measure a spontaneous internal loop. It
measures the multi-round causal retention of a loop closed explicitly between the
transformer's return channel and its latent intervention battery — a distinction the
manuscript must keep explicit.

## v0.6 execution error (archived, not a refutation)

File: `TRANSFORMER_LATENT_CLOSED_LOOP_v0_6_EXECUTION_ERROR_ADJUDICATION.md`.

**Status: `EXECUTION-ERROR / NO-VERDICT`.** The v0.6 runner produced **no scientific
result**. Observed exception: `KeyError: (2, 0)` inside `cmi_source_to_y(prior, uprior,
class_y)`.

**Exact cause.** The protocol defines four contexts (2 keys × 2 fillers = 4), but the
runner enumerated them as:

```python
for u,(lk,_lt) in enumerate(loop_keys):
    for lf,_ft in loop_fillers:
```

so `u` only ever took the values 0 and 1, each reused for both fillers — an
`enumerate` bound to the outer loop and reused across the inner loop. The dictionaries
`class_y`, `pre_y`, and `transitions` therefore only ever held contexts 0 and 1.
Further down, the runner correctly built `uprior = {u: 1/4 for u in range(4)}` and then
tried to access contexts 2 and 3, which had never been populated — a pure indexing
defect, not a conceptual or protocol error.

**Second gap found before re-running.** The v0.6 runner also did not explicitly execute
the required L3 duplication control.

**Repair authorized for v0.6.1.** The repair changes no scientific decision. It only:
(1) explicitly enumerates the four `(loop_key, loop_filler)` pairs, and (2) explicitly
computes a duplicated kernel and compares its transitions and its `R_n` profile against
the undoubled one. The protocol, tokens, model, quotient, transport, contexts, and
rounds `n = 1,2,4,8,16` are all unchanged from v0.6.

**Claim status.** The faulty v0.6 run must not appear in any table as `LOOP-V3`. It
reached no protocolal verdict at all and is archived as:

```
IMPLEMENTATION FAILURE BEFORE MEASUREMENT
```

This adjudication — including the exact bug — is kept in this clean edition
deliberately, per the project's reproduction discipline: genuine adjudicated defects
are documented, not hidden.

## v0.6.1 — fixed run, reported result

Script: `transformer_latent_closed_loop_pythia70m_v0_6_1.py` (only file used for
this campaign's reported loop result; the pre-fix `..._v0_6.py` script is not part of
this clean edition).

**Verdict: `LOOP-V1`.**

```
H(C_0)                          = 2.75 bit
I(C_0; Y_ret | U)  [pre-project] = 0.0144286677519159 bit
I(C_0; C_1  | U)  [post-project] = 0.0108997700427762 bit
R_1                               = 0.00396355274282772
```

Multi-round profile:

```
(R_1, R_2, R_4, R_8, R_16) =
(0.00396355274282772,
 1.80125874286848e-05,
 5.0926939219345e-10,
 6.56648047639975e-18,
 -5.91445820305333e-18)
```

Values at `n = 8, 16` are numerically zero to machine rounding. In raw information:
`B_1 = 0.0108997700427762 bit`, `B_2 = 4.95346154288833e-05 bit`,
`B_4 = 1.40049082853199e-09 bit`, `B_8 ≈ 0`, `B_16 ≈ 0`. The pre-declared
data-processing monotonicity passes.

**Controls.** Temporal control `I(C_0; Y_pre | U) = 2.159e-17 bit` (compatible with
numerical zero). Duplication control passes exactly: `max|ΔT| = 0`,
`max|ΔR_n| = 0`. Projection consistency `I(C_0; C_1|U) ≤ I(C_0; Y_ret|U)` holds.

**Post-hoc diagnostics (descriptive, not part of the protocol verdict).** The
`Y_ret → C_1` projection retains `I(C_0;C_1|U) / I(C_0;Y_ret|U) = 0.755425` (≈75.54%)
of the pre-projection information. From round 1 to round 2, the retained fraction is
`B_2/B_1 = 0.004545` (≈0.454%). The second-largest eigenvalue modulus of the four
per-context kernels: `{0: 0.0705602057838818, 1: 0.0830814658235815,
2: 0.06496711220496904, 3: 0.0587740380122855}` — all below `0.083081`, consistent
with very fast mixing.

**Interpretation.** The explicit loop shows no durable multi-round causal memory. The
one-round signal is positive but weak (`R_1 ≈ 3.96×10⁻³`); by two rounds it is down to
`R_2 ≈ 1.80×10⁻⁵`; by four rounds, `R_4 ≈ 5.09×10⁻¹⁰`; retention becomes numerically
undetectable beyond that.

**Authorized claim statement:**

> For Pythia-70M-deduped, on the frozen hook, intervention battery, top-2 source
> quotient, and explicitly declared loop transport, a positive but weak one-round
> causal re-entry is measured. In the externally closed loop tested, this information
> dissipates almost entirely within two to four iterations.

**Explicitly disallowed statements:** "Pythia has an internal recurrent loop of
consciousness"; "Pythia's intrinsic R equals [any value above]".

## Overall closure (`POINT4_TRANSFORMER_AND_LATENT_LOOP_CLOSURE_v0_1.md`)

Status: **POINT 4 — CLOSE-CONDITIONAL**, combining both sub-parts:

- **4A (re-entry):** top-2 quotient discovered in v0.4, frozen, reproduced exactly on
  the v0.5 candidate-fixed holdout — `K_eff = 2.75 bit`,
  `B_reentry = 0.01362349150610347 bit`, `R = 0.004953996911310352`,
  `η = 0.9022417474341858`.
- **4B (closed loop):** v0.6 failed with an indexing bug before any verdict
  (`EXECUTION-ERROR / NO-VERDICT`, not a refutation); v0.6.1, same protocol, bug fixed,
  reached `LOOP-V1` with `R_1 ≈ 3.96×10⁻³` decaying to numerical zero by round 4.

This level of evidence justifies `CLOSE-CONDITIONAL`, but not generalization to other
layers, models, task families, or unassisted internal loops.
