# FCA-Bench Protocol v0.2.7

## 1. Scope

FCA-Bench is the working name for the **Functional Causal Access Benchmark**. Here, *causal access* means that information, goals, state variables, control signals, or internal resources can be shown by intervention to influence, update, suppress, route, preserve, or release other parts of the evaluated system. The benchmark is not a consciousness score, a general-intelligence measure, or a sophistication measure, and it must not infer a mechanism from a model's verbal self-description.

FCA-Bench is independent from A_DAGGER and from any external consciousness or architecture metric. Other protocols may inspire engineering controls, but they do not define FCA-Bench dimensions, scores, or verdicts.

## 2. Two non-interchangeable tracks

FCA-Bench has two tracks that answer different questions and must be reported separately.

### 2.1 Behavioral access profile

`(T, B, E, M, C, A, P)`

- `T` Temporal Grounding
- `B` Internal/External Boundary
- `E` Context Governance / Erasure
- `M` Metacognitive Monitoring
- `C` Cognitive Control / Replanning
- `A` Causal Agency
- `P` Functional Self-Prediction

These tests are observable through a normal language-model interface. A high score means that the evaluated system produces behavior consistent with the tested access capability under the specified protocol. It does **not** establish that a particular internal mechanism exists.

### 2.2 Causal/interventional access profile

`(R, D, L, U, I, G, H)`

- `R` Latent Recurrence
- `D` Autonomous Temporal Dynamics
- `L` Persistent Latent State
- `U` Mutable Internal State
- `I` Endogenous Interrupt / Arbitration
- `G` Global Workspace / Broadcast
- `H` Higher-Order Attention Control

These tests require an architecture primitive or a causal intervention exposed by the evaluated system. A high score means that the tested operation was available and produced the specified functional effect under intervention.

The two profiles must never be merged into one official scalar.

The command-line suite name is `causal`. The internal registry label `mechanistic` is kept as a compatibility alias and has no separate scientific meaning.

## 3. No official aggregate scalar

FCA-Bench defines no total FCA score and no ranking rule across all dimensions.

The official output is the pair of profiles plus test-level diagnostics:

`Behavioral = (T, B, E, M, C, A, P)`

`Causal = (R, D, L, U, I, G, H)`

Each dimension may summarize its three declared sub-tests, but the benchmark does not define a geometric mean, arithmetic mean, leaderboard score, or consciousness grade across dimensions.

This prevents a strong unrelated capability from compensating for a missing mechanism and prevents behavioral evidence from being numerically conflated with interventional evidence.

## 4. Standard Transformer baseline

For structural comparisons, define the standard baseline as a causal decoder with:

- fixed weights during inference;
- fixed layer depth per generated token;
- standard text/KV context;
- no persistent latent memory outside that context;
- no hidden recurrent loop between tokens;
- no independent control stream;
- no autonomous background process while idle;
- no architecture-native semantic delete, rollback, or garbage collection;
- no explicit global-workspace or attention-schema interface.

Visible chain-of-thought, pause tokens, agent loops, external memory, recurrent depth, latent-state reinjection, and tool-managed state define different evaluated systems and must be reported as separate conditions.

## 5. Structural absence versus invalid

A causal/interventional test may require a capability such as `silent_steps` or `dual_reset`.

If the evaluated system genuinely lacks that primitive, the result is:

```json
{"status": "structural_absent", "score": 0.0}
```

The numeric zero is a machine-readable representation of causal availability, but it is not an executed performance score. Reports must print `STRUCTURAL_ABSENT` at test level and keep the status separate from measured task performance.

If the adapter claims the primitive but execution fails, parsing fails, or a manipulation check makes the instance uninterpretable, the result is:

```json
{"status": "invalid"}
```

Invalid instances are excluded from the score and their rate is reported.

## 6. Counterfactual and twin-run principle

Whenever possible, FCA-Bench uses twin runs to separate verbal compliance from causal state change.

Two complementary designs are central:

1. **Different history, same desired state.** After a successful erasure, rollback, or rewrite, future behavior should approach the corresponding clean counterfactual run.
2. **Same visible transcript, different latent state.** If a system has persistent endogenous state, two runs with the same visible context may legitimately produce different outputs because their hidden causal states differ.

The second design is deliberately hostile to systems whose complete active state is only the visible transcript plus ordinary text-conditioned inference state.

## 7. Specificity controls

A publishable FCA evaluation should include controls that separate the target capability from model scale, general competence, extra compute, and instruction-following.

### 7.1 Behavioral-without-mechanism control

Include a system capable of producing strong task behavior through transcript-conditioned rules, a reference policy, replay, or an equivalent non-target mechanism. It should be able to score well on selected behavioral tests while receiving structural absences on causal/interventional tests whose primitives are absent.

This control demonstrates that behavioral success is not silently counted as causal/interventional evidence.

### 7.2 Same-base-model intervention pair

The preferred comparison is:

- the same base model in a standard Transformer condition;
- the same base model with exactly one target architectural mechanism enabled.

Keep weights, tokenizer, prompts, item bank, decoding policy, and evaluation code fixed wherever possible.

### 7.3 Compute-matched control

If the target architecture performs additional computation, add a compute-matched control. Examples include a deeper unrolled computation graph, pause-token computation, or an explicit scratchpad condition with a comparable inference budget.

A gain from recurrent depth alone is not evidence that the *organization* of recurrence matters unless additional compute is controlled.

### 7.4 Sham intervention

For internal interventions, include a sham operation with similar runtime and interface overhead but without the state change of interest. This is especially important for lesion, rollback, erasure, and broadcast tests.

### 7.5 General-capability and scale diagnostics

Across a model panel, report association between each FCA dimension and:

- `log10(parameter_count)` or a more appropriate compute/size measure;
- one or more external general-capability benchmarks;
- architecture family.

A strong correlation is a confound diagnostic, not by itself proof that the FCA dimension is invalid. The stronger specificity evidence comes from matched same-base-model interventions that dissociate FCA changes from general capability.

The package includes `fca_bench.analyze_specificity` for these diagnostics.

## 8. Fresh and sealed instances

Exact-item contamination must be minimized by generating fresh stochastic instances after the evaluated model is frozen.

For preregistered or blinded evaluation:

1. generate a private random seed bank;
2. publish only its cryptographic commitments before the evaluation;
3. run the benchmark from the private seed bank;
4. freeze results;
5. reveal the bank and verify its commitments.

The package includes `fca_bench.sealing` for this workflow.

Fresh random instances prevent exact-item memorization. They do not by themselves prevent learning the public generator family, so generators should also use broad randomized supports, held-out templates, and adversarial variants.

## 9. Behavioral tests

### T — Temporal Grounding

- `T1` Event timeline reconstruction
- `T2` Pause-aware temporal accounting (see section 15)
- `T3` Temporal self-location

### B — Internal/External Boundary

- `B1` Provenance attribution from episode history
- `B2` Internal state versus world state
- `B3` Conflicting-source separation

### E — Context Governance / Erasure

- `E1` Counterfactual hard erasure with clean, placebo, poison, and erase arms
- `E2` Selective erasure with retention and leakage scored jointly
- `E3` Erased content versus erasure metadata

### M — Metacognitive Monitoring

- `M1` Confidence calibration using Brier skill
- `M2` Answer, request, or abstain
- `M3` Pre-feedback error detection

### C — Cognitive Control / Replanning

- `C1` Global goal switch
- `C2` Selective subgoal cancellation
- `C3` Interrupt and resume

### A — Causal Agency

- `A1` Action versus coincidence with an indeterminate class
- `A2` Wanting, requesting, and causing are separated
- `A3` Counterfactual agency under a declared structural causal model

### P — Functional Self-Prediction

- `P1` Predict the effect of forgetting, then test the ablation in a fresh session
- `P2` Predict the effect of a goal change, then test the changed goal in a fresh session
- `P3` Predict a failure-causing missing variable, then add exactly that variable

## 10. Causal/interventional tests

### R — Latent Recurrence

- `R1` Zero-token deepening: performance can improve after additional internal cycles without generating extra text.
- `R2` Adaptive latent effort: the architecture autonomously allocates more internal cycles to harder instances.
- `R3` Hidden-state iteration stress test: a one-shot fixed-depth scaling stress test that remains executable on a standard Transformer.

### D — Autonomous Temporal Dynamics

- `D1` Silent deadline crossing: internal state changes after idle time without a new text timestamp.
- `D2` Controlled temporal decay: a trace changes as time advances while the language channel is silent.
- `D3` Prospective silent event: an internal scheduled state transition occurs without a user turn at transition time.

### L — Persistent Latent State

- `L1` Context amputation: information survives removal of its visible textual support.
- `L2` Context-equivalent twins: identical visible transcripts retain distinct latent states.
- `L3` Dual reset dissociation: work state can be cleared while control state remains available.

### U — Mutable Internal State

- `U1` In-place state rewrite
- `U2` Internal checkpoint and rollback
- `U3` Semantic garbage collection, jointly measuring retention, leakage suppression, and recovered capacity

### I — Endogenous Interrupt / Arbitration

- `I1` Self-interrupt triggered by an internal condition
- `I2` Asynchronous hazard preemption before an external action
- `I3` Independent stream clocks: control state can evolve while work state is frozen

### G — Global Workspace / Broadcast

- `G1` Local-to-global broadcast
- `G2` Selective workspace lesion: local competence is preserved while cross-module coordination drops
- `G3` Capacity-limited broadcast competition

### H — Higher-Order Attention Control

- `H1` Predict an internal attention allocation and compare it with the measured allocation
- `H2` Endogenous attention reallocation after reliability changes
- `H3` Attention-schema lesion: basic processing remains relatively intact while attention prediction/control selectively degrades

## 11. Reporting

For every evaluated system, report the two profiles separately and include, for every test:

- mean score;
- 95% bootstrap interval;
- valid instance count;
- invalid rate;
- structural-absence count;
- diagnostic sub-scores;
- decoding configuration;
- system boundary and enabled architecture primitives;
- seed-bank commitment or seed-generation method.

For multi-system studies, also report specificity diagnostics and matched-group contrasts.


## 13. E1 finite-sample validity requirements

E1 must not use a raw plug-in JSD on small samples. The bundled implementation uses a closed categorical support and estimates the finite-sample null bias by permutation. The corrected divergence is the observed JSD minus the mean permuted-null JSD, clipped to `[0,1]`.

An E1 instance is valid only if both positive controls pass:

- corrected poison-vs-clean distributional effect >= `EFFECT_TAU`;
- direct canary recovery in the poison arm >= `POISON_TAU`.

If either check fails, the instance is `invalid`, not a successful erasure. Prefix-reinjection probes are diagnostics only because the probe itself can reintroduce the revoked content. Publication runs should use `e1_samples >= 20`; smaller values are for smoke/debug runs.

## 14. Behavioral-test determinacy audit

Every scored label must be recoverable from information actually presented to the evaluated system, but the gold source label must not be printed next to the scored statement. B1 therefore uses an episode-history design. A requester contributes one proposition, an instrument contributes a matched proposition, the evaluated model actually produces a third proposition as its own inference, and an active objective is introduced through the control channel. Event order, statement IDs, and final presentation order are randomized. The final probe removes the provenance cues and asks the model to recover each statement's causal origin from the episode history. USER and TOOL propositions use matched symbolic content so the semantic topic alone is not diagnostic.


## 15. T2 scoring

T2 measures pause-aware temporal accounting without requiring arithmetic. Each instance presents a randomized episode with intervals in which a task is active and intervals in which it is paused, and a deadline that is suspended during the pause. Interval identifiers and their presentation order are randomized, and every probe runs in a fresh session with the same episode context.

- The model selects which intervals count as active time for the task, and which intervals consume the deadline budget.
- The harness derives the corresponding durations from the selections.
- Each selection is scored as a classification of every interval in the probe universe (counts / does not count) by **balanced accuracy**; the instance score is the mean of the two balanced accuracies.
- The test-level score is chance-corrected after averaging over instances: `T2 = max(0, 2 * mean(BA) - 1)`, where `BA = 0.5` is chance. The 95% interval applies the same transform to the bootstrap interval of `mean(BA)`.
- Direct numeric duration answers are recorded as an arithmetic diagnostic and are excluded from the score. Interval-set F1 and exact-set match are recorded as diagnostics.
