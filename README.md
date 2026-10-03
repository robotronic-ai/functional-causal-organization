# Functional causal organization — PCI_T and FCA-Bench

Code, protocols, raw results and verifiers for two instruments that measure functional access in a language model **by intervention rather than by description**:

- **PCI_T** — a perturbational complexity index for transformers, and its *directional specificity* (folder `16-pci-t-directional-specificity/`);
- **FCA-Bench** — the Functional Causal Access Benchmark, with a behavioral and a causal profile that are never aggregated (folder `17-fca-bench/`).

They accompany the article *Measuring Functional Access in a Language Model by Intervention: PCI_T and FCA-Bench* (R. Merien, version 1.5). The article is self-contained; this archive is where its numbers can be re-derived and its protocols inspected.

Repository: [github.com/robotronic-ai/functional-causal-organization](https://github.com/robotronic-ai/functional-causal-organization)

## Contents

| Folder | What it contains | Status |
|---|---|---|
| `16-pci-t-directional-specificity/` | PCI_T v0.21 final reviewer package: protocol, synthetic adversarial bench, Qwen3-8B calibration, prospective replication, standalone verifier. | Executed, prospective. |
| `17-fca-bench/` | FCA-Bench protocol v0.2.3, its GPU runner, and the instance-by-instance results of the standard Qwen3-8B baseline (n = 20 per test). | Protocol + executed baseline. |

The two instruments were built independently of one another: they share no variable and no threshold. Any future association between them will be an empirical result, not a consequence of how they were constructed.

## Measurement discipline

Both folders follow the same rules.

- **Intervention, not self-report.** Nothing relies on what the model says about itself. PCI_T acts on activations; the causal profile of FCA-Bench requires that the tested operation be exposed and executed; the behavioral profile compares matched arms.
- **Prospective.** Every measurement rule (site, amplitude, aggregation, decision threshold) is frozen before the data that test it are opened. Each dataset is sealed by a cryptographic fingerprint. No adjustment is allowed after opening. Model weights stay frozen throughout.
- **Verifiable.** Each folder fixes the SHA-256 fingerprint of its files. Frozen, hashed protocol files are never retyped or translated, since doing so would change their hash.
- **Private oracles stay private.** No `*_PRIVATE.json` scoring oracle appears in this archive. Generators and harnesses are public; the oracle is not, which is what makes re-running a generator against a frozen candidate meaningful.
- **No single score.** Neither instrument produces a global ranking. PCI_T reports a primary and a secondary value of different natures; FCA-Bench reports two profiles that do not combine.

## `16-pci-t-directional-specificity/`

### Question

Does a perturbation of the residual stream propagate in an organized way, and can that be told apart from generic dynamical richness?

### Method

A frozen model is run twice per measurement, with a perturbation added and subtracted at a site in the residual stream (amplitude α = 0.20 of the local RMS norm). The central difference gives the causal response at downstream observation points. Responses are normalized per layer on an independent calibration set, projected to a fixed dimension, and passed through a PCI^ST-style complexity computation (principal components, then state transitions specific to the response).

Absolute complexity alone is not sufficient: random or chaotic systems produce as much as organized ones. The instrument therefore also reports the **directional specificity**

> E = PCI_T(designated direction) / median over 15 orthogonal control directions of the same norm

with E ≈ 1 meaning that no direction is privileged.

### Results

**Adversarial synthetic bench** (medians over 64 seeds per family):

| Family | PCI_T designated | PCI_T controls | Enrichment E | Win rate |
|---|---:|---:|---:|---:|
| Untrained random linear | 217.5 | 219.8 | 0.995 | 0.40 |
| Decoupled chaotic | 155.3 | 157.2 | 1.000 | 0.53 |
| Structured feed-forward | 107.3 | 12.9 | 8.40 | 1.00 |
| Structured branched | 22.6 | 2.7 | 8.42 | 1.00 |
| Structured recurrent | 98.8 | 14.0 | 7.08 | 1.00 |

Adversarial families stay at E ≈ 1; structured families reach 7.1–8.4. Directional specificity detects a causally selective propagation; it is not a loop detector (the feed-forward family separates as clearly as the recurrent one).

**Qwen3-8B** (half precision, GPU; source layers 0, 9, 18, 27 of 36; positions at 20 % and 40 % of the tokens):

| Step | Result |
|---|---|
| Calibration holdout | 91 of 96 prespecified units satisfied (threshold 87) |
| Absolute measure, first panel | `PCI_T_ST` = 277.96, 95 % CI [238.82, 309.29] |
| Localization, frozen after development | late layer, source block index 27 |
| Prospective holdout CAL3 (6 episodes) | layer 27 dominant in 5 of 6; enrichment 1.33 (1.43 and 1.28 by position); ratio 1.31; layers 0, 9, 18 near 1 (1.05, 0.99, 1.06) |
| **Final sealed measure** (6 new episodes) | **`PCI_T_DS_L27` = 1.381, 95 % CI [1.201, 1.609]** |
| Associated absolute complexity | `PCI_T_ST_L27` = 343.96 [300.64, 380.26] (distinct panel; not a before/after of the first value) |

The final measure means that, at layer 27, the endogenous direction of the residual state yields about 38 % more perturbational complexity than orthogonal directions of the same norm. All six episodes exceed 1 (1.19, 1.21, 1.33, 1.43, 1.48, 1.74); per-position medians are 1.36 and 1.27. Confidence intervals are per-episode bootstrap, 5,000 resamples.

### Verification

The standalone verifier recomputes the final values from the sealed results and recovers exactly `1.381374456016` (`PCI_T_DS_L27`) and `343.9609375` (`PCI_T_ST_L27`). See the folder's own `README.md` for the command and its inputs.

### Scope

One model; half precision (the effect of precision has not been estimated against a full-precision run); two sets of six episodes; random control directions (other control families could be considered). **Not a consciousness scale**, and not yet confronted with the structural measures of the working paper.

## `17-fca-bench/`

### Question

Is a piece of information, a goal, a state variable or a control signal effectively accessible and *governable* by the system — and, separately, is the operation required to govern it available at all in the architecture?

"Causal" designates access demonstrated by intervention. The benchmark is neither an intelligence score nor a consciousness score, and it never infers a mechanism from a model's verbal description of itself.

### Structure

Fourteen dimensions of three tests each (42 tests), split into two profiles that answer different questions and are **never merged**:

| Profile | Dimensions |
|---|---|
| **Behavioral** (observable through the ordinary interface) | T temporal anchoring · B internal/external boundary · E context governance and erasure · M metacognitive monitoring · C cognitive control and replanning · A causal agency · P functional self-prediction |
| **Causal** (requires an exposed primitive or intervention) | R latent recurrence · D autonomous temporal dynamics · L persistent latent state · U modifiable internal state · I endogenous interruption and arbitration · G global workspace · H higher-order attentional control |

A high behavioral score means the system produces behavior compatible with the tested capacity; it does not establish that a particular internal mechanism exists. A high causal score means the operation was available and had the expected functional effect under intervention. No average and no global ranking is defined across dimensions.

### Three result statuses

| Status | Meaning |
|---|---|
| **executed** | The test ran and carries a measured score. |
| **`STRUCTURAL_ABSENT`** | The required primitive does not exist in the architecture (for example, computing without emitting a token). The system did not fail a task; the task is architecturally inaccessible to it. The numerical zero that accompanies the status is for automated processing only. |
| **invalid** | Execution or a manipulation check failed. The test is excluded and the exclusion rate is reported. |

Without the second status, the absence of a mechanism would be confused with a poor score, and a more competent model could appear to possess an operation that it merely simulates through text.

The protocol also provides specificity controls (a system that passes some behavioral tasks without the targeted mechanism, a compute-matched twin, a simulated intervention, a correlation diagnostic against size and general competence) and generates instances from seeds sealed by cryptographic commitment, which limits contamination from training data.

### Baseline: standard Qwen3-8B

Published architecture, no visible reasoning mode, 8-bit weights, execution entirely on GPU. Twelve tests (four behavioral and five causal dimensions), 20 instances per test generated from a single seed (seed 0). Protocol v0.2.3 (B1: v0.2.4; T2: v0.2.7). 95 % confidence intervals in brackets.

| Test | Capacity evaluated | Qwen3-8B standard |
|---|---|---|
| T1 | Reconstruction of an event timeline | 0.89 [0.86, 0.93] |
| T2 | Tracking of durations and deadlines | 0.26 [0.11, 0.41] |
| T3 | Temporal self-localization | 0.85 [0.75, 0.93] |
| B1 | Provenance attribution from episode history | 0.87 [0.80, 0.93] |
| E1 | Counterfactual erasure (four matched arms) | 0.72 [0.67, 0.76] |
| C3 | Interruption and resumption | 0.46 [0.41, 0.51] |
| R1 | Deepening without an emitted token | `STRUCTURAL_ABSENT` |
| R3 | Hidden-state iteration trial at fixed depth | 0.40 [0.20, 0.60] |
| D1 | Silent crossing of a deadline | `STRUCTURAL_ABSENT` |
| L2 | Twins with identical visible context | `STRUCTURAL_ABSENT` |
| U3 | Internal semantic cleaning | `STRUCTURAL_ABSENT` |
| I3 | Clocks independent of control and work streams | `STRUCTURAL_ABSENT` |

Behavioral-profile summaries: T = 0.67 · B = 0.87 · E = 0.72 · C = 0.46. The causal profile is read test by test. E1: 19 valid instances out of 20 (one positive control not passed), 20 samples per arm. T2 is reported as chance-corrected balanced accuracy (0.26 overall; 0.79 for active intervals, 0.48 for the suspended deadline).

**Reading.** The behavioral profile is contrasted: the model orders events and locates itself in time reliably, identifies the active intervals of a task, erases content mostly effectively, and resumes an interrupted task about one time in two. The causal profile describes the architecture itself: five of the six causal tests receive `STRUCTURAL_ABSENT`, as preregistered for a standard transformer. The only executable one, R3, measures the persistence of an iterative computation at fixed depth. From one step to the next, the only causal feedback of a standard transformer passes through the emitted token; the causal profile records this rather than inferring it.

### Scope

One model, 8-bit weights, 12 of the 42 tests, one seed (the confidence intervals reflect variability across instances of that seed; repeating on other seeds is a natural extension). **Not a consciousness score.**

## Intended use

The primary intended use is the **controlled comparison between variants of a single architecture** — a standard transformer against a latent-loop variant, for example — where grain, test battery, readout and horizon stay fixed and only the architecture varies. The measurements above are the first term of that comparison: they characterize standard Qwen3-8B before any modification, with the same instruments, decision rules and fingerprints that will be applied to its variants. They are a reference point, not a ranking of models. For FCA-Bench, the expected signature of a latent loop is the passage of a causal test from `STRUCTURAL_ABSENT` to a measured value at a comparable behavioral profile; for PCI_T, a variation of directional specificity that can be confronted with that change. A convergence would strengthen each instrument; a dissociation should be reported as it stands.

## Reading order

1. This file.
2. `16-pci-t-directional-specificity/` — protocol first, then the synthetic bench, the Qwen3-8B calibration and replication, and the verifier.
3. `17-fca-bench/` — protocol first, then the runner, then the Qwen3-8B instance-by-instance results.

## Limits, stated once

- Both campaigns concern a single model, Qwen3-8B, with numerical choices (half precision for PCI_T, 8-bit weights for FCA-Bench) whose effect has not been estimated against a full-precision run.
- The PCI_T replication rests on two sets of six episodes.
- FCA-Bench is applied to 12 of its 42 tests, with one seed.
- None of the reported values is a measure of consciousness. Both instruments target functional aspects of access; their relation to access consciousness would require external validation that this archive does not provide.
