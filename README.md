# Functional causal organization — research archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23124247.svg)](https://doi.org/10.5281/zenodo.23124247)

Article: *Measuring Functional Access in a Language Model by Intervention: PCI_T and FCA-Bench* — HAL [hal-05776792](https://hal.science/hal-05776792) · Zenodo [10.5281/zenodo.23375827](https://doi.org/10.5281/zenodo.23375827)

Repository: [github.com/robotronic-ai/functional-causal-organization](https://github.com/robotronic-ai/functional-causal-organization)

**License:** code under the [MIT License](LICENSE); data, results and documentation under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [License](#license).

## What this archive is

A reviewer-facing, English-only edition of the project's research campaigns, in 20 numbered folders. Two of them are the instruments described in the article above:

- **PCI_T** — a perturbational complexity index for transformers, and its *directional specificity* (folder `16-pci-t-directional-specificity/`);
- **FCA-Bench** — the Functional Causal Access Benchmark, with a behavioral and a causal profile that are never aggregated (folder `17-fca-bench/`).

The article is self-contained; folders 16 and 17 are where its numbers can be re-derived and its protocols inspected. The other folders are companion campaigns, theoretical and empirical, with very different evidentiary status: read the index below campaign by campaign. Folders 19 and 20 hold the measurements and frozen protocols for the causal-organization profile C_O = (Γ, Δ, R, Λ) on Qwen3-8B.

**Version note.** The badge above is the Zenodo *concept* DOI: it always resolves to the latest release (currently v1.6). Release v1.5.1 (Zenodo [10.5281/zenodo.23124248](https://doi.org/10.5281/zenodo.23124248)) archived folders 16 and 17 only, i.e. the article's package. Release 1.6 adds folders 01–15 and 18–20. Folders 16 and 17 are unchanged between the two releases.

**About the section references.** Several index entries refer to sections of a long working paper. That working paper is **not published** and is not part of this archive. The references are given for provenance only; every folder is meant to be readable on its own.

Each numbered folder is a **clean edition**, not a literal translation of the frozen internal record. Frozen, hashed protocol files are never retyped: translating a hashed document would change its hash and break the pre-registration. Instead, every `README.md` and every file under `docs/` is freshly written in English to describe, accurately, what a protocol did and what it found. Scripts under `scripts/` are the final canonical harnesses only, carried over with comments translated where needed and logic untouched. Data under `results/` (or `data/`) is copied as-is. Superseded drafts, abandoned diagnostics and other trial-and-error residue are left out — but a genuinely found-and-fixed defect is never hidden: where a campaign hit a real bug or had to correct a definition, that history is kept and narrated (see e.g. `06-transformer-measurement/results/STATUS.json`'s `archived_defects`, or the adjudication notes of `08` and `04`).

No `*_PRIVATE.json` file (a private scoring oracle) appears anywhere in this archive. Their absence is what makes candidate-fixed holdout verification meaningful to a reader re-running a campaign's generator: the private oracle stays private, the generator and harness are public.

## Reproduction and measurement discipline

Every campaign follows the same discipline.

- **Protocol first.** A protocol is written and hashed before any candidate is coded against it. A generator produces fixtures from the frozen protocol. A private oracle (never published) records the expected verdict for each fixture. A candidate is written and its hash recorded before it is scored.
- **Candidate-fixed holdout.** Where a result claims one, a second, independent fixture set is generated *after* the candidate's hash is frozen, and the same unmodified candidate is re-run against it — the strong form of reproducibility used here, since nothing about the candidate can adapt to the second set.
- **Intervention, not self-report.** For the instruments of folders 16 and 17, nothing relies on what the model says about itself. PCI_T acts on activations; the causal profile of FCA-Bench requires that the tested operation be exposed and executed; the behavioral profile compares matched arms.
- **Prospective.** Measurement rules (site, amplitude, aggregation, decision threshold) are frozen before the data that test them are opened. Each dataset is sealed by a cryptographic fingerprint. No adjustment is allowed after opening. Model weights stay frozen throughout.
- **No single score.** Neither instrument of the article produces a global ranking.

Verdicts use a fixed vocabulary and are never silently strengthened:

| Verdict | Meaning |
|---|---|
| **V1 / POINT** | A unique value is identified. |
| **V3 / PARTIAL** | Several values remain compatible with the observations; reported as a set, not collapsed to one. |
| **V4 / NA** | The available intervention algebra is insufficient to certify either way. |

A `V3` or `V4` verdict is never silently turned into a `V1` number in any document of this archive.

## Campaign index

| # | Folder | Working-paper reference (unpublished) | Status |
|---|---|---|---|
| 01 | `01-persistent-active-factorization` | §8.2–§8.3 | Closed: exact preregistered collision witness, not a working general extractor — the campaign's own conclusion. |
| 02 | `02-gauge-transport-identification` | §8.4 area (gauge/fiber transport) | Closed: exact gauge-ambiguity and transport-closure characterization. |
| 03 | `03-canonical-boundary-closure` | Boundary/ρ-certification | Closed: theorem + certification + composition results. |
| 04 | `04-gamma-delta-characterization` | §7.1–§7.2, §16.6, §18 | Γ conditionally axiom-characterized (needs one additional explicit axiom); Δ exactly decomposed; Γ/Δ independence witnessed; prospective + candidate-fixed-holdout validation both 32/32. |
| 05 | `05-keff-r-normalization` | K_eff / R | Closed: definition, invariance and normalization theorem, finite audit. Analytic only, no scripts needed. |
| 06 | `06-transformer-measurement` | Transformer re-entry measurement | Closed, with two archived-and-fixed defects documented rather than hidden. |
| 07 | `07-gamma-scaling-and-certification` | §7.1 scaling / certification | Mixed — see below. |
| 08 | `08-lambda-addressing-theory` | §17.5 "Pont d'adressage", §7.3 | Closed constructive theory: 7 theorems, 4 exact finite witness domains, all PASS. Status `STABILIZED-CONSTRUCTIVE-THEORY`. |
| 09 | `09-lambda-addressing-empirical` | §17.5, §7.3 | Screening protocol executed on a real model across 3 stages (up to ~1.26M state pairs); **zero candidate fibers found**, so Λ itself was never opened for scoring. Universal Λ-necessity remains **OPEN**, not settled either way. |
| 10 | `10-continuous-predictive-geometry` | Predictive-state theory, continuous extension | Pure math + exact finite/closed-form verification only. **Not yet applied to any trained model.** |
| 11 | `11-rmt-prospective-physical-cut` | §17.6-adjacent prospective validation | Protocol and analysis engine complete and self-tested; **`NOT_EXECUTED_REAL_MODEL`** — no result yet from the target model of the working paper. |
| 12 | `12-readout-relative-minimal-predictive` | Predictive realization theory | Theoretical derivation + exhaustive finite verification (13,538 cases, 0 failures). Stochastic/continuous extensions explicitly `OPEN`. No trained model involved. |
| 13 | `13-a-dagger-spec-behavioral-protocol` | Independent architecture-neutral functional-access instrument | **`SEALED + EXECUTED`**. A-dagger v0.28 is sealed after independent review GO; executable validation passes (15/15 normative, 11/11 carrier, 15/15 adversarial, 6/6 bootstrap grid). The Qwen3-8B campaign is complete with final profile **`(j*_core, Cert_T, Cert_E, Cert_B, Cert_A) = (2, FAIL, FAIL, PASS, FAIL)`**. Core/Cert_T/Cert_B are migrated from completed v0.27 runs under unchanged verdict-relevant rules; Cert_E and Cert_A are fresh v0.28 reruns. |
| 14 | `14-four-family-closure-and-compression-limit` | H_CO⁺ localization argument; §18/Annexe A caution on c_O | Compression Limit Theorem: **proved**, self-contained. Four-family axis-completeness (Ψ=(𝒮,𝒢,𝒞,𝒪), no 5th/6th family): **reported from a separate conversational record, not independently re-verified with frozen artifacts in this edition**. Claim Ladder L0–L6: notation adapted from folder 15's executed witnesses; L3/L4 currently **not established / refuted** for the compact profile. |
| 15 | `15-causal-construct-validity-protocol` | H_CO⁺, §17.6/17.7 (H_C†,I, PCI_T) | Organizational-closure repair chain (typed-binding → joint-synergy → static sufficiency): **executed**, exact witnesses and an exact conditional sufficiency theorem for static access consequences; temporal consequences explicitly open. Functional-access necessity bridge (Breadth/Temporal/Unity ⟹ Δ,R,Γ>0): **executed** on frozen fixtures, necessity only, not sufficiency. FCI-3 causal construct validity protocol: **frozen design, never executed against real architecture pairs.** Intervention-identification safeguards: adjudicator logic validated on synthetic cases only. |
| 16 | `16-pci-t-directional-specificity` | §17.8 (PCI_T), Table 1 | **Executed, prospective.** On a synthetic adversarial bench, directional specificity separates structured systems (enrichment 7.1–8.4) from random-linear and chaotic ones (≈ 1.0); absolute complexity does not. On Qwen3-8B a late-layer localization (block index 27), frozen after development, replicated on an independent holdout and gave `PCI_T_DS_L27` = 1.381 [1.201, 1.609] in the final sealed measurement. Not a consciousness scale; not yet confronted with C_O. |
| 17 | `17-fca-bench` | §17.9 (FCA-Bench), Table 1 | **Protocol v0.2.3 (B1: v0.2.4; T2: v0.2.7) + executed baseline.** Behavioral and causal/interventional access profiles, never aggregated. Qwen3-8B standard (8-bit, n = 20): T1 0.89, T2 0.26 [0.11, 0.41], T3 0.85, B1 0.87, E1 0.72, C3 0.46, R3 0.40; five causal tests `STRUCTURAL_ABSENT`, as preregistered for a standard Transformer. The same-base-model latent-loop comparison is the next step. |
| 18 | `18-lambda-op-functional-closure` | §17.7, §17.5, §7.3 | Synthetic closure PASS at 2–8 bits (exhaustive at 2–3 bits, declared envelope at 4–8). Qwen3-8B: a sequence of compact descriptors refuted on fresh panels; metrology clean; one frozen OP-level Λ quotient **passed a 32-system prospective holdout** (0 collisions, AMI 0.806). Second replication (64 systems) frozen, **not yet executed**. Not a closure result for the complete C_O. |
| 19 | `19-co-qwen3-8b-c-l24` | §17.5bis | **Executed.** Contract C_L24 on Qwen3-8B (bf16, layer 24 → 25, world u26): Γ, Δ, R (sampled transition) and Λ₀ each `NONZERO_CERTIFIED` on this one declared contract; R on the greedy transition is measured exactly zero. Non-nullity only, not magnitude, not a closure. Includes an exploratory, non-certified perturbation pilot (manuscript Table 17.13). |
| 20 | `20-co-closure-and-rivals-preregistration` | §17.7, §17.6 | **Frozen, not executed.** Protocols for the graded functional closure and for the comparison of C_O against preregistered rivals on FCA-E1 and PCI_T_DS_L27. Closure `OPEN_NOT_ESTABLISHED`; rival superiority `NOT_TESTED`. |

## Reading this archive honestly

Three campaigns (10, 11, and the target-transformer sub-bundle of 07) remain designs or tooling that have not yet been run against the target trained model, while campaign 09 is a completed negative screen rather than a positive confirmation. Folder 13 is different: its A-dagger v0.28 instrument is sealed **and** it has been executed against Qwen3-8B. The folder therefore separates two evidence layers that a reader should not conflate: release validation of the measurement specification, and the instance-relative behavioral results of the Qwen campaign.

For folder 13, the final system profile is `(2, FAIL, FAIL, PASS, FAIL)`. This is not a scalar consciousness score. A-dagger is explicitly behavioral and architecture-neutral; its certificates and core rung are reported separately. The core result carries `LOW_REFERENCE_POWER` annotations on inherited DELAYED-USE and SUBST estimates, Cert_E carries the documented erasure/convention flags, and Cert_A carries `CONVENTION_WEAK`. Those annotations narrow interpretation without silently changing the recorded verdicts.

Folder 14 still mixes one proved theorem with a separately reported axis-completeness narrative, and folder 15 still distinguishes executed repair/necessity work from the not-yet-executed FCI-3 forward protocol. Folders 16–19 are executed on Qwen3-8B but carry different kinds of evidence: 16 is a prospectively replicated perturbational measurement, 17 is a single-system baseline whose causal profile is largely fixed by architecture, 18 is a finite-battery holdout of one frozen quotient with its replication still pending, and 19 certifies non-nullity of four components on one declared contract. Folder 20 contains frozen protocols only and no result. The archive should therefore be read campaign by campaign, not as if every folder had the same evidentiary status.

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

## `19-co-qwen3-8b-c-l24/`

### Question

Is each component of the organizational profile C_O = (Γ, Δ, R, Λ) non-zero in a language model, on a declared causal interface?

### Contract and results

The contract **C_L24** fixes Qwen3-8B in bf16, the passage from layer 24 to layer 25, world u26, twelve donor sites with frozen donors, twelve frozen readers, ten gauge replicas (identity and nine exact recodings) and 4,096 donor masks. Eleven roles are retained; one site is context.

| Component | Status on C_L24 | Certified bound |
|---|---|---|
| Γ (integration) | `NONZERO_CERTIFIED` | ≥ 1.567e-4 (measured minimum 0.2891) |
| Δ (differentiation) | `NONZERO_CERTIFIED` | ≥ 6.29e-4 (per-reader game: 0.4) |
| R (re-entry), sampled transition, role mlp0 | `NONZERO_CERTIFIED` | R_state ≥ 7.39e-8 |
| R, greedy transition | measured exactly zero | no donor configuration changes the emitted token |
| Λ₀ (typed content → role) | `NONZERO_CERTIFIED` | 68 of 132 cells at the primary threshold; 11 of 11 contents |

Each certificate was verified independently of the measurement chain (raw witnesses for Γ, exact game for Δ, recomputation from the raw export for R, audit plus witness extract for Λ₀). The bounds certify non-nullity; they do not quantify magnitude.

### Scope

Each statement is relative to the C_L24 contract, that is, to the layer 24 → 25 interface. Another interface (later layers, other horizons, other intervention alphabets) is certified by its own contract with the same protocol; non-nullity at 24 → 25 does not logically entail non-nullity elsewhere. It does support the existence statement: Qwen3-8B has non-zero causal organization on a declared contract. The bundle is **not** a functional closure of C_O (see folder 20).

### Provenance and verification

The folder is the publication bundle as produced, with its own `README.md`, `CONTRACT.md`, `METHODS.md`, `RESULTS.md`, `REPRODUCE.md` and `MANIFEST_SHA256.txt` (152 files; all digests match). Scripts keep their acquisition-time names and pinned input paths, because the pins check digests; they are therefore not renamed. Raw arrays are not included; `VERIFY_NPZ_SHA.py` checks a local copy against the recorded digests. `exploratory_t346/` is a descriptive perturbation pilot and does not change any status above.

## `20-co-closure-and-rivals-preregistration/`

Frozen protocols, byte-identical to their hashed originals: graded functional closure (direct within-cell pair criterion, simultaneous uncertainty, five verdicts) and a rival comparison (`M0`, `M1_CO`, `MR_INFLUENCE4`) on two external targets, `FCA_E1` and `PCI_T_DS_L27`. The joint dataset they require (numerical C_O and both targets on at least five groups of independent matched realizations) has not been acquired. Closure is `OPEN_NOT_ESTABLISHED` and rival superiority is `NOT_TESTED`. See the folder's `README.md`.

## Suggested reading order

1. The article (HAL or Zenodo link above), then `16-pci-t-directional-specificity/` — protocol first, then the synthetic bench, the Qwen3-8B calibration and replication, and the verifier — and `17-fca-bench/` — protocol first, then the runner, then the Qwen3-8B instance-by-instance results.
2. `04-gamma-delta-characterization` and `05-keff-r-normalization` — the core Γ/Δ/R axis results.
3. `07-gamma-scaling-and-certification` — what changes, and what doesn't yet, at larger scale.
4. `08-lambda-addressing-theory` then `09-lambda-addressing-empirical` — the Λ addressing-sufficiency theory and its (negative) empirical screen.
5. `01`–`03` and `06` — the supporting factorization, gauge-transport, boundary-closure and transformer-measurement results.
6. `10`–`12` — the three protocol extensions, all instrument-complete, not yet run on a real model.
7. `13-a-dagger-spec-behavioral-protocol` — read the sealed v0.28 instrument first, then its executable validation/sealing record, then the completed Qwen3-8B campaign and final profile `(2, FAIL, FAIL, PASS, FAIL)`.
8. `14-four-family-closure-and-compression-limit` — why axis-completeness alone does not certify the compact profile: read the Compression Limit Theorem first, the axis-completeness narrative and Claim Ladder second.
9. `15-causal-construct-validity-protocol` — the executed adversarial repair chain and necessity bridge behind folder 14's L3/L4 verdicts, followed by the identification safeguards and the not-yet-executed FCI-3 protocol that would advance L5.
10. `18-lambda-op-functional-closure` — closure tested on a real model: read `docs/RESULTS.md` for why compact descriptors fail and what the prospective holdout supports.
11. `19-co-qwen3-8b-c-l24` — `README.md`, `CONTRACT.md`, `RESULTS.md`; then `REPRODUCE.md` and the verifiers.
12. `20-co-closure-and-rivals-preregistration` — `README.md`, then the frozen protocols.

## License

© 2026 Ronan Merien.

| Material | License |
|---|---|
| Source code (scripts, harnesses, generators, verifiers) | [MIT License](LICENSE) |
| Data, raw results, protocols, documentation and this README | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |

Unless a file states otherwise, these terms apply to everything under the 20 numbered folders. The CC BY 4.0 license requires attribution: cite the work as described below and indicate any changes you make. The MIT License requires that the copyright notice and permission notice be kept in copies of the code.

The licenses cover this archive's contents. The Qwen3-8B model used in the experiments is a third-party artifact (Qwen Team) distributed under its own terms; it is not included here and is not relicensed by this repository.

## How to cite

If you use the instruments of folders 16 and 17, please cite the article:

> Merien, R. (2026). *Measuring Functional Access in a Language Model by Intervention: PCI_T and FCA-Bench*. HAL hal-05776792; Zenodo, https://doi.org/10.5281/zenodo.23375827.

For the code, data and other campaigns, cite the archived release of this repository: the DOI and release tag are given in `CITATION.cff`, in the release notes and on the Zenodo record. GitHub's "Cite this repository" button reads `CITATION.cff`.

## Reproducibility and integrity

- Each folder fixes the SHA-256 fingerprint of its files; verify them before reuse.
- Cite a **release tag or commit hash**, not only the repository URL: a branch can move, a tag and its archived release cannot.

## Intended use

The primary intended use of the instruments of folders 16 and 17 is the **controlled comparison between variants of a single architecture** — a standard transformer against a latent-loop variant, for example — where grain, test battery, readout and horizon stay fixed and only the architecture varies. The measurements above are the first term of that comparison: they characterize standard Qwen3-8B before any modification, with the same instruments, decision rules and fingerprints that will be applied to its variants. They are a reference point, not a ranking of models. For FCA-Bench, the expected signature of a latent loop is the passage of a causal test from `STRUCTURAL_ABSENT` to a measured value at a comparable behavioral profile; for PCI_T, a variation of directional specificity that can be confronted with that change. A convergence would strengthen each instrument; a dissociation should be reported as it stands.

## Limits, stated once

- The campaigns of folders 16 and 17 concern a single model, Qwen3-8B, with numerical choices (half precision for PCI_T, 8-bit weights for FCA-Bench) whose effect has not been estimated against a full-precision run.
- The PCI_T replication rests on two sets of six episodes.
- FCA-Bench is applied to 12 of its 42 tests, with one seed.
- None of the reported values is a measure of consciousness. The instruments target functional aspects of access; their relation to access consciousness would require external validation that this archive does not provide.
- The C_O certificates (folder 19) are non-nullity statements on one finite contract, not magnitudes and not a closure; closure and rival superiority are untested (folder 20).
- The other folders carry the status stated in the campaign index; none of them should be read as stronger than that status.
