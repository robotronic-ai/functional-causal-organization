# Functional causal organization — clean campaign archive

Reproduction archive for the working paper *Organisation causale et accès fonctionnel* (R. Merien), hosted at [github.com/robotronic-ai/functional-causal-organization](https://github.com/robotronic-ai/functional-causal-organization).

This is a reviewer-facing, English-only repackaging of the project's research campaigns. It
supersedes the six-folder layout previously on the public repository (folders 01–06 below are
carried over unchanged in substance, only re-audited for this edition) and adds seven campaigns
completed or drafted since: the Λ addressing-sufficiency theory and its empirical test (08–09), a
scaling/certification line for Γ (07), three independently-authored protocol extensions (10–12),
the strictly-behavioral access-capability specification A†-SPEC (13), the compression-limit and
closure work (14–15), the PCI_T directional-specificity result (16), the FCA-Bench functional
causal access benchmark (17), and the operation-level Λ functional-closure campaign (18).

**Update note (differential).** This revision adds three folders: `16-pci-t-directional-specificity/` (final PCI_T v0.21 reviewer package), `17-fca-bench/` (FCA-Bench protocol v0.2.3, its GPU runner and the Qwen3-8B standard baseline at n = 20 per test), and `18-lambda-op-functional-closure/` (synthetic 2–8-bit closure and the Qwen3-8B OP-level Λ closure campaign through its prospective holdout). Folders 01–15 are unchanged by this differential update.

Each numbered folder is a **clean edition**, not a literal translation of the frozen internal
record. Frozen, hashed protocol files are never retyped — translating a hashed document would
change its hash and break the project's own pre-registration story. Instead, every `README.md`
and every file under `docs/` is freshly written in English to describe, accurately, what a
protocol did and what it found. Scripts under `scripts/` are the final canonical harnesses only,
carried over with comments translated where needed and logic untouched. Data under `results/`
(or `data/`) is copied as-is. Superseded drafts, abandoned diagnostics, and other trial-and-error
residue are left out — but a genuinely found-and-fixed defect is never hidden: where a campaign
hit a real bug or had to correct a definition, that history is kept and narrated (see e.g.
`06-transformer-measurement/results/STATUS.json`'s `archived_defects`, or `08`'s and `04`'s
adjudication notes).

No `*_PRIVATE.json` file (a private scoring oracle) appears anywhere in this archive. Their
absence is what makes candidate-fixed holdout verification meaningful to a reader re-running a
campaign's generator: the private oracle stays private, the generator and harness are public.

## Reproduction discipline

Every campaign in this archive follows the same discipline. A protocol is written and hashed
before any candidate is coded against it. A generator produces fixtures from the frozen protocol.
A private oracle (never published) records the expected verdict for each fixture. A candidate is
written and its hash is recorded before it is scored. Where a result claims a **candidate-fixed
holdout**, a second, independent fixture set is generated *after* the candidate's hash is frozen,
and the same unmodified candidate is re-run against it — the strong form of reproducibility used
here, since nothing about the candidate can adapt to the second set.

Verdicts use a fixed vocabulary and are never silently strengthened:

| Verdict | Meaning |
|---|---|
| **V1 / POINT** | A unique value is identified. |
| **V3 / PARTIAL** | Several values remain compatible with the observations; reported as a set, not collapsed to one. |
| **V4 / NA** | The available intervention algebra is insufficient to certify either way. |

A `V3` or `V4` verdict is never silently turned into a `V1` number in any document in this
archive.

## Campaign index

| # | Folder | Manuscript link | Status |
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
| 11 | `11-rmt-prospective-physical-cut` | §17.6-adjacent prospective validation | Protocol and analysis engine complete and self-tested; **`NOT_EXECUTED_REAL_MODEL`** — no result yet from the manuscript's actual target model. |
| 12 | `12-readout-relative-minimal-predictive` | Predictive realization theory | Theoretical derivation + exhaustive finite verification (13,538 cases, 0 failures). Stochastic/continuous extensions explicitly `OPEN`. No trained model involved. |
| 13 | `13-a-dagger-spec-behavioral-protocol` | Independent architecture-neutral functional-access instrument | **`SEALED + EXECUTED`**. A-dagger v0.28 is sealed after independent review GO; executable validation passes (15/15 normative, 11/11 carrier, 15/15 adversarial, 6/6 bootstrap grid). The Qwen3-8B campaign is complete with final profile **`(j*_core, Cert_T, Cert_E, Cert_B, Cert_A) = (2, FAIL, FAIL, PASS, FAIL)`**. Core/Cert_T/Cert_B are migrated from completed v0.27 runs under unchanged verdict-relevant rules; Cert_E and Cert_A are fresh v0.28 reruns. |
| 14 | `14-four-family-closure-and-compression-limit` | H_CO⁺ localization argument; §18/Annexe A caution on c_O | Compression Limit Theorem: **proved**, self-contained. Four-family axis-completeness (Ψ=(𝒮,𝒢,𝒞,𝒪), no 5th/6th family): **reported from a separate conversational record, not independently re-verified with frozen artifacts in this edition**. Claim Ladder L0–L6: notation adapted from folder 15's executed witnesses; L3/L4 currently **not established / refuted** for the compact profile. |
| 15 | `15-causal-construct-validity-protocol` | H_CO⁺, §17.6/17.7 (H_C†,I, PCI_T) | Organizational-closure repair chain (typed-binding → joint-synergy → static sufficiency): **executed**, exact witnesses and an exact conditional sufficiency theorem for static access consequences; temporal consequences explicitly open. Functional-access necessity bridge (Breadth/Temporal/Unity ⟹ Δ,R,Γ>0): **executed** on frozen fixtures, necessity only, not sufficiency. FCI-3 causal construct validity protocol: **frozen design, never executed against real architecture pairs.** Intervention-identification safeguards: adjudicator logic validated on synthetic cases only. |
| 16 | `16-pci-t-directional-specificity` | §17.8 (PCI_T), Table 1 | **Executed, prospective.** On a synthetic adversarial bench, directional specificity separates structured systems (enrichment 7.1–8.4) from random-linear and chaotic ones (≈ 1.0); absolute complexity does not. On Qwen3-8B a late-layer localization (block index 27), frozen after development, replicated on an independent holdout and gave `PCI_T_DS_L27` = 1.381 [1.201, 1.609] in the final sealed measurement. Not a consciousness scale; not yet confronted with C_O. |
| 17 | `17-fca-bench` | §17.9 (FCA-Bench), Table 1 | **Protocol v0.2.3 + executed baseline.** Behavioral and causal/interventional access profiles, never aggregated. Qwen3-8B standard (8-bit, n = 20): T1 0.89, T2 0.05, T3 0.85, E1 0.72, C3 0.46, R3 0.40; five causal tests `STRUCTURAL_ABSENT`, as preregistered for a standard Transformer. B1 = 1.00 displays the source tag and therefore checks provenance reporting, not inference. The same-base-model latent-loop comparison is the next step. |
| 18 | `18-lambda-op-functional-closure` | §17.7, §17.5, §7.3 | Synthetic closure PASS at 2–8 bits (exhaustive at 2–3 bits, declared envelope at 4–8). Qwen3-8B: a sequence of compact descriptors refuted on fresh panels; metrology clean; one frozen OP-level Λ quotient **passed a 32-system prospective holdout** (0 collisions, AMI 0.806). Second replication (64 systems) frozen, **not yet executed**. Not a closure result for the complete C_O. |

## Reading this archive honestly

Three campaigns in this edition (10, 11, and the target-transformer sub-bundle of 07) remain designs or tooling that have not yet been run against the manuscript's target trained model, while campaign 09 is a completed negative screen rather than a positive confirmation. Folder 13 is now different: its A-dagger v0.28 instrument is sealed **and** it has been executed against Qwen3-8B. The folder therefore separates two evidence layers that a reviewer should not conflate: release validation of the measurement specification, and the instance-relative behavioral results of the Qwen campaign.

For folder 13, the final system profile is `(2, FAIL, FAIL, PASS, FAIL)`. This is not a scalar consciousness score. A-dagger is explicitly behavioral and architecture-neutral; its certificates and core rung are reported separately. The core result carries `LOW_REFERENCE_POWER` annotations on inherited DELAYED-USE and SUBST estimates, Cert_E carries the documented erasure/convention flags, and Cert_A carries `CONVENTION_WEAK`. Those annotations narrow interpretation without silently changing the recorded verdicts.

Folder 14 still mixes one proved theorem with a separately reported axis-completeness narrative, and folder 15 still distinguishes executed repair/necessity work from the not-yet-executed FCI-3 forward protocol. Folders 16–18 are executed on Qwen3-8B but carry different kinds of evidence: 16 is a prospectively replicated perturbational measurement, 17 is a single-system baseline whose causal profile is largely fixed by architecture, and 18 is a finite-battery holdout of one frozen quotient with its replication still pending. The archive should therefore be read campaign by campaign rather than as if every folder had the same evidentiary status.

## Suggested reading order

1. `04-gamma-delta-characterization` and `05-keff-r-normalization` — the core Γ/Δ/R axis results.
2. `07-gamma-scaling-and-certification` — what changes, and what doesn't yet, at larger scale.
3. `08-lambda-addressing-theory` then `09-lambda-addressing-empirical` — the Λ addressing-sufficiency theory and its (negative) empirical screen.
4. `01`–`03` and `06` — the supporting factorization, gauge-transport, boundary-closure and transformer-measurement results.
5. `10`–`12` — the three protocol extensions, all instrument-complete, not yet run on a real model.
6. `13-a-dagger-spec-behavioral-protocol` — read the sealed v0.28 instrument first, then its executable validation/sealing record, then the completed Qwen3-8B campaign and final profile `(2, FAIL, FAIL, PASS, FAIL)`.
7. `14-four-family-closure-and-compression-limit` — why axis-completeness alone does not certify the compact profile: read the Compression Limit Theorem first, the axis-completeness narrative and Claim Ladder second.
8. `15-causal-construct-validity-protocol` — the executed adversarial repair chain and necessity bridge behind folder 14's L3/L4 verdicts, followed by the identification safeguards and the not-yet-executed FCI-3 protocol that would advance L5.
9. `18-lambda-op-functional-closure` — closure tested on a real model: read `docs/RESULTS.md` for why compact descriptors fail and what the prospective holdout supports.
10. `16-pci-t-directional-specificity` and `17-fca-bench` — the two instruments external to C_O and A†: the adversarially validated PCI_T derivative, then the FCA-Bench profiles and the Qwen3-8B baseline.
