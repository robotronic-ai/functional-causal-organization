# A-dagger v0.28 — sealed behavioral protocol and Qwen3-8B campaign

This folder is the clean reviewer-facing edition of A-dagger. It replaces the previous contents of folder 13 rather than adding a new numbered campaign.

A-dagger v0.28 is a sealed, strictly behavioral functional-access specification. It does not measure consciousness and does not collapse its outputs into a scalar consciousness score. The release passed its executable validation suite after an independent human review decision of GO.

The same folder now also contains the completed Qwen3-8B measurement campaign. The final profile is:

```text
(j*_core, Cert_T, Cert_E, Cert_B, Cert_A) = (2, FAIL, FAIL, PASS, FAIL)
```

The core ladder, Cert_T and Cert_B are carried forward from completed v0.27 runs because their measured primitives, target/null contrasts and verdict-relevant scoring rules remain applicable under v0.28. No observations were altered or re-estimated. Cert_E and Cert_A were rerun with fresh v0.28 score-producing data because v0.28 changed rules that had affected their earlier executions.

## Layout

- `spec/` — sealed v0.28 specification, normative implementation description, adjudicated W1-W6 grid, independent-review record, sealing record, release manifest and folder hash manifest.
- `scripts/` — final normative validation scripts plus the exact canonical Qwen campaign runners.
- `results/protocol-validation/` — final executable validation and finite characterization artifacts.
- `results/qwen3-8b/` — final campaign profile, campaign report/manifest, raw per-round results, decision proofs, verification files and migration records where applicable.
- `docs/` — concise reviewer-oriented reading guides.

## Final campaign result

- core ladder: `j*_core = 2`
- `Cert_T = FAIL`
- `Cert_E = FAIL`
- `Cert_B = PASS`
- `Cert_A = FAIL`

Annotations are kept separate from verdicts. The inherited DELAYED-USE and SUBST estimates carry `LOW_REFERENCE_POWER`; HARD-ERASURE carries `CONVENTION_UNVERIFIED` and `PREREQUISITE_WEAK`; SELECTIVE-ERASURE carries `CONVENTION_UNVERIFIED`; CAUSAL-OWNERSHIP carries `CONVENTION_WEAK`.

## Reviewer reading order

1. `spec/A_DAGGER_SPEC_v0.28.md`
2. `spec/A_DAGGER_NORMATIVE_IMPLEMENTATION_v0.28.md`
3. `spec/A_DAGGER_W1_W6_VERIFICATION_GRID_v0.28_ADJUDICATED.md`
4. `docs/external-sealing-review.md`
5. `results/protocol-validation/EXECUTABLE_VALIDATION_REPORT_v0.28.md`
6. `docs/qwen3-8b-campaign.md`
7. `results/qwen3-8b/CAMPAIGN_REPORT.md` and `FINAL_PROFILE.json`
8. per-round raw results and verification files under `results/qwen3-8b/`

This edition intentionally omits issue-history files, superseded drafts, pre-run packages, failed development attempts and duplicated intermediate manifests.
