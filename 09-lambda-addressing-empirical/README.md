# 09 — Λ addressing empirical validation

Clean edition of the empirical campaign backing manuscript §17.5 ("Pont d'adressage") and the
§7.3 Λ discussion: does a real trained model ever produce two states that are equal under a
declared, frozen screening representation while differing in the Λ target, and — separately —
can a purpose-built system be exhibited whose declared baseline channel matches exactly while
a late causal degree of freedom still separates it?

This repository deliberately removes debugging/hotfix history. It retains only the canonical scientific stages, frozen identifiers, final result summaries, raw canonical result files available in the evidence set, and reproducibility checks.

## Reproduction discipline

This campaign follows the same discipline as every other campaign in this repository (see the
root `README.md` for the full statement), expressed in this campaign's own vocabulary. The
Real-C0 and rich-Delta screens are protocol-frozen and hash-verified before any candidate
search: model checkpoints, the validation dataset, and the declared intervention family are
all fixed and hashed in `HASHES.json` before Λ descriptors and task outcomes are sealed — the
role played elsewhere in this repository by a generator plus a private oracle. Stage A can
only nominate candidate fibers under the frozen baseline representation; Stage B, where Λ
would actually be opened, never instantiates unless Stage A first produces a nonempty
candidate list — this anti-circularity gate is this campaign's analogue of a
candidate-fixed holdout: the screening representation is fixed before Λ is examined, not
adjusted afterward to fit it. The one stage that does not follow this screen-then-open
structure, Stage 4's constructive late-role-rebind witness, is instead an exact,
fully-specified executable system (`scripts/run_lambda_rebind_witness_v3160.py`) whose
baseline-equality and gate-separation claims are recomputed from scratch, not fit to prior
data. Where the other campaigns in this repository report verdicts as `V1/POINT` /
`V3/PARTIAL` / `V4/NA`, this campaign reports its five claims as `ESTABLISHED`,
`NOT ESTABLISHED`, `WITHHELD`, or `OPEN` (see `docs/RESULTS.md` and `STATUS.json`) — the same
discipline of never silently upgrading a negative or partial finding into a positive one.

## Scientific arc

1. **Frozen Lambda lineages and data anchors** established the reusable model/data provenance.
2. **Real-C0 fiber search** asked whether equal baseline screening representations could coexist with a Lambda difference. No candidate fiber was found.
3. **Rich-Delta fiber search** replaced scalar/paired summaries with the full typed four-probe perturbation-response channel. No candidate fiber was found.
4. **Layer-complete rich-Delta extension** expanded the same declared head-swap intervention family to all 12 layers. The measured rich-Delta object remained injective on all 1,586 states at tolerance `1e-7`; no Stage-B candidate existed.
5. **Constructive late-role-rebind witness** deliberately built a fiber-like use case rather than searching for one. The declared baseline typed channel was exactly identical, while a late context-to-role causal channel separated maximally.

## Main results

- Real-C0 nonderivability: **NOT ESTABLISHED — NO FIBER WITNESS**.
- Real rich-Delta nonreduction: **NOT ESTABLISHED — RICH DELTA SEPARATED ALL DECLARED STATES**.
- Global C0-Lambda promotion: **WITHHELD**.
- Universal Lambda necessity: **OPEN**.
- Constructive late-role-rebind witness: **ESTABLISHED** within its declared synthetic use case.

The constructive witness is not a retroactive PASS of the frozen RMT Stage-B gate.

## Quick verification

```bash
python scripts/verify_campaign.py
```

The constructive witness can also be rerun:

```bash
python scripts/run_lambda_rebind_witness_v3160.py --output /tmp/rebind.json
```

The scripts use only the Python standard library.
