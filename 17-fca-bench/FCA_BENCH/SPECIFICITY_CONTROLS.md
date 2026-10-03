# Specificity Controls

## Purpose

FCA-Bench should distinguish access organization from generic language-model competence, model scale, extra inference compute, and verbal compliance.

The controls below are part of the recommended experimental design. They are not additional FCA dimensions.

## Control C0 — Behavioral success without the target mechanism

Construct a reference policy, replay system, lookup-style controller, or transcript-conditioned program that can solve selected behavioral tasks but does not implement the target internal primitive.

Expected pattern:

- selected behavioral tests: high;
- causal/interventional tests requiring absent primitives: `structural_absent`.

This is a sanity check that the benchmark does not infer internal organization from output quality alone.

## Control C1 — Standard versus transformed architecture

Use the same base model and compare:

1. standard decoder Transformer;
2. transformed architecture with the target mechanism enabled.

Keep the following fixed wherever possible:

- base weights;
- tokenizer;
- prompt and system-message format;
- item seeds;
- decoding parameters;
- scoring code;
- hardware precision.

This is the primary causal contrast.

## Control C2 — Compute-matched alternative

When the transformed architecture uses more inference compute, compare it with a non-target way of spending a similar budget.

Examples:

- pause-token computation;
- explicit scratchpad tokens;
- a deeper unrolled computation graph;
- repeated ordinary forward passes whose intermediate state is not recurrently reinjected.

The purpose is to distinguish "more compute" from "the target organization of compute".

## Control C3 — Loop-disabled or mechanism-disabled twin

Keep the transformed code path and parameterization but disable the functional mechanism.

Examples:

- recurrent block with loop count fixed to one;
- dual-stream architecture with cross-stream update disabled;
- workspace interface present but broadcast blocked;
- persistent-state allocation present but writes disabled.

This is stronger than comparing unrelated model families.

## Control C4 — Sham intervention

For tests based on an intervention, execute a sham operation with comparable interface and runtime overhead but no target state change.

A lesion or erasure effect should be interpreted relative to both the intact and sham conditions.

## Control C5 — General capability and scale panel

Across several systems, collect:

- parameter count or another declared scale measure;
- external benchmark scores chosen before FCA analysis;
- architecture family;
- training or instruction-tuning condition where known.

Run:

```bash
python -m fca_bench.analyze_specificity \
  --results result_a.json result_b.json result_c.json \
  --metadata model_metadata.json \
  --output specificity_analysis.json
```

The script reports Pearson and Spearman associations of every FCA dimension with log parameter count and with a standardized external-capability composite.

A strong correlation is a warning that a dimension may partly track generic capability. It is not conclusive because a genuine access capability may also scale with competence.

## Control C6 — Fresh sealed item bank

Generate private seeds after the evaluated model is frozen:

```bash
python -m fca_bench.sealing create \
  --suite all \
  --n 20 \
  --private sealed_seed_bank.private.json \
  --public sealed_seed_bank.public.json
```

Publish or timestamp the public commitment before evaluation. Keep the private file hidden until results are frozen.

Run from the private bank:

```bash
python -m fca_bench.run \
  --config config.qwen3_8b.example.json \
  --suite all \
  --seed-bank sealed_seed_bank.private.json \
  --output qwen3_8b_sealed.json
```

Verify after disclosure:

```bash
python -m fca_bench.sealing verify \
  --private sealed_seed_bank.private.json \
  --public sealed_seed_bank.public.json
```

Fresh seeds protect against exact-instance contamination, not generator-template contamination. Publication runs should therefore also reserve private template variants.

## Recommended minimum publication matrix

At minimum, include:

- standard base model;
- same base model plus target architecture;
- target architecture with the mechanism disabled;
- compute-matched non-target control;
- at least one model of a different size for scale diagnostics.

The result should be a pattern of dissociations, not a leaderboard ordering.
