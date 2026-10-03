# FCA-Bench

FCA-Bench is the working code name for the **Functional Causal Access Benchmark**. It evaluates causal accessibility and governance of functional state in language-model systems and transformed architectures. It is independent from A_DAGGER and it is not a consciousness, general-intelligence, or sophistication score.

Protocol v0.2.7 defines two separate outputs:

- **Behavioral access profile**: `T, B, E, M, C, A, P`
- **Causal/interventional access profile**: `R, D, L, U, I, G, H`

There is **no official aggregate FCA scalar**. Behavioral evidence and causal/interventional evidence are never merged into a single score.

## Qwen3-8B baseline (GPU, 8-bit weights)

```bash
python run_qwen3_8b_gpu.py \
  --config config.qwen3_8b_gpu8.json \
  --n 20 --seed 0 \
  --output results/qwen3_8b_gpu8_results.json
```

Set `model_path` in `config.qwen3_8b_gpu8.json` to a local Qwen3-8B directory. The runner accepts the twelve-test baseline set (`--tests` selects a subset) and requires a CUDA GPU. To recompute the reported aggregates from the per-instance records:

```bash
python verify_results.py ../results/FCA027_GPU_R6_12TESTS_n20_FINAL.json
```

## Local Qwen3-8B quick start

The supplied Qwen configuration assumes this layout:

```text
<working directory>/
  Qwen3-8B/
  fca_bench/
  config.qwen3_8b.example.json
```

Therefore `model_path` is set to `./Qwen3-8B` and `local_files_only` is enabled. Relative model paths are resolved against the process working directory before Hugging Face is called. Missing local paths fail early with an explicit path error instead of being interpreted as Hub repository identifiers.

1. Start the shell in the working directory that contains `Qwen3-8B/` and `fca_bench/`.
2. Keep `system_id` stable across repeated runs.
3. Make sure the Python environment contains PyTorch and Transformers compatible with the local model.
4. Run the lightweight preflight check before loading model weights:

```bash
python -m fca_bench.preflight --config config.qwen3_8b.example.json
```

5. List tests:

```bash
python -m fca_bench.run --list-tests
```

6. Run a behavioral smoke test:

```bash
python -m fca_bench.run \
  --config config.qwen3_8b.example.json \
  --suite behavioral \
  --n 2 \
  --output results_behavioral.json
```

Reference baseline command (the test set reported in `../results/`):

```bash
python -m fca_bench.run \
  --config config.qwen3_8b.example.json \
  --tests T1,T2,T3,B1,E1,C3,R1,R3,D1,L2,U3,I3 \
  --n 2 \
  --output baseline.json
```

7. Run the causal/interventional track on the same standard Transformer baseline:

```bash
python -m fca_bench.run \
  --config config.qwen3_8b.example.json \
  --suite causal \
  --n 2 \
  --output results_causal.json
```

A standard Hugging Face Qwen3-8B adapter exposes only the `chat` capability. Causal/interventional tests requiring absent primitives return `status="structural_absent"` and score `0.0`. This is intentional and distinct from `status="invalid"`, which denotes a harness, parsing, or manipulation-check failure.

## Recommended publication workflow

Generate fresh sealed seeds after the model and benchmark version are frozen:

```bash
python -m fca_bench.sealing create \
  --suite all \
  --n 20 \
  --private sealed_seed_bank.private.json \
  --public sealed_seed_bank.public.json
```

Freeze or publish the public commitment before the run. Then execute:

```bash
python -m fca_bench.run \
  --config config.qwen3_8b.example.json \
  --suite all \
  --seed-bank sealed_seed_bank.private.json \
  --output qwen3_8b_sealed.json
```

After results are frozen, reveal and verify the bank:

```bash
python -m fca_bench.sealing verify \
  --private sealed_seed_bank.private.json \
  --public sealed_seed_bank.public.json
```

See `SEALING_PROTOCOL.md` for limitations and stronger contamination controls.

## Specificity controls

FCA-Bench should not be interpreted without controls for generic model competence, model scale, additional inference compute, and instruction-following.

The recommended design includes:

- standard Transformer baseline;
- same base model with the target mechanism enabled;
- mechanism-disabled twin;
- compute-matched non-target control;
- sham interventions;
- a multi-model scale/general-capability panel.

See `SPECIFICITY_CONTROLS.md`.

To analyze correlations with model scale and external benchmark performance:

```bash
python -m fca_bench.analyze_specificity \
  --results result_a.json result_b.json result_c.json \
  --metadata model_metadata.json \
  --output specificity_analysis.json
```

The analysis is diagnostic. High correlation with model size or general capability is a confound warning, not a proof that a dimension is invalid. Matched same-base-model interventions are stronger evidence of specificity.

## Qwen3 thinking mode

The example configuration sets `enable_thinking=false`. This is recommended for a clean standard-Transformer baseline because visible reasoning tokens add sequential computation through extra autoregressive steps. If thinking mode is studied, report it as a separate evaluated system.

## Custom architectures

The causal/interventional track does not infer hidden mechanisms from text. A modified architecture must expose benchmark operations through a custom adapter. Start from `custom_adapter_template.py` and read `ARCHITECTURE_HOOKS.md`.

## Naming

The current package keeps `FCA-Bench` as a working identifier for compatibility. The protocol uses **Functional Causal Access Benchmark**. The term `causal` refers to interventionally demonstrated access and state governance; it does not denote general intelligence, model sophistication, or consciousness.

Before public release, review `NAMING.md`. The name `FCA Bench` is already used by an unrelated public benchmark, so a distinctive publication name is advisable.


## Runtime progress

Local Hugging Face runs print a generation-level progress counter by default. The first line reports the estimated total number of language-model generation calls. Subsequent `[GEN current/total]` lines report per-call duration and an approximate ETA. Use `--quiet` to suppress these messages.

E1 is intentionally expensive because it uses matched clean/placebo/poison/erase arms with repeated stochastic probes. For a quick environment smoke test, run without E1 first.


## Result-status interpretation

`executed` means the model or architecture actually performed the test and received a measured score. `structural_absent` means the required architecture primitive is unavailable; its numeric zero is retained for machine aggregation of causal availability but must not be described as a failed task attempt. `invalid` means the instance cannot be interpreted because execution, parsing, or a required manipulation check failed.

For E1, keep `e1_samples=5` only for smoke/debug work. Use `e1_samples>=20` for publication-oriented runs.
