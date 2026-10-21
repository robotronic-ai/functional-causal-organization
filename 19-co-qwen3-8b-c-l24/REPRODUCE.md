# Reproduce

## Setup

- **Folder and interpreter.** Run everything from the working folder `quantum/` with
  `/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u` (written `PY` below).
- **Copying the scripts.** Copy the files of `scripts/` and `independent_verification/*/` into `quantum/`.
- **Pinned inputs.** The pinned inputs listed in [DEPENDENCIES.md](DEPENDENCIES.md) must be present.
- **Outputs.** Outputs are written under `quantum/FAST_GPU_RESULTS/`. Every script refuses to overwrite an existing
  output folder. Long scripts accept `--resume`.
- **Timings.** "measured" means the measured duration on an RTX 5060 Ti 16 GB; "estimated" durations have not been
  measured.

Every script has a CPU selftest: `PY <script> --selftest`.

## 1. Gauge cube (GPU, measured 1912.6 s for u26)

```
PY acquire_gauge_cube_gpu.py --preflight
PY acquire_gauge_cube_gpu.py --gpu --world 26
PY VERIFY_NPZ_SHA.py
```

## 2. Γ

```
PY gamma_frozen_rule_cpu.py --world 26
PY T322_VERIFY_U26_GAMMA_CERT_CPU.py --report <bundle>/independent_verification/gamma/T322_RECEIVED_T321_U26.json --require-raw
```

The verifier rechecks the 10 230 gap witnesses against the raw gauge cube.

## 3. Δ

```
PY T325_DELTA_FULL11_GAME_CPU.py
PY T325_DELTA_GAME_VERIFY_CPU.py
PY T327_DELTA_FINITE_NONZERO_PROOF_CPU.py
PY delta_joint_closure_cpu.py --world 26
```

## 4. R, greedy transition (GPU, measured 928.5 s)

```
PY r_greedy_return_gpu.py --preflight
PY r_greedy_return_gpu.py --gpu
PY r_token_channel_cpu.py --world 26
```

Optional structural check of the lemma on a generic prompt (GPU, a few minutes, estimated):
`PY T325_R_L24_TOKEN_CLAMP_GPU.py`.

## 5. R, sampled transition (GPU, measured 6005 s with the 10 replicas)

```
PY r_sampled_return_gpu.py --preflight
PY r_sampled_return_gpu.py --gpu
PY run_r_chain_cpu.py
```

The last command runs `export_t330_schema_cpu.py`, the independent verifier `T330_S27_DETERMINISTIC_R_BOUNDS_CPU.py`
(selftest, then on the export) and `r_return_bound_cpu.py`. It needs no GPU and takes about a minute.

## 6. Role identity t → t+1 (GPU, measured 2062 s)

```
PY role_identity_t1_gpu.py --preflight
PY role_identity_t1_gpu.py --gpu --world 26
PY role_identity_t1_cpu.py --world 26
PY role_signature_t1_cpu.py --world 26
```

## 7. Λ₀ (CPU, seconds, estimated)

```
PY lambda_relation_cpu.py --world 26
PY extract_lambda_witnesses_cpu.py --world 26              # witnesses above delta (primary)
PY extract_lambda_witnesses_cpu.py --world 26 --factor 2   # witnesses above 2 delta (secondary)
PY lambda0_l23_replay_cpu.py
```

The first two need the gauge cubes at t and t+1 (steps 1 and 6). The replay needs the L1 checkpoint folders
`NUIT_T127/T135/ck_L1_L23/` and `ck_L1_L24/` and the recorded outputs `NUIT_T127/T135/V_T135_L1_L2{3,4}.json` under
`quantum/` (or pass `--root` with the folder that contains `NUIT_T127`), and torch to read the .pt files.

## 8. Optional replication on u25

```
PY acquire_gauge_cube_gpu.py --gpu --world 25
PY gamma_frozen_rule_cpu.py --world 25
```

## Exploratory pilot T346 (optional)

Scripts, episodes and results are in `exploratory_t346/` (see its README for the exact commands, authorization flags and
checksums). Copy `exploratory_t346/scripts/*.py` into `quantum/`, then run each script with `--selftest` first.
The PCI_T branch needs one GPU run per dtype (about 9.5 minutes in float16, measured); the FCA-E1 branch durations are not measured.
