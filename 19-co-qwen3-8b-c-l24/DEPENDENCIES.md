# Dependencies

## Software

Python 3.12 environment with `torch` (CUDA), `transformers`, `numpy` and `scipy`. Qwen3-8B weights are in
`quantum/Qwen3-8B`, loaded in bf16.

## Pinned inputs on disk (relative to `quantum/`)

The GPU scripts load the experimental preparation from pinned files and refuse to run if a digest differs.

| Input | Role | Check |
|---|---|---|
| `T292_SAME_CODE_MICRO_DONOR_FIBERS_GPU.py` | preflight entry point. It loads the chain of pinned modules that define the sites, frozen donors, frozen background pairs, frozen L25 readers and historical arms | SHA-256 `64955123f179ff1fc9d798be0d7954a689ce1fadc933c372e14e4a4839423bc3`, then the recursive pins of the chain |
| `CLAUDE_CO_Q2L1/` | shared modules: `role_quotient`, `gamma_local_qwen`, `qwen_local_transition` | verified by the preflight chain |
| `TURN203/REFERENCE_ENGINE_COMPLETE/` | reference prompt engine and runtime | verified by the preflight chain |
| `direct_co_op_scripts_v42/DIRECT_CO_OP_DEVELOPMENT_SCHEDULE_34.json` | development schedule (worlds, bindings) | verified by the preflight chain |
| `FAST_GPU_RESULTS/CLAUDE_S5_GAUGE_FIBER_COMMUTATION_GPU/` | reference gauge sub-cubes | optional additional guard of the gauge cube acquisition |

## Reference copies included in `dependencies/`

These files are provided for reading. The scripts import the pinned originals from `quantum/CLAUDE_CO_Q2L1/`.

| File | SHA-256 |
|---|---|
| `role_quotient.py` (version `ROLEQ_v3.2_T126_CONFORMANT`) | `ca337f33e42c62fd77a925a978db7dd09b490d77944029e95b5219c6669d9195` |
| `gamma_local_qwen.py` | `789e95627f13a846f4739680310a5e7fa79a2d594418c3780deb00ce965bdbdc` |
| `qwen_local_transition.py` | `b7829567f9952aab9c7289b08c038a1d9abcf51d6b74e62146403afd696b42e4` |

The L1 instrument of the restricted Λ₀ runs, `role_lambda.py`, is included for reading in `historical_lambda0/instrument/`.
