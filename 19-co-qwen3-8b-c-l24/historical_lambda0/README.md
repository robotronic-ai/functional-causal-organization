# Primary evidence of the restricted Λ₀ verdict (L23/L24 anchor roles)

- `T135/V_T135_L1_L23.json`, `T135/V_T135_L1_L24.json`: outputs of the L1 runs (ROLELAMBDA_v1, bf16, 10 exact-recoding
  replicas, calibration cases u15|SUBSTITUTE|H1 (reference), u16|REVOKE|H2, u17|SUBSTITUTE|H2, u18|REVOKE|H1).
- `T135/ck_L1_L23/meta.json`, `T135/ck_L1_L24/meta.json`: signatures of the per-replica raw write files (the .pt files
  themselves are not included; their SHA-256 are in `VERIFY_NPZ_SHA.py`).
- `T135/T135_Q2_VALIDATION.json`: validation of the companion Q2 distinction test (heads[24-27] vs mlp3 at L23).
- `TURN133/`: frozen L1 anchor contract and anchor list.
- `TURN137/`: protocol and note recording the argument (every admissible completion contains the non-empty lower relation).
- `instrument/role_lambda.py`: the producing instrument (SHA-256 ed57c4b45e9978ded1f8805634109ccd9791a3df72c4985a30e273cb9d1e77f9),
  for reading. It imports pinned modules of `CLAUDE_CO_Q2L1/`. The replay `scripts/lambda0_l23_replay_cpu.py` does not
  import it.

The files in `T135/`, `TURN133/` and `TURN137/` are included byte-identical to the originals.
