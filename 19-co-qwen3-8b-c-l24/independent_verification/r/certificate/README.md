# Qwen3-8B R certificate — minimal English packet

Open the `*_REVISED.md` file for the readable certificate or the `*_REVISED.json` file for machine-readable status.

Status: `NONZERO_CERTIFIED`, for the real bf16 Qwen3-8B under the finite q-typed C_L24-R / u26 / L24.mlp0 contract.
Lower bound: R_state >= 7.385298596317709e-08 > 0.

This revision removes the misleading claim that v45.1 requires exhaustive, full-vocabulary predictive bisimulation before a finite-contract positive certification. Section 8.1 remains a requirement for matching interventionally relevant role signatures **under the declared contract**; S28 provides the declared 11-reader local test.

`EVIDENCE/` holds the four small JSON reports. Large raw NPZ arrays are referenced by SHA-256 in the certificate JSON but intentionally excluded.
