# Qwen3-8B C_O — T346 exploratory calibration pilot (mlp0@L24 activation attenuation)

Status: **EXPLORATORY_CALIBRATION_PILOT, world u26, descriptive only.** No inferential significance, no monotone-law claim, no
change of any certified status. Γ, Δ, R, Λ₀ remain NONZERO_CERTIFIED on the C_L24 contracts; functional closure remains open.

## Classification
The attenuation of mlp0@L24 is an unqualified local-strength / internal-state perturbation: its effect on Γ, Δ, R, Λ was not measured, so it is not a C_O bridge test. Within the planned specificity panel (SHAM, CO_TARGET, STRENGTH_MATCHED, GENERIC_STATE_MATCHED, GAUGE_CONTROL, optional READOUT_CONTROL) it is a candidate for the strength / local-write rival family, not for CO_TARGET. The panel is a proposal of the project's scientific arbiter and has not been executed; CO_TARGET interventions are to be selected after the float16 transport of the C_L24 measurements (T347), with PCI_T and FCA-Bench still closed for target selection.

## What is tested
One frozen model (Qwen3-8B), five arms: `U26_BASELINE` (no hook), `U26_SHAM` (same hook code path, dose 0), `U26_ABLATE_025`,
`U26_ABLATE_050`, `U26_ABLATE_100`. The manipulation multiplies the input columns [0, 3072) of `layers[24].mlp.down_proj`
(the mlp0@L24 role) by (1 - dose). Targets: P0 `PCI_T_DS_L27` and `FCA_E1`. SHAM must be bit-identical to BASELINE.
Descriptive bands (not tests): FCA_E1 ±0.05, PCI_T ±0.10. A Δ-reduction under the ablation is **not measured** by these scripts
(`delta_reduction_valid = NOT_MEASURED_IN_THIS_SCRIPT`), so a null target response is not a refutation of C_O.

Seed streams: tasks 2026101801, donors 2026101802, PCI directions 2026101803, bootstrap 2026101804.

## Contents
| Path | Role | SHA-256 |
|---|---|---|
| `scripts/t346_pcit_l27_pilot.py` | PCI_T L27 branch (selftest CPU, run GPU, analyze) | `49c6a7f33e27b5d746b8b3c7c4955afe9d1e2c82b5f6114fd82a050865459901` |
| `scripts/t346_fca_e1_pilot.py` | FCA-Bench v0.2.7 E1 branch (selftest CPU, run GPU, analyze) | `c0eb6944ab6801ca9a0fce37c9ede08dbb664c783cc983d741957261020ae893` |
| `scripts/t346_restore_fca_v027.py` | extracts the FCA-Bench v0.2.7 package from its zip into `quantum/FCA027/` and checks it against its MANIFEST.json (CPU) | see MANIFEST |
| `scripts/t346_night_chain.py` | overnight chain on one GPU: T347 float16 gauge stage, T347 core CPU stage, then the FCA-E1 pilot; per-step logs and measured durations in CHAIN_SUMMARY.json | see MANIFEST |
| `scripts/t359_delta_f16_replay_11ctx_cpu.py` | CPU replay of the float16 Delta game from the raw u26 gauge cube: variant A (12 active roles) and variant B (11 active roles + context `heads[24,25,26,27]`), P* class counts reported; refuses a cube whose SHA-256 differs from RESULT.json; the raw cube is not in this bundle | see MANIFEST |
| `scripts/t347_deploy_turn053.py` | installs the TURN_053 T347 suite from the handoff archive into the working folder (old files kept as .bak_turn052; manifest-checked) | see MANIFEST |
| `scripts/t346_fca_e1_calibration.py` | targeted calibration of FCA_E1 at mlp0@L24: ZERO, MEAN, matched conditional substitution (PATCH), strength-matched random perturbation, with strength measured on neutral texts and E1 prompts (activations only; weights untouched) | see MANIFEST |
| `scripts/l24_mlp0_ablation_hook.py` | standalone attenuation hook with selftest | `dc3aa9b434d8715e15dfcfb256c3fec39ad553f294d84ee780e5c94097605291` |
| `episodes/T346_PCIT_PILOT_EPISODES.json` | six pilot episodes T346P_001..006 (also embedded in the PCI_T script) | see MANIFEST |
| `results/pcit_l27_float16/` | PCI_T branch, float16 (historical dtype): RESULT.json, records.jsonl, pilot_table.csv | see MANIFEST |
| `results/pcit_l27_bfloat16_diagnostic/` | same run in bfloat16: diagnostic only, the instrument saturates | see MANIFEST |
| `MANIFEST_SHA256.txt` | SHA-256 of every file | |

No NPZ file is included.

## Commands (from `quantum/`, `PY` = `/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u`)
```
PY t346_pcit_l27_pilot.py --selftest
PY t346_pcit_l27_pilot.py --run --authorize-gpu-run I_AUTHORIZE_T346_PCIT_L27_PILOT_GPU_RUN --dtype float16 --out-suffix _F16

PY t346_restore_fca_v027.py --zip <path to 17-fca-bench.zip>
PY t346_fca_e1_pilot.py --selftest
PY t346_fca_e1_pilot.py --run --probe --authorize-gpu-run I_AUTHORIZE_T346_FCA_E1_PILOT_GPU_RUN
PY t346_fca_e1_pilot.py --run --authorize-gpu-run I_AUTHORIZE_T346_FCA_E1_PILOT_GPU_RUN --resume
PY t346_fca_e1_pilot.py --analyze
```
The PCI_T run writes its own RESULT.json and pilot_table.csv; `--analyze` re-reads the default output folder only. The FCA runner needs the FCA-Bench v0.2.7 package (`__version__ 0.2.7`, `__runner_id__ FCA027_GPU_R6`), restored into `quantum/FCA027/` by `t346_restore_fca_v027.py`. It looks for a folder containing `fca_bench/__init__.py` under the working folder and its parent; use
`--fca-dir <folder>` if the search fails. If `config.qwen3_8b_gpu8.json` is absent it uses the embedded historical R6 configuration
(bitsandbytes 8-bit, float16 compute, sdpa, 20 draws, micro-batch 4, 128 new tokens).

## PCI_T branch result (float16, 6 episodes, 2 fractions each, L27, 16 directions)
| Arm | dose | DS_L27 (median over episodes) | median paired difference vs SHAM | median source rel. change at L27 |
|---|---|---|---|---|
| BASELINE | 0 | 1.4650 | 0 | 0 |
| SHAM | 0 | 1.4650 | 0 | 0 |
| ABLATE_025 | 0.25 | 1.4423 | -0.0017 | 0.029 |
| ABLATE_050 | 0.50 | 1.4219 | -0.0231 | 0.058 |
| ABLATE_100 | 1.00 | 1.4085 | +0.0159 | 0.118 |

- SHAM is identical to BASELINE on all 12 sites (maximum absolute score difference 0).
- The manipulation reaches L27 (source relative change 2.9 / 5.8 / 11.8 %, proportional to the dose).
- No median paired difference exceeds the descriptive band ±0.10; the sign of the difference is not monotone in the dose; the
  descriptive 95 % intervals of the differences all contain 0. Six episodes of one world do not support any inference.
- Instrument check: baseline DS 1.465 in float16 is comparable with the historical float16 calibration (median 1.32 on its
  own episodes); in bfloat16 the baseline DS is 1.049 and the scores saturate, so the bfloat16 run is reported as a diagnostic only.
  Each branch should use the dtype of its historical protocol.
- Measured durations: about 9 s per site in float16 (about 110 s per arm, about 9.5 min for five arms); about 10.7 s per site in bfloat16.
  FCA-E1 durations are not measured yet.

## FCA_E1 branch: timing probe (1 instance x BASELINE and SHAM)
Measured on the RTX 5060 Ti (bnb8, float16 compute, sdpa, 20 draws, micro-batch 4, 128 new tokens; deviations from the historical R6 configuration: none):
model loading 92 s; 144.43 s (BASELINE) and 144.89 s (SHAM) per instance and arm, 61 generate calls each (about 2.4 s per call, 4016 generated
tokens, about 28 tokens/s); acquisition loop 289.3 s. SHAM is bit-identical to BASELINE (all 241 raw outputs and the full result). The E1 validity
checks pass on this instance (distributional effect 0.473, poison direct leak 1.0), score 0.5475. One instance, no inference.
Estimated duration of the full run (20 instances x 5 arms = 100 instance-arms at about 144.5 s): about 14,450 s, i.e. about 4.0 h plus loading;
this is an extrapolation from the probe, not a measurement for the ablation arms. Files: `results/fca_e1_probe/`.

## FCA_E1 branch: full run (20 instances x 5 arms, bnb8 + float16 compute, historical R6 configuration, no deviation)
Files: `results/fca_e1_full_float16_bnb8/` (RESULT.json, RUN_HEADER.json, pilot_table.csv; the 5 x 20 raw records.jsonl stay with the operator). Script SHA-256 `c0eb6944ab6801ca9a0fce37c9ede08dbb664c783cc983d741957261020ae893`.
All 100 instance-arms are valid. SHAM is bit-identical to BASELINE on all 20 instances. Measured duration: 14,315.9 s of instance time in total (about 3.98 h, about 2,845 s per arm), model loading not included.

| Arm | dose | FCA_E1 (mean over 20) | descriptive 95 % interval | mean paired difference vs SHAM | descriptive 95 % interval of the difference | median removed activation RMS |
|---|---|---|---|---|---|---|
| BASELINE | 0 | 0.6875 | [0.631, 0.741] | 0 | | 0 |
| SHAM | 0 | 0.6875 | [0.631, 0.741] | | | 0 |
| ABLATE_025 | 0.25 | 0.6857 | [0.633, 0.734] | -0.0018 | [-0.0149, +0.0108] | 0.085 |
| ABLATE_050 | 0.50 | 0.6772 | [0.624, 0.724] | -0.0103 | [-0.0257, +0.0037] | 0.170 |
| ABLATE_100 | 1.00 | 0.6888 | [0.628, 0.745] | +0.0013 | [-0.0251, +0.0287] | 0.340 |

- No mean difference exceeds the descriptive band of 0.05 (FCA_E1), the differences are not monotone in the dose (-0.002, -0.010, +0.001), and the intervals contain 0.
- Per-instance differences are heterogeneous (standard deviations 0.030, 0.034, 0.063 at doses 0.25, 0.5, 1.0; 2, 3 and 6 instances beyond +/-0.05; opposite signs across instances), which is consistent with noise or instance-specific effects of opposite sign and does not point to a systematic shift.
- Instrument note: s_direct equals 1.0 in every arm (ceiling), so the E1 score varies only through s_distributional (half of the score range); 4 of 20 instances sit at exactly 0.5 in all or almost all arms.
- All validity checks pass in every arm (poison direct leak 0.98-0.99, distributional effect 0.83-0.85).
- Scope: one world, 20 instances, one manipulation (mlp0@L24 input columns 0-3071 scaled by 1 - dose), `delta_reduction_valid` not measured. This is a descriptive null for a local-strength perturbation, not a test of C_O. The runner uses FCA-Bench v0.2.7 unmodified (HFLocalAdapter, E1 scoring), 20 instances, five arms,
61 generate calls per instance and arm (60 batched E1 calls plus one diagnostic). The `--probe` mode measures the per-call duration first.

## Overnight chain
`PY t346_night_chain.py --selftest`, then `--dry-run`, then
`PY t346_night_chain.py --run --authorize-t347 I_AUTHORIZE_T347_CL24_FLOAT16_GPU_RUN --authorize-fca I_AUTHORIZE_T346_FCA_E1_PILOT_GPU_RUN`.
The steps run one after the other (a single GPU): `t347_gauge` (GPU), `t347_core` (CPU, only if the gauge step succeeded), `fca_e1` (GPU). A failed or
missing step is recorded and the chain continues with the independent steps. The T347 suite itself is not part of this bundle. PCI_T is not read.

### Overnight chain, first execution (2026-10-19 20:47 to 2026-10-20 00:52, `results/night_chain_20261019/`)
- `fca_e1`: OK, 14,506.3 s including model loading (acquisition loop 14,318.1 s). It is the run reported in `results/fca_e1_full_float16_bnb8/`.
- `t347_gauge`: FAILED after 218.1 s with `RuntimeError: D0 background pairs differ from the frozen pairs` raised by `t347_f16_acquire_gauge_cube_gpu.py`; the log shows the launcher went straight to the gauge script without the frozen bf16 background capture, and the launcher SHA-256 on disk (`4164c406...`) differs from the TURN_053 launcher (`25443391...`). The working folder held the TURN_052 suite, not the TURN_053 hotfix. No gauge-cube result was produced.
- `t347_core`: SKIPPED (dependency `t347_gauge` not OK), as designed.
- Remedy: `PY t347_deploy_turn053.py --zip <TURN_053 handoff archive>` installs the TURN_053 suite (backups of replaced files as `.bak_turn052`), then run `test_t347_turn053.py` and re-run the chain with `--steps t347_gauge,t347_core`.

### Overnight chain, second execution (2026-10-20 11:37 start, `results/t347_float16_cl24_u26/`)
After `t347_deploy_turn053.py` (TURN_053 suite installed, 6 of 6 checks passed) the chain was re-run with the two T347 steps only (`t347_gauge`, `t347_core`):
- `t347_gauge`: OK, 2184.6 s for the whole step (measured), of which the GPU acquisition of the float16 gauge cube was 1886.9 s (4096 arms x 10 gauges, world u26; the frozen bf16 background capture precedes it).
- `t347_core`: OK, 1120.5 s (measured, operator terminal output). Its final log and the final `CHAIN_SUMMARY.json` were not provided; the files kept here are snapshots taken while the core stage was running, plus the verbatim terminal output.

## T347: float16 transport of the C_L24 measurements (world u26) - operator results, exploratory, no formal status
Contract: C_L24 frozen rule, bf16 D0 background donors frozen, float16 evaluation. Nothing was re-tuned; the declared status of every stage is `AWAITING_INDEPENDENT_CERTIFICATE`, and `formal_Qwen_Gamma_certified` is `false`.

| Quantity (u26, 10 gauges) | bf16 (certified C_L24 bundle) | float16 transport (T347) |
|---|---|---|
| gauge deltas of the 12 readers | 0.116 - 0.76 | 0.014 - 0.056 |
| retained sites (frozen rule) | 11, plus `heads[24..27]` as context | 12, no context site |
| Gamma, delta_abs, minimum over gauges | 0.289 | 0.5 on every gauge (critical cut `heads[0,1,2,3]`) |
| Gamma, delta_contrast, minimum over gauges | 0.00195 | 0.5 on every gauge |
| cuts | 1023 | 2047 (1614 fully distinguishable in both directions) |
| Delta_cap (delta_abs) | 0.968 - 0.970 | 1.000 (v(N) = 12 bit for 12 roles) |
| Pi_H / Pi_S (delta_abs) | 0.766 - 0.771 / 0.670 - 0.678 | 0.9530 - 0.9547 / 0.9161 - 0.9190 |
| Delta_H / Delta_S minimum (delta_abs) | 0.742 / 0.650 | 0.9530 / 0.9161 |
| Delta_H / Delta_S minimum (delta_contrast) | 0.708 / 0.638 | 0.9392 / 0.8963 |

Reading and limits:
- The two regimes use different numerical tolerances (the float16 gauge deltas are one to two orders of magnitude smaller), so the columns are not a like-for-like comparison and do not rank the regimes.
- Gamma equals 0.5 on all 10 gauges and both deltas, with the next cuts at 0.5028, 0.5056, 0.5097, 0.513: it is the floor of the cut table at this resolution, not a measure of gauge sensitivity.
- Delta_cap saturates at 1.0 (all 4096 donor configurations are separated at the float16 resolution), so Delta_H and Delta_S reduce to the Shapley participation Pi of the 12 roles (shares 0.032 - 0.117 in delta_contrast, 0.040 - 0.117 in delta_abs), which are not uniform.
- The gauge sub-cube that is nominally the bf16 reference sub-cube is not bit-identical in float16 (max absolute difference 0.352 reported in the log); this is a declared diagnostic only.
- D0 `heads[16..19]` fresh units [16,19] versus frozen [15,19] were reported by the log as a diagnostic.
- The role-identity, r-greedy and r-sampled stages of the suite were not run.
- NPZ tables are not archived; their SHA-256 are in `results/t347_float16_cl24_u26/NPZ_SHA256.txt`.
- An independent arithmetic audit of the 12-role game tables was run by the project's arbiter (T358, CPU): Shapley values, participation and monotonicity were recomputed from the 4096 coalition values and reproduce the figures above (min Pi_H 0.952972 / 0.939212, min Pi_S 0.916129 / 0.896301). Delta_cap, the operational P* and the 11-role + context replay were not recomputed because the raw cube was not supplied; `t359_delta_f16_replay_11ctx_cpu.py` is the CPU replay prepared for that purpose (selftest and a synthetic end-to-end run only; not run on the real cube). Status unchanged: `AWAITING_INDEPENDENT_CERTIFICATE`.
- No effect on any status: Gamma, Delta, R, Lambda0 on C_L24 (bf16) are unchanged and functional closure remains open. The float16 results do not by themselves select a CO_TARGET for the calibration panel.

### Calibration, first launch in bfloat16 without quantization (operator terminal output, 2026-10-20 afternoon)
Measured by the script: reference means and 13 E1 prompts 131.91 s; amplitudes of 7 arms 407.57 s; first instance-arm (CAL_BASELINE, no hook) 1148.46 s for 61 generate calls, against about 85 s per instance-arm in the T353 bf16 run. The device reported 15.26 GiB allocated and 15.27 GiB reserved of 15.93 GiB, so the bf16 model leaves under 1 GiB of headroom. Working hypothesis (not verified): the amplitude stage, which held vocabulary-sized float32 tensors, pushed the device into spill to shared host memory. Change made afterwards: the amplitude KL is computed in chunks of 64 positions, CUDA caches are released after the reference and amplitude stages and after every instance, and allocated / reserved / free memory is logged. The slow first record is kept as acquired.

Second launch of the same calibration with the corrected script (SHA-256 `e80f4944...`, new output folder suffix `_BF16_V2`; the first folder is kept as an abandoned launch, and `--resume` was refused because the script SHA-256 is part of the provenance): reference means and 13 E1 prompts 133.07 s; amplitudes of 7 arms 104.91 s (407.57 s before); the log line after these stages reports allocated 15.27 GiB, reserved 15.29 GiB, free 0.00 of 15.93 GiB. First records: CAL_BASELINE 147.98 s, CAL_SHAM 154.72 s, MLP0_L24_ZERO 128.95 s per instance-arm (1148.46 s before; about 85 s in T353). BASELINE and SHAM scores on E1-2026101801 are identical (0.7237). The gap to the T353 figure is not explained: the device has no free memory left, which is compatible with residual memory pressure but was not measured as such. Total time for 42 instance-arms is not measured; at about 144 s per arm it would be about 1 h 41 min.

## Cut-table structure of Gamma in float16 (u26), recomputed from `S22_CUT_TABLE_u26.npz` by the build side as a consistency check, not as an audit
- On all 10 gauges and both tolerances the minimum cut is the mask 1 (`heads[0,1,2,3]` alone) with score 0.5, whose directional terms are J(A->B) = 1 and J(B->A) = 0 on every gauge. A positive cut score therefore does not imply influence in both directions.
- Minimum cut score restricted to cuts with at least 2 roles on each side: 0.9250 (delta_abs), 0.7550 (delta_contrast), attained by mask 10 (`heads[4,5,6,7]` and `heads[20,21,22,23]` against the rest). At least 3 roles on each side: 0.9996 and 0.9878. Balanced 6/6 cuts: attained by mask 1071 (includes `mlp0`).
- Both directional terms strictly positive: 20,460 of 20,470 (gauge, cut) pairs under delta_abs and 20,445 of 20,470 under delta_contrast; the exceptions all have J(B->A) = 0.
- This characterizes the float16 transport tables only; it is not a Gamma certificate and does not change the bf16 certificate. These figures reproduce those reported by the project's arbiter (T359).

## Targeted FCA_E1 calibration, bfloat16 without quantization, 6 instances (`results/fca_e1_calibration_bf16_V2/`, operator run, descriptive only)
Script SHA-256 `e80f4944...`; instances E1-2026101801 to E1-2026101806 (the first six of the 20 used in T353, so not independent of it); 13 E1 reference prompts and mean activations from two disjoint instances; 42 instance-arms. Measured: reference 133.07 s, amplitudes 104.91 s, sum of instance-arm times 6016.8 s (about 100 min; 138.6 to 148.3 s median per arm; model loading about 85 s in the first launch). SHAM is identical to BASELINE on all 6 instances (0.7705 both).

| Arm | FCA_E1 | mean paired difference vs SHAM (descriptive 95 % interval) | positive / negative | relative change of the hidden state after layer 27 (E1 prompts) | mean KL, E1 prompts (nats) |
|---|---|---|---|---|---|
| BASELINE = SHAM | 0.7705 | 0 | - | 0 | 0 |
| MLP0_L24_ZERO (down_proj input channels 0..3071 set to 0) | 0.8028 | +0.0323 [-0.0127, +0.0759] | 4 / 2 | 0.0252 | 0.0390 |
| MLP0_L24_MEAN (same channels set to reference means) | 0.8041 | +0.0335 [-0.0112, +0.0768] | 4 / 2 | 0.0239 | 0.0372 |
| MLP_L24_MEAN (all 12,288 channels set to reference means) | 0.8336 | +0.0630 [+0.0139, +0.1279] | 5 / 1 | 0.0465 | 0.1364 |
| MLP0_L24_PATCH_MATCHED (code/category substituted, same token length) | 0.7629 | -0.0076 [-0.0247, +0.0075] | 2 / 3 (1 zero) | 0.0033 | 0.0010 |
| STRENGTH_MATCHED_RANDOM (fixed random direction, per-token norm of the ZERO change at the output) | 0.7456 | -0.0250 [-0.0588, +0.0166] | 1 / 5 | 0.0257 | 0.0379 |

Distributional score (baseline 0.541): ZERO 0.606, MEAN 0.608, MLP all MEAN 0.667, PATCH 0.526, RANDOM 0.491; residual effect (baseline 0.322): 0.260, 0.257, 0.225, 0.329, 0.349. s_direct, poison direct leak and erase direct leak do not move; all 42 instance-arms are valid.

Reading (descriptive, six instances, no inference):
- The ZERO arm at the full dose reproduces the paired differences of the T353 `U26_ABLATE_100` arm on the same six instances to every printed digit (+0.0207, -0.0328, +0.0548, +0.0865, -0.0395, +0.1040), with a different script and session. The reference-means file has the same SHA-256 (`c52821c6...`) as the one in the arbiter's T358 report, so the reference stage is also reproducible.
- Replacing the channels by reference means instead of zeros gives the same mean difference (+0.0335 versus +0.0323): at this strength the change does not depend on the substituted value being zero.
- The strength-matched random perturbation (layer-27 relative change 0.0257 versus 0.0252 for ZERO) does not reproduce the positive change; its mean difference is negative (5 of 6 instances). Intervals include 0 for ZERO, MEAN and RANDOM.
- The whole-layer MEAN arm has about twice the layer-27 change and a larger difference (+0.063, interval above 0), so the response is not specific to the first quarter of the channels.
- The PATCH arm is about 8 times weaker than ZERO at layer 27 (0.0033 versus 0.0252), was applied to 186 of 366 prompts (180 skipped because the prompt has no context sentence) and to 7 of the 13 reference prompts; it is not strength-matched and says little about content specificity.
- `hook_change_rms` is not comparable between arms: ZERO and MEAN report the removed activation per input element (about 0.34), RANDOM the per-token norm of the added output vector (about 33.5).
- Not measured: Gamma, Delta, R, Lambda0 on the modified states (`delta_reduction_valid` NOT_MEASURED); float16 without quantization (not run in this launch). The result therefore does not support a statement that the FCA_E1 change is mediated by organization.

## T364: arm-level acquisition manifest and structural audit (CPU, no model)
Context: the arbiter's T363 status keeps `CO_RIVAL_SUPERIORITY = NOT_TESTED` and graded closure `OPEN_NOT_ESTABLISHED`; the four out-of-fold contrasts D0 = R2(M1) - R2(M0) and D4 = R2(M1) - R2(MR), on FCA_E1 and on PCI_T_DS_L27, are not computable because no table has numerical Gamma, Delta, R, Lambda0, the M0 and influence-rival covariates and both targets on the same independent realizations.
- `scripts/t364_acquisition_manifest_audit_cpu.py` (standard library only) checks a typed manifest, one row per planned realization: family id and status (`PRECOMMITTED` with a commitment time earlier than any target measurement), regime and boundary, intervention, four C_O coordinates each with a contract id and a MEASURED / NOT_MEASURED status, four M0 covariates, four MR covariates, and the two targets. It returns, for each of the four contrasts, STRUCTURALLY_COMPUTABLE or NOT_COMPUTABLE with the reasons (usable rows, number of independent families against the 5 required for family-held-out folds, rows per training fold against an audit parameter, precommitment failures, mixed regimes). It fits no model and computes no R2; the closure cell rule is not evaluated (it belongs to the frozen T341 test).
- `scripts/t366_acquisition_budget_cpu.py` (standard library only, no model) converts measured stage durations (bf16 u26: gauge cube 1912.6 s, R greedy 928.5 s, R sampled 6005 s, role identity t to t+1 2062 s; FCA_E1 about 1128 s for 6 instances; PCI_T about 127 s for 6 episodes; one operator run each) into a GPU time estimate for a joint acquisition. With the T364 audit parameters (3 training rows per predictor, 8 predictors in the largest model, 5 families) it needs 34 rows; one full bf16 row is about 12,160 s (3.4 h), so about 115 GPU hours (12 nights of 10 h); with 2 rows per predictor, 23 rows and about 78 h. Without sampled R: about 58 h and 39 h. The four C_O stages were never run on an intervened state, so these durations are an assumption, and the row count is a lower bound on independent realizations: interventions on one set of weights are not shown to be independent systems. Output: `manifest_t364/T366_ACQUISITION_BUDGET.json`.
- `scripts/t367_intervened_c_o_gpu.py` (GPU, authorization flag `I_AUTHORIZE_T367_INTERVENED_C_O_GPU_RUN`) runs the certified bf16 C_L24 chain (gauge cube, Gamma, Delta, R greedy and sampled, role identity, Lambda0) on a model whose mlp0@L24 is attenuated by a dose: after the model is loaded, `layers[24].mlp.down_proj.weight[:, 0:3072] *= (1 - dose)` in memory (equal to the ALL_PROCESSED_TOKENS activation hook; dose 0 edits nothing, dose 1 is the zero ablation). Each row has its own folder `<rows-dir>/<row-id>/` made of symbolic links to the working folder and a private `FAST_GPU_RESULTS`; the base outputs are never touched. The frozen donors, readers (origins and directions), masks, world and seeds are unchanged, so the game is the same function on edited weights, scored relative to the unmodified model's readers: a stated design choice. At dose 0 all guards of the legacy scripts stay strict; at dose above 0 only the guards that compare with the unmodified model (frozen historical arms, identity sub-cube and reference gauge sub-cubes, clean active token) become recorded deviations (`ROW_DEVIATIONS.jsonl`); internal consistency guards stay strict. It does not measure FCA_E1, PCI_T_DS_L27 or the M0 and influence covariates. The legacy scripts recompute the frozen background donors (units of T286.D0) from the live model; to keep these donors those of the unmodified model, the launcher switches the edit off during that capture and on afterwards (it inserts two calls around `values_bg` and `T286.freeze_background_pair`, refuses a script where these lines are not found exactly once, and marks the stage invalid if the two phases are not executed exactly once each). A first attempt at dose 0.75 without this switch stopped at the strict guard 'D0 background pairs differ from T286' after a recorded deviation 'unexpected active token u15': with the edit on, the active token of that donor unit changed. Tested on CPU with a tiny random Qwen3 and a stand-in for the engine. Control at dose 0 on the real model (`results/t367_control_dose0_u26/`): the gauge cube stage passed with strict guards and its `raw_sha256` (8d421b29...da3cc8) and 12 gauge deltas are identical to the certified bf16 u26 cube; it took 2292.9 s against 1912.6 s for the certified run. No attenuated row has been run yet.
- `manifest_t364/T364_ACQUISITION_MANIFEST_DRAFT.json` lists the 12 arms measured so far (5 from the T353/T355 pilot, 7 from the bf16 calibration) with their FCA_E1 means, the pilot PCI_T medians and candidate intervention rms values; every C_O, M0 and MR coordinate is null by design, family ids are proposals (`PROPOSED_NOT_PRECOMMITTED`), and `MLP0_L24_ZERO` is flagged as identical to `U26_ABLATE_100` on the shared instances.
- `manifest_t364/T364_MANIFEST_AUDIT_DRAFT.json`: result on that draft: 12 rows, 4 controls, FCA_E1 present on 12 rows and PCI_T on 5, C_O / M0 / MR on none; all four contrasts NOT_COMPUTABLE (0 usable rows, 0 families); closure NOT_READY. This is a readiness statement, not a result about C_O.
- The thresholds `--min-families 5` (from the handoff) and `--rows-per-predictor 3` (an audit parameter, not a scientific criterion) are to be frozen by the arbiter.

## Numerical C_O coordinates available today (descriptive; operator and published results)
| Realization | active roles / context | Gamma, minimum over gauges (delta_abs / delta_contrast) | Delta_H / Delta_S, minimum over gauges (delta_abs) | Delta_H / Delta_S (delta_contrast) |
|---|---|---|---|---|
| u26, bf16 (certified contract) | 11 / `heads[24,25,26,27]` | 0.289 / 0.002 | 0.742 / 0.650 | 0.708 / 0.638 |
| u26, float16 transport | 12 / none | 0.5 / 0.5 | 0.953 / 0.916 | 0.939 / 0.896 |
| u25, bf16 (`results/delta_u25_bf16/`) | 10 / `heads[16,17,18,19]`, `heads[24,25,26,27]` | 0.0 (cut isolating `heads[8,9,10,11]` has zero integration; not positive on the frozen rule) | 0.644 / 0.521 (Delta_cap 0.923 to 0.951) | 0.664 / 0.600 |
- R and Lambda0 have numerical values only for u26 in bf16; there is none for u25 or for float16.
- The three rows have different active-role sets and tolerances, so they are not on one common scale. The table shows that the coordinates are not constant across realizations (Gamma: 0, 0.289, 0.5), and that Delta is positive in u25 although Gamma is not positive there. Both worlds use the same weights; this is not a set of independent physical systems.
- Delta u25: all 10 gauges positive under both tolerances; status `DELTA_POSITIVE_ON_C_L24__ALL_GAUGES__AWAITING_INDEPENDENT_CERTIFICATE`. The u25 individual (10 roles) is a different causal individual from the u26 11-role contract.

## Where the FCA_E1 residual sits (computed from the bf16 calibration records, `results/fca_e1_calibration_bf16_V2/records.jsonl`)
Permutation-bias-corrected Jensen-Shannon residual between the erase and placebo arms, mean over the six instances:
| Arm | codename family | category family | mean (residual_effect) |
|---|---|---|---|
| BASELINE | 0.551 | 0.093 | 0.322 |
| MLP0_L24_ZERO | 0.468 | 0.051 | 0.260 |
| MLP0_L24_MEAN | 0.463 | 0.051 | 0.257 |
| MLP_L24_MEAN | 0.416 | 0.035 | 0.225 |
| MLP0_L24_PATCH_MATCHED | 0.537 | 0.122 | 0.330 |
| STRENGTH_MATCHED_RANDOM | 0.586 | 0.113 | 0.349 |
The residual is dominated by the codename family and stays large after ZERO (0.468 of a maximum of 1); the relative reduction is about 15 % for the codename family and about 45 % for the category family. The score gain therefore comes from a partial reduction of a large residual, and the erasure is far from complete in every arm.

Same computation on the float16 unquantized run (`results/fca_e1_calibration_float16_V2/records.jsonl`, same six instances, same arms; RUN_HEADER, calibration_table.csv and AMPLITUDE files of that run were not supplied to the build side, the amplitude blocks are inside its RESULT.json):
| Arm | codename (f16) | category (f16) | mean (f16) | mean (bf16) |
|---|---|---|---|---|
| BASELINE | 0.724 | 0.125 | 0.424 | 0.322 |
| MLP0_L24_ZERO | 0.463 | 0.057 | 0.260 | 0.260 |
| MLP0_L24_MEAN | 0.463 | 0.051 | 0.257 | 0.257 |
| MLP_L24_MEAN | 0.417 | 0.030 | 0.224 | 0.226 |
| MLP0_L24_PATCH_MATCHED | 0.716 | 0.123 | 0.420 | 0.329 |
| STRENGTH_MATCHED_RANDOM | 0.678 | 0.112 | 0.395 | 0.349 |

The two numeric regimes differ in the baseline residual (0.424 versus 0.322), not after the hook: the ZERO and MEAN residuals agree to three decimals in the mean and, family by family, to within 0.005 (codename) and 0.006 (category). Per instance, three of the six instances give identical post-ZERO residuals in both regimes (0.2447, 0.2323, 0.2334) and the other three differ by up to 0.015. The reading is a residual floor of about 0.23 to 0.35 per instance, reached from different starting points, rather than a dose-dependent erasure. The larger score gain in float16 under ZERO (+11.31 points against +3.23 in bf16; paired difference +8.08, bootstrap interval of the six instance differences [+3.10; +13.78], descriptive) is therefore a larger distance to the same floor: the float16 baseline score is lower (0.692 against 0.771). The same regime difference appears for the PATCH and RANDOM arms (+1.06 and +4.44 points, intervals [-0.37; +2.65] and [+0.40; +9.33]). With six instances and a floor-limited quantity, none of this is an inferential result.

## Nature of the pilot manipulation, and the sensitivity calibration
The pilot intervention acts on activations, not on weights: a forward pre-hook multiplies the input of `layers[24].mlp.down_proj` on channels 0..3071 by
(1 - dose) at every processed token. At dose 1 this is a zero-ablation of those 3072 channels (of 12,288 in that layer's MLP, one layer of 36); for the forward
pass it is equivalent to zeroing the corresponding input columns of the weight matrix. It is not a functional (information-removing, on-distribution) ablation,
and it is not an organizational (C_O-targeted) intervention.

`t346_fca_e1_calibration.py` (targeted version) asks whether an FCA_E1 change obtained with ZERO at mlp0@L24 persists with MEAN, with a matched conditional
substitution (PATCH), and beyond an untargeted perturbation of matched strength (RANDOM_MATCHED). Default arms, 6 instances (seeds 2026101801-2026101806):
CAL_BASELINE, CAL_SHAM, MLP0_L24_ZERO, MLP0_L24_MEAN, MLP_L24_MEAN, MLP0_L24_PATCH_MATCHED, STRENGTH_MATCHED_RANDOM; optional strong-perturbation controls:
MLP_L24_ZERO, ATTN_L24_MEAN, MLP_L20_28_MEAN, MLP_L16_31_MEAN.
- MEAN: per-channel means over every processed token of reference E1 runs on two disjoint instances (seeds 2026102001-2026102002; padding positions of batched
  decoding are included).
- PATCH: in prompts containing "Auxiliary identifier for OBJ: CODE. Semantic category: CAT." (poison and erase conditions, reinjection diagnostic), the mlp0@L24
  activations of the prompt positions are replaced by those of the same prompt with another listed code and category, chosen so that the token length is equal and
  the tokens differ only inside the span of that sentence; decoding steps and prompts without the sentence are untouched; applications and skips are counted.
- RANDOM_MATCHED: the output of layers[24].mlp.down_proj receives, token by token, a vector along one fixed random direction (seed 2026101805) whose norm equals
  the norm of the change ZERO on mlp0 would produce at that output.
- Strength of every arm on four neutral texts and on 13 E1 prompts captured from the reference runs: relative change of the hidden state after layer 27 and of the
  last hidden state, KL(base || arm) (mean over positions and at the last position), top-1 agreement.
- Numerical regime: historical FCA R6 (bnb8, float16 compute) by default; `--quantization none --dtype bfloat16` selects the unquantized bf16 regime (recorded).
- Not measured here: Gamma, Delta, R, Lambda on the modified states.

```
PY t346_fca_e1_calibration.py --selftest
PY t346_fca_e1_calibration.py --run --authorize-gpu-run I_AUTHORIZE_T346_FCA_CALIBRATION_GPU_RUN [--quantization none --dtype bfloat16] [--resume] [--instances N] [--arms A,B]
```
Duration: not measured. Extrapolation from the bnb8 / float16 pilot (about 144 s per instance and arm): 7 arms x 6 instances, about 1.7 h, plus about 5 min of reference runs; the bf16 regime has not been timed.

## FCA_E1 in two numerical regimes (same pilot script, same instances, same sampling seeds)
Second run (T353): the same `t346_fca_e1_pilot.py` (SHA-256 `c0eb6944...`) launched by the wrapper `T354_grouped_gpu_pilot.py` (SHA-256 `4798e9ec687c71228a1c9dfd8174345df8152ea25d1a47d802aea67e51d7082a`,
written outside this bundle, copied in `results/t353_bf16_unquantized/wrapper/`) with `--quantization none --dtype bfloat16`, 20 instances, five arms. Files: `results/t353_bf16_unquantized/`
(the 5 x 20 FCA records.jsonl stay with the operator).

| Regime | BASELINE = SHAM | ABLATE_025 | ABLATE_050 | ABLATE_100 | mean paired difference vs SHAM (descriptive 95 % interval) | instances with a positive difference |
|---|---|---|---|---|---|---|
| bnb8 + float16 compute (historical R6) | 0.6875 | 0.6857 | 0.6772 | 0.6888 | -0.0018 [-0.015, +0.011]; -0.0103 [-0.026, +0.004]; +0.0013 [-0.025, +0.029] | 9, 6, 7 of 20 |
| bfloat16, no quantization (T353) | 0.7515 | 0.7960 | 0.8046 | 0.8044 | +0.0444 [+0.020, +0.070]; +0.0531 [+0.029, +0.079]; +0.0528 [+0.027, +0.081] | 16, 18, 17 of 20 |

- SHAM is bit-identical to BASELINE on all 20 instances in both regimes; all 200 instance-arms are valid.
- In bf16 the change comes entirely from the distributional part of the score: s_distributional 0.503 -> 0.592 / 0.609 / 0.609 and residual_effect
  0.405 -> 0.332 / 0.320 / 0.320 (less residual influence of the erased code word on the arbitrary-choice distributions); s_direct stays at 1.0 and the
  validity quantities do not move. In bnb8 / float16 neither quantity moves (residual_effect 0.519 -> 0.530 / 0.554 / 0.540).
- Across regimes, the per-instance baselines correlate (r = 0.65; bf16 is 0.064 higher on average) but the per-instance ablation differences do not
  (r = -0.01, +0.03, -0.11 at the three doses). One bf16 instance (E1-2026101814) leaves the 0.5 floor under ablation (+0.21 to +0.25); without it the
  dose-1 mean difference is +0.042 with 16 of 19 instances positive.
- The ZERO effect on FCA_E1 therefore depends on the numerical regime of the same model. The C_L24 certifications were obtained in bf16; PCI_T is only
  usable in float16 (it saturates in bf16); FCA_E1 has not been measured in float16 without quantization.
- Measured durations (bf16): acquisition loop 8,498.1 s for 100 instance-arms (about 85 s each, faster than the 144 s of bnb8), model loading 89 s.
- The T353 PCI_T bf16 run reproduces the earlier bf16 PCI_T run exactly (60 of 60 records identical apart from timings).
