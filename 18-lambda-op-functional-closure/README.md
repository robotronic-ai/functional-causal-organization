# 18 — Λ at operation level: functional closure (synthetic universes + Qwen3-8B)

This campaign asks the closure question of manuscript §17.7 at the operation (`OP`) level of generalization: when two systems receive the same compact organizational description, do they also receive the same functional access profile? A description is **functionally closed** on a panel when no fiber of the description — no set of systems sharing it — contains two systems with different behavioral targets.

The operation algebra is fixed throughout: `ID / KEEP / REVOKE / SUBSTITUTE`, applied to three addressed content handles. The generalization level stays at `OP`; no global scalar C_O is ever formed.

It supports manuscript **§17.7** (closure and minimal kernel) and the Λ discussion of **§7.3 / §17.5**. It is distinct from folder `08` (the deductive theory of quotient sufficiency, which this campaign applies) and from folder `09` (the Real-C0 / rich-Δ fiber screens of the CF-A Λ lineage): no file is shared with either.

## Results in brief

**Synthetic universes.** Closure holds at every width tested.

| Width | Coverage | Result |
|---|---|---|
| 2 bits | exhaustive (256 systems) | PASS — trivial, the description is injective |
| 3 bits | exhaustive (16,777,216 deterministic maps) | PASS — non-trivial fibers, zero functional collision |
| 4, 6, 8 bits | declared transport envelope (1,024 systems each) | PASS — non-trivial fibers, zero functional collision |

**Qwen3-8B.** On a real model the state space is continuous, so closure can only be tested on sampled panels: one collision refutes a description, while its absence certifies nothing beyond the panel. Every candidate is frozen before a fresh panel is opened, and never refit on it.

- A sequence of compact descriptors — fixed operation-address descriptors, active causal kernels, relational and cross-target routing topologies, a high-fidelity PCA quotient, and typed content-effect kernels — was **refuted on fresh panels** of 64 to 128 systems. The failures are informative: they are compressive rather than near-injective, and several descriptors that transport their geometric fidelity still merge functionally different systems. Magnitude of causal propagation is not enough; which content is bound to which role must be kept.
- A **metrology audit** (6 systems × 3 independent reloads) found exact repeatability of both behavior and internal measurements in the CUDA/bfloat16 stack, so the refutations are not run-to-run noise.
- A **component audit** separated what an estimator measures from what the manuscript defines, and froze formula contracts for Γ, Δ, R and Λ before any further selection.
- Under those contracts, a Λ-only quotient — the joint typed Λ signature truncated to its top-ranked response at every operation × source × internal-intervention cell — was selected on 12 development systems and then **passed a single prospective holdout** of 32 fresh systems with no refit: 7 fibers, 90.6 % of systems in non-trivial fibers, **zero heterogeneous collisions**, adjusted mutual information 0.806 against a permutation 95th percentile of 0.119.
- A **second fresh replication** (64 systems, identical frozen candidate, same promotion rule) is frozen and ready; it has not been run.

## Scope

The holdout supports one frozen OP-level Λ quotient on the declared finite Qwen battery. It is not a closure result for the complete profile C_O, nor for all Qwen3-8B systems, and it does not show that Γ, Δ or R are unnecessary outside this battery. The development selection that preceded the holdout was adaptive (a grid of 66 declared quotient/resolution screens on 12 systems); the holdout and the pending replication are what carry the evidence. The information captured by the quotient is concentrated on the `SUBSTITUTE` cells of the 12-cell behavioral target.

## Layout

- `docs/RESULTS.md` — phase-by-phase results with the figures that matter.
- `campaign/` — the frozen campaign record: protocols (`*_PROTOCOL.md`, `*_PLAN.md`), per-phase result notes (`*_RESULT.md`, adjudication), canonical scripts (`qwen3_8b_*.py`, `qwen3_8b_lambda_op_engine/`), raw results (`qwen3_8b_results/`, `results/`), the synthetic-closure bundle (`synthetic_closure/`), `CAMPAIGN_STATUS.json` and `MANIFEST_SHA256.json`.

Protocol and result files are carried over unchanged, so their recorded hashes remain valid. In the scripts, the author's local working-directory constant was replaced by `FCO_WORKDIR` (default: current directory); no logic was changed.

## Reproduce

```bash
cd campaign
python verify_package.py                           # prints PACKAGE_VERIFICATION_PASS
python synthetic_closure/scripts/run_lambda_op_closure.py --help
```

`verify_package.py` checks the recorded verdicts, recomputes scorer fixtures (including the holdout and replication scorers on pass/refutation fixtures) and parses every script; it needs `numpy` only. The Qwen3-8B phases require `torch`, `transformers`, a local `Qwen3-8B/` directory and a CUDA GPU; each phase runs as `prepare → run → score`.

See the root `README.md` for the protocol → manuscript-section table.
