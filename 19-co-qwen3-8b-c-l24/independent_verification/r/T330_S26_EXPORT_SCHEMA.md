# S26 → S27 deterministic interface (version T330.1)

S26 is the GPU acquisition proposed by the user/Claude; T330 implements the independent CPU S27 lower-bound calculation. No new S26 GPU acquisition is claimed here. Produce one compressed `.npz` file `S26_U26_FULLSOFTMAX_EXPORT.npz` with these keys, using `allow_pickle=False`-compatible arrays:

| Key | Shape / type | Required meaning |
|---|---|---|
| `world` | scalar int (26) | u26 only |
| `fixed_v_before_analysis` | scalar bool (True) | token set V frozen before donor/gauge return analysis |
| `gauge_names` | `(10,)` string | identity and ordered Q1 gauges shown in script |
| `token_ids` | `(V,)` distinct integers | same tokens in all gauges and donor masks |
| `logp_v` | `(10,4096,V)` float64 | natural logarithms of **full-softmax** token probabilities, temperature 1, no truncation |
| `tail_mass` | `(10,4096)` float64 | exact probability outside V (`1-sum(exp(logp_v))`) |
| `return_scores` | `(10,11,V)` float64 | same-role L24 t+1 frozen scalar readout `f_(g,r)(token)` using fixed prefix/context; role indices 0–10 |
| `delta_t1` | `(10,11)` float64 | calibrated absolute Q1 resolution at the **t+1** readout; no inferred common decoder |

Optional but highly recommended:

| Key | Shape / type | Meaning |
|---|---|---|
| `tail_signed_posneg` | `(10,11,2048,2)` float64 | for each pair `x0=x with bit r zero`, `x1=x0 OR (1<<r)` in increasing `x0` order: index 0 = `sum_{token outside V} max(p0-p1,0)`; index 1 = `sum max(p1-p0,0)`; must use identical full-token identities in both donors |
| `token_pair_js_bits` | `(10,11,2048)` float64 | exact full-vocabulary Jensen–Shannon information for each binary donor pair, same ordering, **not** an unconditional bit-token MI |

If full-token distributions are already computed to obtain `I(X;T)`, both optional fields can be computed without running `L24(t+1)` on additional tokens. Their absence does not invalidate the conservative bound, but may make it zero when the excluded probability mass is large.

### Normalization and evidence

S27 checks every `sum_t in V exp(logp_v) + tail_mass == 1` within `2e-9`, requires 4,096 donor arms in all ten gauges, and checks that a return lower bound never exceeds the supplied conditional full-token channel. Hash the raw S26 export and preserve the fixed token set and its selection rationale. The same-timestep prefix, cache, scalar projection, gauge reconstruction, and `delta_t1` provenance must be documented by S26; the CPU analyzer cannot establish these physical properties from numeric arrays alone.

### S27 numerical witness

A real threshold is chosen **only from a gap in sorted `return_scores` of observed V tokens that exceeds `delta_t1`**. This event is a measurable function of the *entire physical scalar f(T)* even if unknown tokens lie on either side; unknown tokens are treated adversarially through the missing tail. The event supports a sound information lower bound for `f(T)` but **does not certify** that two full-vocabulary *delta-connected equivalence classes* remain distinct. All-support delta-closure and temporal functional-role typing are separate final gates.

### Launch

```bash
PY=/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python
"$PY" -I -u T330_S27_DETERMINISTIC_R_BOUNDS_CPU.py --selftest
"$PY" -I -u T330_S27_DETERMINISTIC_R_BOUNDS_CPU.py \
  --input FAST_GPU_RESULTS/S26_U26_FULLSOFTMAX/S26_U26_FULLSOFTMAX_EXPORT.npz \
  --output FAST_GPU_RESULTS/T330_S27_U26_CPU/RESULT.json
```

If Claude's actual S26 output uses other names or axes, use a small schema adapter to convert its arrays; do not guess the axes or the definition of `I(X;T|bit)`.
