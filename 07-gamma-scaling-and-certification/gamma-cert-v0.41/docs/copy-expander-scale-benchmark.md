# The copy-expander causal benchmark: Γ as exact graph conductance, calibrated to n = 1024

Translated and restated from `GAMMA_CERT_V04_NOTE.md` §VI, with the executed numbers from
`scripts/gamma_copy_expander_scale.py` / `results/copy_expander_scale_results.json`.

## Construction

Build `n` causal roles, each carrying `d` independent binary bits. A `d`-regular undirected
graph assigns one independent binary channel to each edge, in each direction; the next state
simply copies these bits to neighboring roles. For any cut `A|B`:

\[
J(A\to B) = |E(A,B)|,
\qquad
J(B\to A) = |E(A,B)|.
\]

Since each role carries `d` bits, `log|A| = d|A|`, so

\[
\boxed{\Gamma(A|B) = \frac{|E(A,B)|}{d\min(|A|,|B|)}}
\]

and **Γ is exactly the conductance of the `d`-regular graph** — this is a legitimate scale
certificate for Γ itself, not a surrogate, because the causal objective is analytically identical
to graph conductance. Cheeger's inequality (`λ₂/2 ≤ Γ ≤` spectral-sweep candidate) therefore
applies directly to Γ on this benchmark.

## Analytic calibration: complete copy network

For the complete graph `K_n` (every role connected to every other), `Γ = ⌈n/2⌉/(n-1)` exactly,
and for `n` even the Cheeger lower bound is exactly equal to Γ:

| `n` | `d` | `Γ_exact` | Cheeger lower bound | gap |
|---:|---:|---:|---:|---:|
| 64 | 63 | 0.507937 | 0.507937 | 0.000000 |
| 128 | 127 | 0.503937 | 0.503937 | 0.000000 |
| 256 | 255 | 0.501961 | 0.501961 | 0.000000 |
| 512 | 511 | 0.500978 | 0.500978 | 0.000000 |
| 1024 | 1023 | 0.500489 | 0.500489 | 0.000000 |

## Sparse benchmark: random 8-regular graphs, exact calibration up to n = 20

For small `n`, brute-force exact conductance is computed and checked against the spectral
interval (assertion `lower ≤ exact + ε ≤ upper + ε` holds on every case):

| `n` | `d` | `λ₂` (normalized) | Cheeger lower | spectral-sweep upper | `Γ_exact` |
|---:|---:|---:|---:|---:|---:|
| 12 | 8 | 0.802115 | 0.401057 | 0.416667 | 0.416667 |
| 16 | 8 | 0.686369 | 0.343185 | 0.375000 | 0.375000 |
| 20 | 8 | 0.511214 | 0.255607 | 0.300000 | 0.300000 |

The Fiedler sweep recovers the exact optimum in all three cases (spectral-sweep upper bound
equals `Γ_exact`).

## Scaling without cut enumeration, up to n = 1024

Same procedure, no exhaustive partition search (brute force is combinatorially infeasible past
`n≈20`); intervals are certified directly from the declared causal benchmark — lower bound from
Cheeger, upper bound from an actually evaluated Fiedler-sweep cut:

| `n` | `d` | `λ₂` (normalized) | certified `Γ` interval |
|---:|---:|---:|---|
| 64 | 8 | 0.397757 | [0.198879, 0.296875] |
| 128 | 8 | 0.379180 | [0.189590, 0.277344] |
| 256 | 8 | 0.348194 | [0.174097, 0.283203] |
| 512 | 8 | 0.357323 | [0.178661, 0.286133] |
| 1024 | 8 | 0.345946 | [0.172973, 0.267090] |

All five random-8-regular runs complete in well under a second (`elapsed_sec` `0.005`–`0.021` in
`results/copy_expander_scale_results.json`), illustrating the scalability claim of §VII in
`additive-packing-certificate.md`: this fast graph-certificate stage alone gives a non-trivial,
provably correct two-sided interval at 1024 roles without ever enumerating a cut exhaustively.
