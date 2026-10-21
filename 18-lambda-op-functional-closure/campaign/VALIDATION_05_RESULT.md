# Active Causal-Kernel Validation 05 Result

Fresh Validation 05 executed 64 Qwen3-8B systems with the frozen exploratory candidate:

`HYBRID_K3K2__SUB_H1_K4__SUB_H3_K12`

Result:

- closure: `REFUTED`
- promotion status: `REFUTED`
- validation systems: 64
- fibers: 42
- nontrivial fibers: 14
- nontrivial coverage: 56.25%
- collision fibers: 9
- collision pairs: 18
- validation panel SHA-256: `4939aa08dc9946c9a371f4d042b60ddde3bc3dc24034c46df7a3f9ebf7afe2ca`

The important change relative to the exploratory panel is that the candidate is no longer close to injective on the fresh sample: more than half of the systems belong to nontrivial fibers. The failure is therefore a genuine insufficiency of this frozen active-kernel compression, not merely a low-coverage artifact.

No parameter from Validation 05 is reused to retroactively alter this candidate.
