# Canonical Campaign

## Stage 0 — Frozen provenance

The later Lambda audits used frozen model lineages, source snapshots, environment anchors, and validation data.

Canonical checkpoint hashes:

- CONTROL: `9d44d4211e13d4f195c81333277d7bc6c260afe3e96e3bd505b97bf17816ffab`
- DEPTH: `9ba7961df75fd81d8714ed19f2e6f6e99b048cac4072e3d04b348d899806a2aa`
- TRIDEPTH: `87e48c5be456128202fbe5a8eb7f1b91b019800f61af0235b36dde8c825c837e`

The validation source contained 512 rows: 256 BIND and 256 UPDATE, organized into 128 context groups with probes 0,1,2,3.

## Stage 1 — Real-C0 fiber search

Purpose: find two frozen intervention states equal under the preregistered baseline screening representation while leaving Lambda and task outcomes sealed during candidate selection.

Declared intervention family:

- 3 frozen lineages,
- 67 cells per lineage,
- 1 identity + 66 unordered head swaps,
- 201 total states.

Final Stage-A result:

- candidate count: `0`,
- empty candidate list SHA-256: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`,
- Stage B not instantiated.

Interpretation: no same-C0 fiber witness was found. This is not evidence that Lambda is redundant.

## Stage 2 — Full rich-Delta fiber search

The baseline was strengthened to the full typed four-probe perturbation-response trace channel.

Frozen structure:

- CONTROL + DEPTH,
- 67 cells per lineage,
- 134 states,
- 8,911 unordered state pairs,
- 128 context groups,
- probes 0,1,2,3,
- three roles: post-write memory, post-filler memory, query classifier input,
- raw feature dimension: 13,056,
- rich feature dimension: 52,224,
- no dimensional reduction,
- exact componentwise equality within tolerance `1e-7`.

Result:

- all 8,911 pairs rejected by the screening representation,
- candidate count `0`,
- Lambda never opened for candidate selection,
- Stage B not instantiated.

Interpretation: rich-Delta was injective on this finite 134-state declared intervention set at the frozen tolerance.

## Stage 3 — Layer-complete rich-Delta extension

The same structural intervention family was extended across all 12 layers without selecting favorable layers.

Frozen structure:

- CONTROL + DEPTH,
- 12 layers,
- 66 unordered head swaps per layer,
- 792 swaps + one global identity per lineage,
- 793 cells per lineage,
- 1,586 total states,
- 1,256,905 unordered state pairs,
- same full four-probe rich-Delta channel,
- tolerance `1e-7`.

Result:

- all 1,256,905 pairs rejected by the screening representation,
- candidate count `0`,
- Stage B not instantiated.

Interpretation: the measured rich-Delta object was injective on the declared layer-complete state set. This does not establish that Lambda is a function of rich-Delta in general.

## Stage 4 — Constructive late-role-rebind witness

Instead of searching a pre-existing model state space for an accidental fiber, a small executable system was designed to guarantee an exact baseline match while isolating one late causal degree of freedom.

Both systems share:

- the same payload bank `C[4]`,
- the same role-to-content pointer `P[4]`,
- the same late cue `U[4]`,
- the same readout `Y[r] = C[P[r]]`.

Only the late COMMIT edge differs:

- RELINK: `P <- U`,
- LOCK: `P <- P`.

Baseline channel:

- 262,656 paired comparisons,
- 0 mismatches,
- identical baseline SHA-256.

Late-rebind challenge:

- RELINK task accuracy: `1.0`,
- LOCK task accuracy: `1/24`,
- normalized `I(U;P_after)/H(U)`: RELINK `1.0`, LOCK `0.0`.

All decoder/control checks passed for both systems.

Interpretation: this is an exact constructive executable witness that an identical declared baseline typed channel can coexist with a maximally different late context-to-role causal channel. It is synthetic evidence of existence, not a retroactive RMT nonreduction PASS.
