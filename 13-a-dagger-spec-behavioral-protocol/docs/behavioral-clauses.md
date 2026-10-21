# Behavioral clauses — reviewer map

A-dagger v0.28 contains eleven behavioral clauses. Each is defined by a target/null contrast and a declared inference rule rather than by an internal architecture claim.

| Clause | Role | Qwen3-8B final disposition |
|---|---|---|
| DELAYED-USE | delayed use of fresh episodic content | PASS (migrated result) |
| SUBST | sensitivity to substituted initial content | PASS (migrated result) |
| INTERFERENCE | retained use in the presence of a distractor | PASS (migrated result) |
| POSTCUE-MULTI-USE | multiple post-cue uses of one fresh episode | PASS (migrated result) |
| CONTROL | suppress-now / recover-later control | PASS (migrated result) |
| TEMPORAL-AUTONOMY | interval-sensitive behavior without target leakage | FAIL (migrated result) |
| ELAPSED-TIME-CONTROL | action conditional on elapsed time | FAIL (migrated result) |
| HARD-ERASURE | loss of later influence after revocation | FAIL (fresh v0.28) |
| SELECTIVE-ERASURE | revoke one binding while preserving another | FAIL (fresh v0.28) |
| SOURCE-BINDING | bidirectional source-content binding | PASS (migrated result) |
| CAUSAL-OWNERSHIP | advantage in a genuinely controllable environment over an action-decoupled null | FAIL (fresh v0.28) |

The resulting five-field profile is `(2, FAIL, FAIL, PASS, FAIL)`.

For migrated rounds, `V027_TO_V028_MIGRATION.json` records why the completed result is carried forward without re-estimation. For the two v0.28 reruns, the raw result and final verification files are provided directly.
