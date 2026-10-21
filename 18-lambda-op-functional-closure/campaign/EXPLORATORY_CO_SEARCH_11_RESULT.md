# Exploratory Co Search 11 Result

## Verdict

- Status: `USEFUL_EXPLORATORY_CANDIDATE`
- Candidate: `CO_EXPLORE_PCA20_K18_LOCAL_SPLIT`
- Selection mode: `COLLISION_GUIDED_LOCAL_SPLIT`
- Source systems: 24
- Frozen global representation: PCA20
- Base quotient: K18
- Locally split base fibers: 2
- Final fibers: 20
- Nontrivial fibers: 2
- Systems in nontrivial fibers: 6
- Nontrivial coverage: 25.00%
- Within-fiber rich distortion: 8.6912%
- Behavioral collision fibers: 0
- Behavioral collision pairs: 0

## Search context

The search is explicitly target-informed. The behavioral target is used to select the final quotient and the local split rules. The global PCA20 representation was already frozen from a target-blind fidelity screen.

Pure K18 remained fidelity-qualified but had two behavioral collision pairs. Pure K20 remained fidelity-qualified but had one behavioral collision pair. The selected candidate keeps K18 and applies two local splits inside construction collision fibers.

The two local splits use frozen PCA coordinate 0 and are part of the candidate definition. They must not be modified using validation data.

## Scientific status

This is a construction candidate, not a validated closure result. It requires a fresh independent Validation 11 panel. No parameter, center, PCA component, threshold, or split rule may be refit after the validation panel is opened.

The candidate remains an operation-indexed cross-target organizational proxy. It is not a scalar validation of complete `C_O=(Gamma,Delta,R,Lambda)`.
