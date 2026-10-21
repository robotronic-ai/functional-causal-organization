# Fidelity-First 10 Result

Status: `FIDELITY_QUALIFIED_BUT_CLOSURE_REFUTED`.

The 288D rich proxy was reduced with a target-blind PCA and quotient selection procedure. PCA20 was the smallest representation that passed the declared fidelity gate:

- pairwise-distance Spearman: 0.9988795822
- kNN-3 overlap: 0.9305555556
- leave-one-out rich reconstruction R2: 0.6032845178
- construction variance retained: 0.9810200390

The first quotient that passed the declared distortion/compression gate was K16:

- fibers: 16
- nontrivial fibers: 6
- systems in nontrivial fibers: 14 / 24
- nontrivial coverage: 0.5833333333
- within-fiber rich distortion: 0.1722882231
- rich variance explained by fibers: 0.8277117769

After this representation and quotient were frozen, behavioral closure was inspected. The K16 quotient contained four collision fibers and four collision pairs. The behavioral disagreements were localized to one REVOKE:H1 coordinate, one SUBSTITUTE:H2 coordinate, and two SUBSTITUTE:H3 cases.

This result establishes that the current rich-proxy geometry can be represented faithfully enough to pass the declared metrology gate while the first fidelity-qualified quotient remains functionally insufficient.
