# Raw witnesses of the Λ₀ lower relation on C_L24, u26 — secondary reading (> 2 δ_k)

These files list, for each cell positive at 2 δ_k (39 cells at t, 27 at t+1), the witness configuration with the largest
margin, its 20 raw scores (10 replicas × x, x^i), the 10 differences and δ_k. Column `margin_over_2delta` is
min over replicas of |difference| − 2 δ_k. Every cell listed is also positive at the primary threshold δ_k (the
differences exceed 2 δ_k, hence δ_k). `delta_per_reader.csv` gives δ_k with the replica and mask at which it is attained.

The current `scripts/extract_lambda_witnesses_cpu.py` writes the same layout with the column renamed
`margin_over_threshold`, and `--factor 1` produces the witnesses for the primary reading (> δ_k, 68 cells at t).
