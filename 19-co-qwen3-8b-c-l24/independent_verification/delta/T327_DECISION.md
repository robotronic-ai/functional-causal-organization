# T327 Decision — Delta Nonzero on the Declared Qwen3-8B Donor Contract; R Next

## Evidence received

The uploaded T326 independent verifier reports `DELTA_FINITE_U26_EXACT_GAME_VERIFIED__R_STRUCTURAL_ONLY`. For Delta, it verifies all 20,480 entries of the already acquired ten-gauge, 2,048-coalition finite game; the earlier verifier independently replayed 100 selected gauge-coalitions against the source FULL12 cube. This is NOT a new exhaustive raw-cube replay. The game and its Shapley vector were rechecked independently in this turn from the uploaded T325 NPZ, with its pinned SHA-256.

For each of the ten Q1 gauges and both context settings, the declared eleven donor-policy bits yield the measured conditional information game

\[
 v(S) = I(P;Y_S\mid C) = \begin{cases}
 1\ \mathrm{bit}, & S\cap F\ne\emptyset;\\
 0,& S\cap F=\emptyset,
 \end{cases}
\]

where the five distinct declared reader roles are

`F = {heads[4,5,6,7], mlp0, mlp1, mlp2, mlp3}`.

Their exact Shapley information contributions are 1/5 bit each; six other role readouts receive 0 in the observed game. The operational one-layer P* count reported by T325 is 2. Therefore Delta_cap=1 and Pi_H=Pi_S=2/5, giving **Delta_H=Delta_S=0.4** on the measured C_L24 donor contract, under q_donor.

## New exact finite-alphabet refinement theorem

Assume (i) precisely the eleven retained independently intervened binary donor sites and the twelfth site conditioned as context; (ii) the same eleven distinct causal roles and Shapley definition; (iii) the richer reader for each role preserves the original one-layer delta-class as a deterministic coarsening; (iv) no additional source policies outside the finite donor alphabet are introduced. Let v'(S)=I(P;Y'_S|C) be any *monotone* richer-future game with those properties. Data processing gives v'({r})>=1 bit for all five r in F, while 1<=v'(N)<=H(P)=11 bits. The Shapley formula includes an empty-coalition term with coefficient 1/11, so each of these five roles has phi'_r>=1/11 bit. Its normalized share is at least epsilon=1/121.

For any probability vector on eleven roles with at least five entries >=epsilon, the sum of squared shares is at most

\[
 (1-4\epsilon)^2+4\epsilon^2 = 13693/14641.
\]

Thus inverse-Simpson normalized participation and Shannon normalized participation satisfy

\[
 \Pi_S,\Pi_H\ \ge\ \frac{1/(13693/14641)-1}{10}
 = \frac{474}{68465} >0.
\]

The number of source-policy causal classes obeys 2<=|P*|<=2^11 on the stated finite source alphabet. Therefore Delta_cap>=1/11, and

\[
 \boxed{\Delta_H,\Delta_S\ \ge\ \frac{474}{753115}
 \approx 0.00062938595035 >0}
\]

under **any such readout refinement**, without needing the exact count of those refined P* classes. This is a deliberately conservative theorem, not a claim that the measured value 0.4 survives numerically under an enriched future.

### Status decision

- `Delta_H_Qwen3_8B_C_L24 = NONZERO_CERTIFIED` on the declared eleven-bit u26 donor interface and the unchanged causal-role contract.
- `Delta_S_Qwen3_8B_C_L24 = NONZERO_CERTIFIED` on the same contract.
- Claim that the **full, all-state, all-intervention Qwen3-8B quotient** has been identified: **not made**. A change of causal boundary or source-policy alphabet is outside the theorem.

## R remains open; execute real u26 donor experiment

The latest R report concerns a surrogate MLP projection perturbation on a generic prompt. The emitted greedy token did not change (12095 in both arms), while forcing the same token made the next L24 activations exactly equal bitwise, and subsequent higher-layer logits could still differ. This supports the L24-specific cache bypass lemma and does not yet demonstrate positive same-role reentry for the frozen donors in world u26.

The accompanying T327 GPU script explicitly builds the T276/T286 frozen donor and background assignments, screens one-bit donor flips in u26, measures emitted token differences, then tests free-token L24(t+1) role readings and same-token clamped controls for the first witness. It optionally repeats the identical pair across all Q1 gauges. It is a NEW GPU integration script; only its CPU selftests have been run here. A recorded 1e-6 contrast is an exploratory signal, **not** a calibrated temporal gauge delta or a formal R quotient certificate. No intrinsic R status is promoted by code that has not been run.
