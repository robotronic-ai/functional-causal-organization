# Stochastic extension notes and open questions

Source material: `STOCHASTIC_EXTENSION_NOTES.md` and `OPEN_QUESTIONS.md`.

## 1. Expectation-level extension

Let `P(dx'|x)` be a Markov kernel and define the Koopman expectation operator
`(U f)(x) = E[f(X_{t+1}) | X_t=x]`. Given a declared readout space `D`, define
`W = span{ U^k d : d in D, k >= 0 }`. The same evaluation-state construction gives a
linear predictive representation for all expected future readouts,
`E[d(X_{t+k}) | X_t=x]`. For finite state spaces this is an algebraic corollary of
`docs/semigroup-generalization.md` and is verified exactly in the included rational
audit (`docs/finite-verification-results.md`, Section 5). For infinite state spaces,
topological / measurability assumptions remain to be specified.

## 2. What this does not yet preserve

Equality of all one-time future expectations — or even all one-time future marginal
distributions — does not automatically determine the full joint future path law.
**The expectation-level construction must not be called full probabilistic
bisimulation without additional assumptions.**

## 3. Path-law extension (open)

A full path-law version can be approached in at least two ways: (1) enlarge the
observable family to a separating algebra of cylinder functions on future
trajectories; or (2) define predictive equivalence directly by equality of all
admissible future experiment laws. The second route is closer to classical
probabilistic bisimulation and predictive-state representations. Neither is developed
in this bundle.

## 4. Controlled stochastic systems

For controls/interventions `u`, use a family of Markov Koopman operators `U_u` and
close the declared readouts under every finite operator word. The resulting
representation preserves expected future readouts under every declared control
sequence — this is the stochastic analogue of the controlled deterministic corollary
(`docs/semigroup-generalization.md`).

## 5. Continuous approximate systems (open)

For continuous RNN states, exact closure may be infinite-dimensional. A practical
protocol should specify: an approximation family for `W`; a predictive norm or
divergence; a held-out horizon set; a stability criterion under retraining/resampling;
and an explicit approximation error budget. **No continuous minimality claim is frozen
in this archive** — this is the same open boundary named in campaign
`10-continuous-predictive-geometry/`, which develops the continuous approximate-metric
side of this project's theory but likewise stops short of a frozen minimality claim.

## Open questions

1. **Full stochastic path-law theorem.** Expectation-level prediction is now covered
   by the observable-semigroup theorem and a finite exact audit. The next problem is
   to characterize the minimal object preserving full future experiment laws, not only
   expected declared observables.
2. **Task-specific compression below the predictive realization.** The predictive
   realization is task-agnostic relative to the declared readout class. Characterize
   the coarsest quotient for source-labelled horizon-stable tasks, role-specific
   tasks, common-reader constraints, and constrained decoder subclasses.
3. **Reachable-excitation identification theorem.** Given a partially known parametric
   transition law, characterize the minimal additional parameter information needed
   only on the actually reached frontier. Current candidate: a quotient of unresolved
   parameter directions by those vanishing on the reachable set.
4. **Polynomial dynamics and polynomial readouts.** For Boolean polynomial systems,
   determine when Koopman closure of low-degree readouts stabilizes at finite degree
   or fills the complete function space; relate this to the project's earlier U1
   curvature-signature and higher-order derivative results.
5. **Stochastic path-law minimality.** Rigorously distinguish expected-readout
   realization, marginal-law realization, full path-law realization, and
   probabilistic bisimulation.
6. **Continuous approximate realization.** Develop an epsilon-predictive quotient and
   a low-rank approximate realization with held-out transport guarantees.
7. **Organizational co-factorization.** Keep the architecture
   `C_O <- rich causal organization -> predictive transport -> access claims`, and
   determine which organizational structures constrain the predictive realization
   without incorrectly demanding a direct function `C_O -> A`.
