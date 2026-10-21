# Observable-semigroup minimal realization theorem

Source material: `SEMIGROUP_THEOREM.md`. This generalizes the deterministic Koopman
formulation (`docs/theorem-and-proofs.md`) to an algebraic structure that also covers
deterministic controlled systems and finite-state Markov expectation dynamics.

## Motivation and algebraic setup

The essential ingredients are: a vector space of observables, a semigroup of linear
operators that propagate observables backward through future dynamics or
interventions, a set of initial source functionals, and a declared readout subspace.

Formally: let `K` be a field, `W0` a `K`-vector space of observables, `U` a set of
declared generators, `T_u : W0 -> W0` a linear operator per generator, `D subset W0`
the declared readout subspace, and `eta_z in W0*` an initial source functional per
`z in Z`. Let `M` be the free monoid of finite generator words, `T_w` the operator
product for word `w`. Define the invariant readout closure
`W = span{ T_w d : d in D, w in M }`, and restrict every source functional to `W`.

## Predictive state space and exact generation

Define `V = span{ T_w* eta_z : z in Z, w in M } subset W*`. For each generator `u`,
`A_u = T_u* |_V`; for each `d in D`, `c_d(v) = v(d)`. The initial predictive state is
`eta_z|_W`.

**Exact generation theorem.** For every source `z`, word `w`, readout `d`:
`c_d(T_w* eta_z) = eta_z(T_w d)`. So `(V, {A_u}, {eta_z}, {c_d})` exactly generates
every declared future readout series under every declared generator word.

## Universal minimality theorem

For any reachable exact linear realization `(S, {B_u}, {s_z}, {c'_d})` with
`S = span{ B_w s_z }` and `c'_d(B_w s_z) = eta_z(T_w d)` for every `d, z, w`, there is a
surjective linear map `P : S -> V` with `P(B_w s_z) = T_w* eta_z`, intertwining every
generator (`P B_u = A_u P`) and preserving declared outputs (`c'_d = c_d P`).
Consequently `dim(V) <= dim(S)` when finite, and any two minimal reachable exact linear
realizations are isomorphic.

**Well-definedness proof.** Suppose `sum_i a_i B_{w_i} s_{z_i} = 0` in `S`. Apply any
future continuation `v` and declared output `d`:
`0 = sum_i a_i c'_d B_v B_{w_i} s_{z_i}`; exactness converts this to
`sum_i a_i eta_{z_i}(T_{w_i} T_v d) = 0`, i.e.
`sum_i a_i (T_{w_i}* eta_{z_i})(T_v d) = 0`. Since `{T_v d}` spans `W`,
`sum_i a_i T_{w_i}* eta_{z_i} = 0` in `W*`. So `P` is well defined.

## Generalized Hankel operator

`H[(d,v),(z,w)] = eta_z(T_w T_v d)` — rows are future readout continuations, columns
are source/history states. When the relevant spans are finite-dimensional,
`rank(H) = dim(V)`: the semigroup version of the future-readout Hankel identity.

## Two corollaries

**Deterministic controlled corollary.** For deterministic generators
`F_u : X -> X`, take `T_u d = d o F_u` and `eta_z = epsilon_{E(z)}`. Then
`T_w* epsilon_x = epsilon_{F_w(x)}`, and the semigroup theorem reduces exactly to the
controlled Koopman predictive realization. Exhaustively verified: 0 failures across
5,103 (three-state, two-generator) cases
(`results/controlled_finite_theorem.json`).

**Markov expectation corollary.** For a finite-state Markov kernel `P_u(x'|x)`,
define `(T_u d)(x) = E[d(X') | X=x, u]`. The same theorem gives a minimal linear
realization of all **expected** future declared readouts under every control word; the
predictive states are generally linear functionals corresponding to distributions, not
only point evaluations. Exhaustively verified in exact rational arithmetic: 0 failures
across 243 (two-state, two-kernel) cases
(`results/markov_expectation_corollary.json`).

## Scope warning

The Markov expectation corollary preserves expected future declared observables. It
does **not** automatically preserve the full joint future path law. Full path-law
equivalence requires either a sufficiently rich family of history/cylinder observables,
or a direct path-law / probabilistic-bisimulation formulation — see
`docs/stochastic-extension-and-open-questions.md`, which is explicit that this
extension remains `OPEN`.
