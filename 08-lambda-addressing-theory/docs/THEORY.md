# Theory

## 1. Typed setup

Let

\[
x=(O,E,D)\in\mathcal C
\]

be a coupled system in a declared causal envelope `C`.

- `O`: intrinsic organization of the agent/system.
- `E`: environment.
- `D`: fixed-port coupling/interface between them.
- `L_tau(x) = Lambda_{E,D}^{(tau)}`: coupled target at horizon `tau`.

A candidate representation is

\[
F:\mathcal C\to\mathcal Y.
\]

The original coarse hypothesis is

\[
F_0=(\Lambda_O(O),V_E(E)),
\]

with strict typing: `Lambda_O` depends on `O` only, `V_E` on `E` only, and `D` is not hidden inside either descriptor.

## 2. Fibers and exact insufficiency

Define

\[
x\sim_F x' \iff F(x)=F(x').
\]

The unresolved-pair set is

\[
\mathcal U(F)=\{\{x,x'\}:F(x)=F(x'),\;L(x)\neq L(x')\}.
\]

`U(F)` is the primary exact object. A metric defect is secondary:

\[
\Omega(F)=\sup_{\{x,x'\}\in\mathcal U(F)}d_L(L(x),L(x')).
\]

## 3. Minimal admissible representations

Let `A` be a finite or well-founded library of descriptors constructed independently of `L`.

For a subset `K` of descriptors, let `F_K` concatenate them with any fixed baseline descriptors.

A key `K` is sufficient when

\[
\mathcal U(F_K)=\varnothing.
\]

It is minimal when no strict sub-key remains sufficient.

Different minimal keys can induce the same semantic partition, so the invariant object is the induced partition of the domain, not the chosen coordinate names.

## 4. Interaction order

Descriptors can be indexed by their support, e.g. `D`, `E`, `ED`, or `OED`.

The minimal interaction order is the smallest maximum support size among sufficient admissible keys.

This order is relative to the declared descriptor library. If raw `O`, `E`, and `D` are admitted without compression, every function is trivially first-order.

## 5. Symmetry view

If a candidate quotient is invariant under a transformation that the target still detects, factorization is impossible.

This is the core mechanism behind the relative-alignment and ternary-parity witnesses.

## 6. Structural certification

A purely empirical equality on a finite domain is weaker than a structural result.

A structural certificate is built from target-blind causal conjugacies: typed maps on states, interventions, and boundary roles that make the intervention kernel commute.

The relevant object is a causal groupoid whose orbits represent causally admissible recodings.

If the fibers of `F` are exactly the groupoid orbits, and the target is invariant under every morphism, factorization is structurally certified on the declared envelope.

## 7. Temporal closure

A quotient valid at one horizon need not be dynamically closed.

For deterministic dynamics, a quotient `q` is closed when

\[
q\circ T_i=\bar T_{\bar i}\circ q.
\]

For a stochastic kernel `K`, closure means the pushed-forward transition law depends only on the quotient state, not on the microscopic representative.

Closed quotients can be iterated. Horizon-specific coincidences cannot.

## 8. Composition

If a causal quotient descends the dynamics from `S` to `Q`, and a second quotient descends the resulting dynamics from `Q` to `Z`, the composite quotient is closed.

Certification therefore composes when each stage genuinely descends the relevant causal object. Minimality does not generally compose.
