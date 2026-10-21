# Exact Finite Witnesses

These examples are demonstrations of the theory, not claims about a particular trained model.

## W1. Coupling-specific information (`D`)

A fixed agent and environment expose fixed ports. Couplings vary by:

- port pairing `pi`,
- sign/polarity `s`,
- delay `d`.

At horizon 1, the target is the typed endpoint effect matrix.

Successive descriptors forget fewer coupling distinctions. The coarse representation fails because changing only `D` can change the target while all intrinsic/environment descriptors remain fixed.

A horizon-1 minimal compressed invariant can ignore signs/delays that have not yet become causally available. Extending the target to later horizons refines the minimal quotient.

## W2. Environmental dynamics (`E`)

Six environments share the same binary boundary alphabet and uniform instantaneous marginal.

They differ by copy, flip, iid, and delayed dependencies. Two iid environments also differ in a hidden variable with no causal path to the boundary.

Instantaneous variety is insufficient. A dynamic boundary quotient becomes sufficient, while the hidden causally irrelevant distinction remains safely removable.

This demonstrates the difference between:

- a variable with no causal path to the declared readout, and
- a variable whose effect merely happens to be zero in one benchmark.

## W3. Relative alignment (`ED`)

Let an environment permutation `sigma` map typed contents to boundary ports and a coupling permutation `pi` map boundary ports to roles.

The endpoint map is

\[
M=\pi\sigma.
\]

Quotienting the two sides independently destroys relative alignment. Quotienting the pair under a common diagonal port relabeling preserves `M`.

For four ports:

- 24 environments,
- 24 couplings,
- 576 raw pairs,
- 24 distinct endpoint maps after the correct joint quotient.

This is an exact example where

\[
(E/G)\times(D/G)
\]

is strictly coarser than the useful diagonal quotient of `E x D`.

## W4. Irreducible ternary relation (`OED`)

Let `o,d,e` be binary orientation bits and let the endpoint action depend on

\[
t=o\oplus d\oplus e.
\]

All three pairwise invariants

\[
o\oplus e,\quad o\oplus d,\quad e\oplus d
\]

are unchanged under global complementation

\[
(o,d,e)\mapsto(o\oplus1,d\oplus1,e\oplus1),
\]

but `t` flips.

Therefore all pairwise information can be insufficient while one ternary invariant is sufficient.

This is an exact witness that causal sufficiency need not decompose into singleton and pairwise terms.
