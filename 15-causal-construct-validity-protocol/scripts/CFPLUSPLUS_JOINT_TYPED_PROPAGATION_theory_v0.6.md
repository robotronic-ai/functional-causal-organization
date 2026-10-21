# CF++ JOINT TYPED PROPAGATION REPAIR v0.6

**Date:** 2026-09-14  
**Status:** theory frozen after the v0.5 counterexample and before candidate implementation.

## 1. Failure being repaired

The v0.4 repair retained:

\[
\mathcal B_q
=
[(r_j,K_{j,q})_j]
\]

with one marginal channel per certified role.

The v0.5 witness showed that identical typed marginal channels can hide
different access-relevant information in cross-role dependence.

Therefore the object required by the current evidence is not the list of role
marginals but the **typed joint intervention-response channel**.

## 2. Joint typed propagation object

Let:

\[
Y_{\mathcal R}
=
(Y^{(1)},\ldots,Y^{(m)})
\]

be the jointly observed response of all certified roles under the declared
causal quotient.

Define:

\[
\boxed{
K^{\rm joint}_{q,\mathcal R}
:
p
\mapsto
P_q(Y_{\mathcal R}\mid do(P=p)).
}
\]

The primary differentiation object becomes:

\[
\boxed{
\mathcal D_q^{++}
=
(
\Delta_{\rm prop},
[K^{\rm joint}_{q,\mathcal R}]_{\rm adm}
).
}
\]

The equivalence class is under:

- admissible source recodings;
- type-preserving role transports;
- coherent bijective output recodings within certified role types;
- representation-level duplication removed by the already frozen causal role
  quotient.

The candidate organizational profile is:

\[
\boxed{
C_F^{++}
=
[
(
\Gamma,
\mathcal D_q^{++},
R(k)
)
].
}
\]

No fourth factor is added.

## 3. Static-access sufficiency theorem

Fix:

- source/intervention distribution \(q(P)\);
- certified typed role tuple \(\mathcal R\);
- a target variable \(T=f(P)\), possibly stochastic conditional on \(P\);
- a response/action space;
- a loss function \(\ell(T,a)\);
- an admissible decision-rule class that can read \(Y_{\mathcal R}\).

If two systems have the same typed joint channel:

\[
K^{\rm joint}_{1}
=
K^{\rm joint}_{2}
\]

up to an admissible bijective recoding transported through the decision rule,
then every fixed decision rule has the same expected loss in both systems.
Consequently the optimal achievable expected loss is identical.

Therefore every **static access consequence that depends only on what can be
read from the certified joint response** is constant on fibers of the joint
channel.

### Proof

The joint law:

\[
P(P,T,Y_{\mathcal R})
\]

is determined by \(q(P)\), the fixed target kernel \(P(T|P)\), and
\(K^{\rm joint}(Y_{\mathcal R}|P)\).

Equal channels therefore induce equal joint laws.  Expected loss of any fixed
decision rule is a functional of that joint law, hence equal.  Taking the
infimum over the same admissible rule class preserves equality.

A bijective response recoding changes only coordinates; transporting the
decision rule by the inverse bijection preserves the expected loss.

QED.

## 4. Scope

The theorem covers static typed access tasks, including the information-loss
modes exposed by:

- v0.3 typed content-to-role binding;
- v0.5 cross-role synergy.

It does not yet cover:

- delayed access across time;
- update under intervening correction;
- path-specific gating;
- temporal order;
- hidden-state controllability not expressed in the declared response process.

Those require a trajectory/intervention-process extension rather than a larger
static scalar.
