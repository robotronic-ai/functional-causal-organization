# Γ axiom characterization

## The formulas under test

The manuscript's cut-level causal-integration score is built from the directional
interventional mutual informations of a bipartition `π = (A, B)`:

\[
J_q(A\to B) = \mathbb{E}_{b\sim q_B}\, I\big(A; B' \mid do(B=b)\big),
\]

\[
x_\pi = \frac{J_q(A\to B)}{\min(\log_2|A|,\log_2|B|)},
\qquad
y_\pi = \frac{J_q(B\to A)}{\min(\log_2|A|,\log_2|B|)}.
\]

The cut score is `F(x_π, y_π)` for some aggregator `F : [0,1]^2 → [0,1]`, and the system-level
score is the minimum over admissible bipartitions:

\[
\Gamma_q = \min_\pi F(x_\pi, y_\pi).
\]

The manuscript's current choice is the arithmetic mean, `F(x,y) = (x+y)/2`. This campaign asks
whether that choice is forced by axioms, or merely conventional.

## Round 1 — minimal axioms (protocol v0.1): non-unique

The frozen protocol declares seven minimal axioms on `F`:

- **G0** — domain `[0,1]²`, codomain `[0,1]`.
- **G1** — symmetry: `F(x,y) = F(y,x)`.
- **G2** — invariance under bijective causal relabeling.
- **G3** — coordinatewise monotonicity.
- **G4** — diagonal calibration: `F(t,t) = t`.
- **G5** — boundary values: `F(0,0) = 0`, `F(1,1) = 1`.
- **G6** — the system score is the minimum over admissible bipartitions.

Two candidate *semantics* were distinguished in advance of any audit, so that the audit could
not be read as picking a semantics after seeing the answer:

- **G-WEAK** — a cut is non-zero as soon as *at least one* direction carries causal influence.
- **G-STRONG** — a cut must be zero whenever *either* direction is zero: `F(x,0) = F(0,y) = 0`.

The exact finite audit (`gamma_delta_axiom_audit_v0_1.py`) tests three pre-declared
aggregators — arithmetic mean, geometric mean, min — against G0–G6:

| `F` | `F(1,0)` | `F(.5,.5)` | `F(1,1)` |
|---|---|---|---|
| `F_arith(x,y) = (x+y)/2` | 0.5 | 0.5 | 1.0 |
| `F_geom(x,y) = √(xy)` | 0.0 | 0.5 | 1.0 |
| `F_min(x,y) = min(x,y)` | 0.0 | 0.5 | 1.0 |

All three satisfy G0–G6. The current arithmetic mean realizes **G-WEAK** (`F_arith(1,0) = 1/2
≠ 0`) and exactly fails **G-STRONG**. So the minimal axioms alone do not characterize any
single aggregator, let alone the arithmetic mean.

**Witness G-E** (same Γ, different directional balance): the two directional profiles `(1,0)`
and `(1/2,1/2)` give the identical arithmetic-mean score `Γ_cut = 1/2`, but their directional
balance

\[
\beta = \frac{2\min(x,y)}{x+y}
\]

differs completely: `β=0` for `(1,0)` (fully one-directional) versus `β=1` for `(1/2,1/2)`
(perfectly balanced). The scalar Γ does not carry directional-balance information — this is
the reason β is tracked as a separate reported quantity throughout the rest of the campaign.

**Verdict:** `GAMMA: AX-NONUNIQUE + SEMANTIC-BRANCH(G-WEAK)`.

## Round 2 — adding marginal independence (characterization audit v0.2): unique

The v0.2 characterization audit (`gamma_delta_characterization_audit_v0_2.py`) asks the next
question: is there a *single additional, explicitly declared* axiom that forces the arithmetic
mean uniquely, without smuggling in the answer? It proposes **marginal independence of the two
directional contributions**:

\[
F(x_2,y_1) - F(x_1,y_1) = F(x_2,y_2) - F(x_1,y_2)
\quad\text{for all } x_1,x_2,y_1,y_2 .
\]

Combined with nullity `F(0,0)=0`, symmetry, and diagonal calibration `F(t,t)=t`, marginal
independence characterizes exactly:

\[
\boxed{F(x,y) = \frac{x+y}{2}.}
\]

The exact finite-grid audit (grid `{0, 1/4, 1/2, 3/4, 1}`, exact `Fraction` arithmetic)
confirms:

- **arithmetic mean satisfies marginal independence** (`arith G4: True`, no counter-witness);
- **geometric mean violates it** — counter-witness `(x1,x2,y1,y2) = (0, 1/4, 0, 1/4)`, giving
  `F(1/4,0) − F(0,0) = 0.0` but `F(1/4,1/4) − F(0,1/4) = 0.25`;
- **min violates it** — counter-witness `(x1,x2,y1,y2) = (0, 1/4, 0, 1/4)`, giving
  `F(1/4,0) − F(0,0) = 0` but `F(1/4,1/4) − F(0,1/4) = 1/4`;
- the finite-grid reconstruction `x/2 + y/2` matches `F_arith(x,y)` exactly at every grid
  point.

Marginal independence is a **substantive** hypothesis, not a consequence of causal invariance:
it explicitly rules out synergy or redundancy between the two directions at the aggregator
level. The current arithmetic-mean formula stops being "merely conventional" **if and only
if** the manuscript adopts this axiom explicitly.

**Verdict:** with marginal independence declared as an axiom, `F(x,y) = (x+y)/2` is the unique
admissible aggregator under G0/G1/G4(diagonal calibration)/marginal-independence; without it,
the weaker G0–G6 set leaves the choice open (Round 1 above).

## Scripts

- `scripts/gamma_delta_axiom_audit_v0_1.py` — Round 1 (G0–G6 non-uniqueness, G-E witness).
- `scripts/gamma_delta_characterization_audit_v0_2.py` — Round 2 (marginal-independence
  uniqueness theorem and counter-witnesses).
