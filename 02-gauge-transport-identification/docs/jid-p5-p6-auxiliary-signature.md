# JID-P5 / JID-P6: transport is constructible, conditional on a declared auxiliary signature

## Two abandoned approaches (not shipped in this edition)

Before P5, two attempts to derive the intervention lift `J` from structure already visible in the
transition monoid were tried and abandoned without producing a general selection rule:
`JID-VERT` (vertical-monoid conjugacy) and `JID-CONJ` (vertical equivariance). The source
repository explicitly keeps their protocol files and audit scripts only "for genealogy"; they are
not included here. `docs/partial-identification-policy.md` is the direct response to their
failure — formalize honest reporting instead of continuing to search for a selector among
structures internal to `T`.

## JID-P5: identification via an auxiliary causal signature

`Protocole_JID_P5_v0_1_PROSPECTIVE.md` (frozen before any P5 candidate) tests a different *kind*
of input entirely: for each state, a signature `c:S→C` — the exact summary of an external causal
protocol (a battery of perturbations distinct from the channel used to compute `B`), interpreted
as characterizing the residual actively in a way that could be compared across `P`-fibers. It is
explicitly **not** an arbitrary state label and **not** an output oracle; the candidate never sees
the hidden `(A,P)` decomposition.

### Theorem (`JID_signature_theorem_v0_1.md`)

If, for every fiber `p∈P`, `c|_{π⁻¹(p)}` is injective and all fibers share the identical signature
set `c(π⁻¹(p))=C_*`, then for every `s∈S` and `p∈P` there is a **unique** `J_p(s)∈π⁻¹(p)` with
`c(J_p(s))=c(s)`, and this family satisfies I1–I3 exactly (`π(J_p(s))=p`; `J_{π(s)}(s)=s` since
injectivity within the fiber forces `s` and its own image to coincide; `J_{p'}∘J_p=J_{p'}` since
both compute the unique state of fiber `p'` sharing `s`'s signature). If `c` is not injective
within a fiber but signature sets remain compatible across fibers, several transports preserve
`c`, giving a compatible set `𝒥_c` and, by the FIBER-PARTIAL-ID policy,
`𝓑_c={B_J:J∈𝒥_c}` with no tie-break permitted. If a fiber contains a signature absent from
another, no global signature-preserving transport exists at all: `𝒥_c=∅`, score `NA`. The theorem
does not assert such a signature always exists — it states exactly what an experimental campaign
would need to establish to identify `J` without postulating it: *a causal residual signature,
injective within each fiber and transportable between fibers.*

### The five preregistered families

| Family | Construction | Oracle |
|---|---|---|
| **P5-A** | `S=(A,P)∈{0,1}²`; signature `c(A,P)=A` | `PI-POINT`, hidden transport = constant `A` |
| **P5-B** | `A∈{0,1,2,3}`, `P∈{0,1}`; many-to-one, `P`-dependent active dynamics; `c(A,P)=A` | `PI-POINT` — rules out reduction to a merely bijective dynamics |
| **P5-C** | Same state space; `c(A,P)=A mod 2` — two residuals per fiber collide under `c` | `PI-PARTIAL` — must return the full compatible set / exact `𝓑`, no tie-break |
| **P5-D** | Fibers of equal cardinality but different signature multiplicity (e.g. `c(π⁻¹(0))={x,x,y,y}`, `c(π⁻¹(1))={x,y,y,y}`) | `PI-EMPTY` — no bijective transport preserves `c` |
| **P5-E** | `c(A,P)=(A,r(A))` for deterministic `r` — a redundant but still separating signature | must induce the *same* `J` as P5-B (candidate depends on the equivalence `c` induces, not its encoding) |

Each family is tested under several deterministic state recodings and a bijective recoding of the
signature alphabet; oracles are transported with the same bijections; scoring requires exact
agreement on status, the compatible-lift count, the exact `B`-set (rounded only for display), and
the unique kernel when point-identified.

### Result: 10/10

`JID_P5_RESULT_v0_1.md`, candidate `J-ID-SIGNATURE v0.1`
(SHA-256 `c3cb1d5fe391a5d3a38354531ecc90d509869385d8293d156bd31a8ebc5dd6e7`):

| Family | Result |
|---|---:|
| P5-A | 2/2 |
| P5-B | 2/2 |
| P5-C | 2/2 |
| P5-D | 2/2 |
| P5-E | 2/2 |
| **Total** | **10/10** |

Verdict: **`P5-V4 — finite non-refutation`**. The explicit interpretation given in the result
note: this validates only the *conditional* algorithm — "if an auxiliary causal signature `c:S→C`
is supplied, states in different fibers sharing a signature can be used to construct the lifts
`J_p`." It does **not** show that such a signature exists, or is identifiable, in a real system.
The open question carried forward is experimental, not combinatorial: *which family of auxiliary
interventions produces a causal signature of the residual?*

## JID-P6: finite extension

`jid_P6_finite_harness_v0_1.py` and `JID_P6_FINITE_FIXTURES_v0_1.json` extend the same
signature-identification test to six families, `P6A`–`P6F` (2 fixtures each, 12 total),
mirroring the P5-A…E pattern (point-identified, collision, incompatible-signature, and
redundant-signature cases) at a larger or structurally varied scale. **No separate prose result
note for P6 survives in the source repository** — only the harness and its public fixtures do;
the private oracle is not redistributed in this edition (see the folder `README.md`). Based on the
root repository's own summary table (`"JID-P5 / JID-P6 ... Finite non-refutation, conditional on a
declared auxiliary intervention algebra"`), P6 is understood to have reached a finite
non-refutation verdict, but the exact per-family pass breakdown is not independently documented
here and should not be treated as verified beyond that source statement.

## Scripts

- `scripts/jid_P5_harness_v0_1.py`, `scripts/jid_candidate_signature_v0_1.py` — P5 prospective
  harness and candidate (private oracle not redistributed).
- `scripts/jid_P6_finite_harness_v0_1.py` — P6 harness (private oracle not redistributed; public
  fixtures included for inspection).
