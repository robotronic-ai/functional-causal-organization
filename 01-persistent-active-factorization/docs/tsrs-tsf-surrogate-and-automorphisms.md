# TSRS: a surgery, refuted — and why the semilattice alone cannot select one

## TSRS v0.1 — residual saturation surrogate

IPF fixes a canonical state factor but not a surgery. TSRS v0.1 defines one directly from the
transformation semigroup, without going through IPF's state partition:

Let `η : T → T_SL` be the minimal semilattice congruence, `⊤` its top class, and
`E_⊤ = {e∈T : η(e)=⊤, e²=e}` the idempotent representatives of the top class. Define
`s ≡_R s' ⟺ e(s)=e(s') ∀e∈E_⊤`, then randomize within each `R`-class under `q`:

\[
\widetilde S \sim q(\cdot \mid [s]_R),
\qquad
\widetilde K(y\mid s,u) = \sum_{\tilde s\in[s]_R} q(\tilde s\mid[s]_R)\, K(y\mid \tilde s, u).
\]

**Historical regression — 220/220, no confirmatory value.** On the four deterministic controls
with a pre-existing explicit oracle, TSRS reproduces every one exactly: IR7 96/96, IR10 96/96,
IR11 4/4, IR12 24/24 (`IRP_TSRS_v0_1_result_note.md`; this note's own preface records IPF's
228/228 relabeling equivariance as background, restated in
`docs/af-ts-ipf-factor-recognition.md`). On IR8 it produces a genuinely new partial surgery
(`B_original=1.5 → B_TSRS=1`), with no historical oracle to check it against. TSRS v0.1 was
discovered by inspecting these exact controls, so **this 220/220 has no confirmatory value** — the
result note is explicit that "TSRS is validated" or "TSRS is non-refuted by a preregistered
campaign" is a **forbidden claim** at this stage.

**TSRS-P1 — prospective refutation, 24/40.** A dedicated campaign (`Protocole_TSRS_P1_v0.1
_PROSPECTIVE.md`) freezes three new families *after* TSRS's own candidate hash
(`350e087be5de56bf834f147ec9ef9678b336ddf48119ae99a1be360c3c903288`) is recorded, scored by exact
row-for-row kernel equality (not just scalar `B`):

| Family | Fixtures | Result | What it tests |
|---|---:|---:|---|
| P1-A — cycle + reset active automaton (`Z₃`; `f_0(s)=s+1 mod 3`, `f_1(s)=0`) | 16 | **0/16** | Reset is a genuine irreversible semilattice class with no accumulating memory: `oracle = K` (preserve), but TSRS saturates it to `B_TSRS=0` against `B_oracle≈0.7925` |
| P1-B — `Z₃ × {0,1}` product (active cycle × append-only bit) | 12 | 12/12 | TSRS generalizes correctly from IR10 to a larger active×append-only product |
| P1-C — 3-bit G-set (pure accumulation) | 12 | 12/12 | Full neutralization generalizes from a 2-bit to a 3-bit pure accumulator |
| **Total** | **40** | **24/40** | |

Verdict: **`P1-V3 — TSRS v0.1 refuted prospectively`**
(`TSRS_P1_RESULT_v0_1.md`, candidate SHA-256 as above).

**Why P1-A is decisive.** P1-A's semigroup contains both a non-trivial cyclic subgroup `C_3`
(the reversible succession `f_0`) and an ideal of constant transformations from the reset `f_1`.
`T_SL`'s top class is the reset's irreversible ideal; its idempotent representatives are constant
resets, whose kernel is universal (`S×S`), so the intersection of their kernels is still
universal. TSRS therefore collapses the *entire* state to one residual class and replaces the real
source by `S̃∼q` — under the cyclic context the output becomes uniform and independent of the true
source, and under reset it stays constant, giving `B_TSRS=0` against the true active channel's
`B_original≈0.792481250361`. **The false syllogism this refutes:**
`T_SL ≠ 1 ⟹ persistence`. A reset is an irreversible component of the *action* semigroup without
being an accumulation of memory — `T_SL` alone is too coarse to serve as a persistence detector.
The needed distinction, stated for the next candidate: an inflationary memory action
`m ↦ m∨a` (keeps old content) versus a reset/overwrite action `s ↦ c` (destroys it), formulated
O1-invariantly. No such successor candidate is developed further in this campaign; the chain
moves instead to asking whether the *state factor itself* is even canonical (TSF, below), and then
to the O1 collision search (IR13-S1).

TSRS v0.1 is archived exactly as run: not patched, not re-scored as if it had passed P1.

## TSF / TSF-AUT — the semilattice quotient does not canonically select a state factor

**TSF v0.1** asks a cleaner question than "does `T_SL` define a surgery": does it even select a
*unique* state quotient? A partition `π` of `S` is **TS-compatible** if every transformation
descends to it (TSF-A1) and the induced action factors through `η` (TSF-A2); it is **faithful**
(TSF-A3) if distinct `η`-classes induce distinct maps on the quotient.
`scripts/irp_TSF_action_to_state_factor_audit_v0_1.py` enumerates all faithful, TS-compatible
partitions of IR7, IR8, IR10, IR11, IR12 exhaustively.

On **IR10.01**: **4 faithful TS-compatible partitions** exist — one finest, with 3 classes, and
**three coarsest**, each with 2 classes. The historical persistent bit
`P = {0,2}∣{1,3}` is exactly one of the three coarsest partitions; it is not singled out by
TS-compatibility and faithfulness alone. Verdict: `TSF-MULTI`.

**TSF-AUT v0.1** then asks whether this is a real O1 symmetry obstruction: could an exact
automorphism of the full O1 system map `P` to one of the other coarse partitions?
(`scripts/irp_TSF_IR10_automorphism_audit_v0_1.py`, using `IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json`,
fixture `IR10.01`.) An automorphism here is a pair `(σ,τ)` — a state permutation and a context
permutation — under which the kernel is exactly invariant, `K(σy∣σs,τu)=K(y∣s,u)`.

**Result:** exactly **2 exact O1 automorphisms** of IR10.01. `P`'s orbit under them has size 1 —
it is fixed by both. Verdict: **`AUT-U`**. So the TSF ambiguity is *not* an O1 symmetry that would
make `P` inherently unidentifiable in principle; a deterministic O1-equivariant selector *could*
in principle pick `P` out of the four faithful partitions on this control. IPF's image-profile
construction (`docs/af-ts-ipf-factor-recognition.md`) does exactly this correctly on IR10 — but by
a specific extra rule (image-membership profiling), not one forced by TS-compatibility and
faithfulness alone.

## Scripts

- `scripts/irp_transformation_semigroup_audit_v0_1.py` — `T`, `T_SL` (shared dependency).
- `scripts/irp_image_profile_factor_audit_v0_1.py`, `irp_IPF_exhaustive_relabeling_audit_v0_1.py`
  — IPF construction and its 228/228 equivariance audit.
- `scripts/irp_interface_v0_1_1.py` — frozen candidate interface (dependency of the TSRS
  candidate and harness).
- `scripts/irp_candidate_TSRS_v0_1.py` — the TSRS v0.1 candidate itself.
- `scripts/tsrs_P1_harness_v0_1.py` — TSRS-P1 prospective harness (private oracle not
  redistributed; see the folder `README.md`).
- `scripts/irp_TSF_action_to_state_factor_audit_v0_1.py`,
  `irp_TSF_IR10_automorphism_audit_v0_1.py` — TSF / TSF-AUT audits.
