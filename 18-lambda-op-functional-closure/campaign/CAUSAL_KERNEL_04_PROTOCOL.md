# Active Causal Kernel 04 Protocol and Result

## Purpose

This exploratory stage tests whether active causal response provides a stable quotient where passive operation-indexed internal geometry did not.

The primary object is an internal causal-kernel proxy, not a direct estimate of `Gamma_u`, `Delta_u`, or recurrence `R_u`.

## System operations

The system-operation algebra is:

- `ID`
- `KEEP`
- `REVOKE`
- `SUBSTITUTE`

For each operation and each address `H1/H2/H3`, the model is placed in the corresponding control state.

## Experimental interventions

The experimental intervention is distinct from the system operation. Two interventions are applied only to the control-instruction KV suffix at decoder layers 21-23:

- `ZERO_SUFFIX`: zero the selected KV suffix;
- `HEAD_ROTATE`: cyclically rotate KV heads in the selected suffix.

These interventions measure the response of the system state. They are not added to the system-operation algebra.

## Target-blind internal response

For every operation-address cell, the unmodified control state and each experimentally perturbed state are advanced through two fixed probe-token horizons.

The descriptor records relative RMS changes in:

- the final hidden state;
- layers 21-23;
- layers 24-26;
- layers 30-32.

No behavioral verdict is used to construct these response vectors.

## Closure target

The behavioral closure target is the 12-cell outcome vector over `operation x address`.

Behavior is not used to fit the vector quantizers. It is used only during exploratory family selection and collision scoring.

## Candidate families

The preregistered CPU screen compares:

- `K0_SUBSTITUTE_JOINT`;
- `K1_SUBSTITUTE_PER_ADDRESS`;
- `K2_NON_ID_PER_OPERATION`;
- `K3_ALL_PER_OPERATION`.

Each slice is standardized and quantized by deterministic k-means without behavioral targets.

## Useful-candidate rule

A candidate is useful only if:

- functional collision pairs = 0; and
- nontrivial coverage >= 0.20.

This rule prevents a nearly injective representation from being promoted as a useful closure candidate.

## Executed result

The panel executed 32 systems with zero technical skips.

The preregistered screen returned `NO_USEFUL_CANDIDATE`.

Best zero-collision candidate:

- `K1_SUBSTITUTE_PER_ADDRESS_K10`;
- 31 fibers;
- 1 nontrivial fiber;
- 2 systems in nontrivial fibers;
- 6.25% nontrivial coverage;
- 0 collision pairs.

Near-frontier candidate:

- `K3_ALL_PER_OPERATION_K6`;
- 28 fibers;
- 3 nontrivial fibers;
- 7 systems in nontrivial fibers;
- 21.875% nontrivial coverage;
- 1 collision fiber;
- 1 collision pair.

The single collision occurs between units 8 and 24 and differs only in address-specific `SUBSTITUTE` behavior.

## Follow-up

Because the protocol is explicitly exploratory, Hybrid Refinement 05 searches a larger family that retains the operation-indexed active-kernel base while adding address-resolved `SUBSTITUTE` codes. See `CAUSAL_KERNEL_REFINEMENT_05_PROTOCOL.md`.
