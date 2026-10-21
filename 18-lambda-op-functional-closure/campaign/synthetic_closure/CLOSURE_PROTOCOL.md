# Synthetic Lambda-op Closure Protocol

## Frozen descriptor

For each deterministic transition law `F`, the descriptor contains:

1. the four horizon-1 states from presentations `0, 1, 2, 3`;
2. the four horizon-2 states;
3. the one-step transition law restricted to the horizon-2 reached set;
4. the operation-conditioned Lambda coordinate for exactly `ID`, `KEEP`, `REVOKE`, and `SUBSTITUTE`.

The held-out functional target is the horizon-3 state vector together with the frozen operator target vector.

## Closure rule

For a descriptor `Phi` and functional target `A`, closure holds on the declared universe when:

`Phi(s1) = Phi(s2) => A(s1) = A(s2)`

A single descriptor fiber containing more than one target value is a functional collision.

The result is classified as:

- `PASS / NONTRIVIAL` when there are zero functional collisions and at least one fiber contains more than one system;
- `PASS / TRIVIAL_INJECTIVE` when there are zero functional collisions and every fiber is a singleton;
- `COLLISION_PRESENT` when at least one fiber contains multiple target values.

## Coverage

The 2-bit campaign exhausts all `4^4 = 256` deterministic maps.

The 3-bit campaign exhausts all `8^8 = 16,777,216` deterministic maps using an exact collision-free integer encoding of the descriptor and target.

The 4-, 6-, and 8-bit campaigns use deterministic transport envelopes. Each envelope contains 512 nontrivial descriptor fibers, with two systems per fiber, for 1,024 systems per width.

## Reproduction

```bash
python synthetic_closure/scripts/run_lambda_op_closure.py
```
