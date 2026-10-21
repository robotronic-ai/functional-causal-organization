"""Small GF(2) utilities used by the protocol verifiers."""

from itertools import product


def bitparity(x: int) -> int:
    return x.bit_count() & 1


def gf2_rank_rows(rows, width=None):
    rows = [int(r) for r in rows if int(r) != 0]
    if not rows:
        return 0
    if width is None:
        width = max(r.bit_length() for r in rows)
    rank = 0
    for col in range(width - 1, -1, -1):
        pivot = next((i for i in range(rank, len(rows)) if (rows[i] >> col) & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and ((rows[i] >> col) & 1):
                rows[i] ^= rows[rank]
        rank += 1
        if rank == len(rows):
            break
    return rank


def add_to_basis(vec, basis, width):
    """Add an integer bit-vector to a row basis if independent."""
    before = gf2_rank_rows(basis, width)
    after = gf2_rank_rows(basis + [vec], width)
    if after > before:
        basis.append(vec)
        return True
    return False


def function_to_bits(values):
    out = 0
    for i, v in enumerate(values):
        out |= (int(v) & 1) << i
    return out


def compose_function_bits(f_bits, F, nstates):
    """Return the truth table bits for f o F."""
    vals = [((f_bits >> F[x]) & 1) for x in range(nstates)]
    return function_to_bits(vals)


def closure_under_koopman(readout_basis, F, nstates):
    """Smallest GF(2)-linear function space containing D and invariant under f -> f o F."""
    basis = []
    for f in readout_basis:
        add_to_basis(f, basis, nstates)
    changed = True
    while changed:
        changed = False
        current = list(basis)
        for f in current:
            uf = compose_function_bits(f, F, nstates)
            if add_to_basis(uf, basis, nstates):
                changed = True
    return basis


def reachable_states(F, sources):
    seen = set(sources)
    frontier = list(sources)
    while frontier:
        x = frontier.pop()
        y = F[x]
        if y not in seen:
            seen.add(y)
            frontier.append(y)
    return sorted(seen)


def eval_signature(x, function_basis):
    return tuple((f >> x) & 1 for f in function_basis)


def rank_binary_matrix(rows):
    if not rows:
        return 0
    width = len(rows[0])
    ints = []
    for row in rows:
        v = 0
        for j, bit in enumerate(row):
            v |= (int(bit) & 1) << j
        ints.append(v)
    return gf2_rank_rows(ints, width)


def mat_vec_mul_rows(rows, vec):
    """Rows are integer bitmasks; vec is integer bitmask."""
    out = 0
    for i, row in enumerate(rows):
        out |= bitparity(row & vec) << i
    return out


def mat_mul_rows(A_rows, B_rows, n):
    """Multiply n x n GF(2) matrices stored as row bitmasks."""
    cols = []
    for j in range(n):
        col = 0
        for i in range(n):
            col |= ((B_rows[i] >> j) & 1) << i
        cols.append(col)
    out = []
    for arow in A_rows:
        row = 0
        for j, col in enumerate(cols):
            row |= bitparity(arow & col) << j
        out.append(row)
    return out


def mat_pow_rows(A_rows, k, n):
    result = [1 << i for i in range(n)]
    base = list(A_rows)
    while k:
        if k & 1:
            result = mat_mul_rows(result, base, n)
        base = mat_mul_rows(base, base, n)
        k >>= 1
    return result


def all_binary_matrices(n):
    for flat in range(1 << (n * n)):
        rows = []
        for i in range(n):
            row = 0
            for j in range(n):
                bit = (flat >> (i * n + j)) & 1
                row |= bit << j
            rows.append(row)
        yield rows


def rows_times_columns(A_rows, columns, n):
    out = []
    for col in columns:
        out.append(mat_vec_mul_rows(A_rows, col))
    return out
