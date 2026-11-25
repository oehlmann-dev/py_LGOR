"""Exact integer linear algebra.

Python integers never overflow, so the kernels here are always exact.  This
replaces Matrix::FindKernel from the C++ code, which did fraction-free Gaussian
elimination in 32-bit ints.  There, Matrix::DivideByRowGCD started its gcd at 1
(so it never divided anything) and the entries could grow until they overflowed.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import List, Sequence


def _row_gcd_normalize(row: List[int]) -> List[int]:
    g = 0
    for x in row:
        if x:
            g = gcd(g, x)
            if g == 1:
                return row
    if g > 1:
        return [x // g for x in row]
    return row


def primitive(vec: Sequence[int]) -> List[int]:
    """Divide an integer vector by the gcd of its entries."""
    g = 0
    for x in vec:
        g = gcd(g, x)
    if g > 1:
        return [x // g for x in vec]
    return list(vec)


def kernel(rows: Sequence[Sequence[int]], ncols: int) -> List[List[int]]:
    """Basis of the right null space {x : A x = 0} of an integer matrix.

    ``rows`` is the matrix as a list of rows, each of length ``ncols``.  The basis
    is the standard one read off from the reduced row echelon form (one vector per
    free column, with a positive entry in that column and zeros in the other free
    columns), with every vector made primitive.  The C++ code read off the same
    basis, so its states agree with these up to an overall factor.
    """
    if ncols == 0:
        return []
    A = [_row_gcd_normalize(list(r)) for r in rows if any(r)]
    nrows = len(A)
    pivot_cols: List[int] = []
    r = 0
    for c in range(ncols):
        if r == nrows:
            break
        p = r
        while p < nrows and A[p][c] == 0:
            p += 1
        if p == nrows:
            continue
        A[r], A[p] = A[p], A[r]
        prow = A[r]
        pv = prow[c]
        for k in range(nrows):
            if k == r:
                continue
            row = A[k]
            x = row[c]
            if x == 0:
                continue
            g = gcd(pv, x)
            a, b = pv // g, x // g
            A[k] = _row_gcd_normalize([a * u - b * v for u, v in zip(row, prow)])
        pivot_cols.append(c)
        r += 1
    pivot_set = set(pivot_cols)
    basis = []
    for f in range(ncols):
        if f in pivot_set:
            continue
        vec = [Fraction(0)] * ncols
        vec[f] = Fraction(1)
        for row_idx, pc in enumerate(pivot_cols):
            num = A[row_idx][f]
            if num:
                vec[pc] = Fraction(-num, A[row_idx][pc])
        den = 1
        for v in vec:
            if v.denominator != 1:
                den = den * v.denominator // gcd(den, v.denominator)
        ivec = primitive([int(v * den) for v in vec])
        basis.append(ivec)
    return basis


def rank(rows: Sequence[Sequence[int]], ncols: int) -> int:
    return ncols - len(kernel(rows, ncols)) if ncols else 0


def matmul(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]], inner: int, ncols_b: int) -> List[List[int]]:
    """A (n x inner) times B (inner x ncols_b)."""
    out = []
    for row in A:
        res = [0] * ncols_b
        for k, a in enumerate(row):
            if a:
                bk = B[k]
                for j in range(ncols_b):
                    if bk[j]:
                        res[j] += a * bk[j]
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# Integer linear systems and lattices (used for the R-symmetry and discrete
# charge corrections, which are linear congruences over the integers).
# ---------------------------------------------------------------------------

def _column_reduce(A: List[List[int]], ncols: int):
    """Column-reduce A with unimodular column operations.

    Returns (H, U, pivots) with H = A U, U unimodular (ncols x ncols, as a list of
    columns), and pivots = {row: column}.  In row r with a pivot p, H[r][c] = 0 for
    every c > p.
    """
    H = [list(r) for r in A]
    U = [[1 if i == j else 0 for i in range(ncols)] for j in range(ncols)]   # U[j] = column j

    def colop(dst, src, q):          # column dst -= q * column src
        for row in H:
            row[dst] -= q * row[src]
        Ud, Us = U[dst], U[src]
        for i in range(ncols):
            Ud[i] -= q * Us[i]

    def swap(a, b):
        if a == b:
            return
        for row in H:
            row[a], row[b] = row[b], row[a]
        U[a], U[b] = U[b], U[a]

    pivots = {}
    col = 0
    for r in range(len(H)):
        if col >= ncols:
            break
        row = H[r]
        while True:
            nz = [c for c in range(col, ncols) if row[c] != 0]
            if not nz:
                break
            cmin = min(nz, key=lambda c: abs(row[c]))
            swap(col, cmin)
            clean = True
            for c in range(col + 1, ncols):
                if row[c]:
                    colop(c, col, row[c] // row[col])
                    if row[c]:
                        clean = False
            if clean:
                break
        if row[col] != 0:
            pivots[r] = col
            col += 1
    return H, U, pivots, col


def solve_integer(A: List[List[int]], b: List[int], ncols: int):
    """All integer solutions of A x = b.

    Returns (x0, kernel_generators) with every solution x0 + sum z_k kernel_k
    (z_k integers), or None if there is no integer solution.
    """
    H, U, pivots, rank_ = _column_reduce(A, ncols)
    y = [0] * ncols
    for r, row in enumerate(H):
        s = b[r] - sum(row[c] * y[c] for c in range(ncols) if y[c])
        if r in pivots:
            p = pivots[r]
            if s % row[p]:
                return None
            y[p] = s // row[p]
        elif s != 0:
            return None
    x0 = [sum(U[c][i] * y[c] for c in range(ncols)) for i in range(ncols)]
    kernel_ = [list(U[c]) for c in range(rank_, ncols)]
    return x0, kernel_


def lattice_basis(generators: List[List[int]], dim: int) -> List[List[int]]:
    """A basis of the integer lattice spanned by (possibly dependent) generators."""
    if not generators:
        return []
    # rows of A = coordinates, columns = generators
    A = [[g[i] for g in generators] for i in range(dim)]
    H, U, pivots, rank_ = _column_reduce(A, len(generators))
    return [[H[i][c] for i in range(dim)] for c in range(rank_)]


def lll(basis: List[List[int]], delta: Fraction = Fraction(3, 4)) -> List[List[int]]:
    """LLL reduction (exact, small dimensions)."""
    B = [list(b) for b in basis]
    n = len(B)
    if n <= 1:
        return B

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    def gso(B):
        Bs, mu = [], [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            v = [Fraction(x) for x in B[i]]
            for j in range(i):
                mu[i][j] = Fraction(dot(B[i], Bs[j])) / dot(Bs[j], Bs[j]) if dot(Bs[j], Bs[j]) else Fraction(0)
                v = [a - mu[i][j] * b for a, b in zip(v, Bs[j])]
            Bs.append(v)
        return Bs, mu

    Bs, mu = gso(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                B[k] = [a - q * b for a, b in zip(B[k], B[j])]
                Bs, mu = gso(B)
        if dot(Bs[k], Bs[k]) >= (delta - mu[k][k - 1] ** 2) * dot(Bs[k - 1], Bs[k - 1]):
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gso(B)
            k = max(k - 1, 1)
    return B


def shortest_in_coset(x0: List[int], generators: List[List[int]]) -> List[int]:
    """A short vector in x0 + L (L spanned by generators): Babai rounding on an
    LLL-reduced basis, followed by a small local search."""
    dim = len(x0)
    basis = lll(lattice_basis(generators, dim))
    if not basis:
        return list(x0)

    def norm(v):
        return sum(a * a for a in v)

    # Babai nearest plane
    v = [Fraction(a) for a in x0]
    Bs = []
    for b in basis:
        w = [Fraction(a) for a in b]
        for u in Bs:
            uu = sum(a * a for a in u)
            if uu:
                c = sum(a * b2 for a, b2 in zip(w, u)) / uu
                w = [a - c * b2 for a, b2 in zip(w, u)]
        Bs.append(w)
    x = list(x0)
    for b, bs in reversed(list(zip(basis, Bs))):
        bb = sum(a * a for a in bs)
        if not bb:
            continue
        c = round(sum(Fraction(a) * b2 for a, b2 in zip(x, bs)) / bb)
        if c:
            x = [a - c * b2 for a, b2 in zip(x, b)]
    # local improvement: try subtracting / adding single basis vectors
    improved = True
    while improved:
        improved = False
        for b in basis:
            for s in (1, -1):
                y = [a - s * b2 for a, b2 in zip(x, b)]
                if norm(y) < norm(x):
                    x, improved = y, True
    return x


def smith(A: List[List[int]], ncols: int):
    """Smith normal form: returns (diag, V) with U A V = diag(d_1, d_2, ...) for some
    unimodular U.  V (ncols x ncols, as a list of columns) is unimodular; diag has
    length ncols (zeros for the kernel directions)."""
    M = [list(r) for r in A]
    m = len(M)
    V = [[1 if i == j else 0 for i in range(ncols)] for j in range(ncols)]   # V[j] = column j

    def colop(dst, src, q):
        for row in M:
            row[dst] -= q * row[src]
        for i in range(ncols):
            V[dst][i] -= q * V[src][i]

    def colswap(a, b):
        for row in M:
            row[a], row[b] = row[b], row[a]
        V[a], V[b] = V[b], V[a]

    t = 0
    while t < min(m, ncols):
        nz = [(abs(M[i][j]), i, j) for i in range(t, m) for j in range(t, ncols) if M[i][j]]
        if not nz:
            break
        _, i, j = min(nz)
        M[t], M[i] = M[i], M[t]
        colswap(t, j)
        while True:
            done = True
            p = M[t][t]
            for j in range(t + 1, ncols):            # clear row t
                if M[t][j]:
                    colop(j, t, M[t][j] // p)
                    if M[t][j]:
                        done = False
            for i in range(t + 1, m):                # clear column t (row ops)
                if M[i][t]:
                    q = M[i][t] // p
                    M[i] = [a - q * b for a, b in zip(M[i], M[t])]
                    if M[i][t]:
                        done = False
            if not done:
                nz = [(abs(M[i][j]), i, j) for i in range(t, m) for j in range(t, ncols)
                      if M[i][j] and (i == t or j == t)]
                _, i, j = min(nz)
                M[t], M[i] = M[i], M[t]
                colswap(t, j)
                continue
            # divisibility condition
            bad = [(i, j) for i in range(t + 1, m) for j in range(t + 1, ncols) if M[i][j] % p]
            if bad:
                i, _ = bad[0]
                M[t] = [a + b for a, b in zip(M[t], M[i])]
                continue
            break
        t += 1
    diag = [abs(M[i][i]) if i < m else 0 for i in range(ncols)]
    for i in range(ncols):
        if i < m and M[i][i] < 0:
            pass  # sign absorbed in U
    return diag, V
