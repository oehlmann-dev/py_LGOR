"""Finite abelian groups as integer lattices (exact, pure Python).

A finite abelian group is Z^n / P for a full-rank period lattice P; a subgroup is a lattice L
with P <= L <= Z^n.  Lattices are stored by generating rows and compared through their
Hermite normal form (HNF), which is unique.  Used by lgor.defects for simple-current
modular invariants and for subgroups of the phase-symmetry group.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd
from typing import Iterable, List, Optional, Sequence, Tuple

Vec = List[int]


def _xgcd(a: int, b: int) -> Tuple[int, int, int]:
    """(g, x, y) with a x + b y = g = gcd(a, b) >= 0."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a - (a // b) * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        a, x0, y0 = -a, -x0, -y0
    return a, x0, y0


def hnf(rows: Iterable[Sequence[int]], ncols: int) -> List[Vec]:
    """Row Hermite normal form: upper-triangular basis of the lattice spanned by ``rows``,
    positive pivots, entries above each pivot reduced to [0, pivot).  Zero rows dropped."""
    A = [list(r) for r in rows if any(r)]
    out: List[Vec] = []
    col = 0
    while A and col < ncols:
        nz = [r for r in A if r[col] != 0]
        if not nz:
            col += 1
            continue
        rest = [r for r in A if r[col] == 0]
        piv = nz[0]
        for r in nz[1:]:
            g, x, y = _xgcd(piv[col], r[col])
            a, b = piv[col] // g, r[col] // g
            new_piv = [x * p + y * q for p, q in zip(piv, r)]
            new_r = [b * p - a * q for p, q in zip(piv, r)]       # column entry 0
            piv = new_piv
            if any(new_r):
                rest.append(new_r)
        if piv[col] < 0:
            piv = [-x for x in piv]
        out.append(piv)
        A = rest
        col += 1
    # reduce above pivots
    for i in range(len(out)):
        p = next(c for c in range(ncols) if out[i][c])
        for j in range(i):
            q = out[j][p] // out[i][p]
            if q:
                out[j] = [a - q * b for a, b in zip(out[j], out[i])]
    return out


def lattice_key(rows, ncols) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(r) for r in hnf(rows, ncols))


def det_full(h: List[Vec], ncols: int) -> int:
    """Index of a full-rank lattice in Z^n from its HNF."""
    if len(h) != ncols:
        raise ValueError("lattice is not of full rank")
    d = 1
    for i, r in enumerate(h):
        d *= r[i]
    return abs(d)


def contains(h: List[Vec], v: Sequence[int], ncols: int) -> bool:
    """Is v in the lattice with HNF basis h?"""
    v = list(v)
    for r in h:
        p = next(c for c in range(ncols) if r[c])
        if v[p] % r[p]:
            return False
        q = v[p] // r[p]
        if q:
            v = [a - q * b for a, b in zip(v, r)]
    return not any(v)


def int_kernel(A: List[Sequence[int]], ncols: int) -> List[Vec]:
    """Z-basis of {x in Z^ncols : A x = 0} (column operations with a unimodular transform)."""
    M = [list(r) for r in A]
    U = [[1 if i == j else 0 for j in range(ncols)] for i in range(ncols)]   # columns of U: U[.][j]
    cols_done = 0
    for row in range(len(M)):
        # gcd-reduce entries of this row over columns cols_done..ncols-1 into column cols_done
        for c in range(cols_done + 1, ncols):
            a, b = M[row][cols_done], M[row][c]
            if b == 0:
                continue
            g, x, y = _xgcd(a, b)
            p, q = a // g, b // g
            # new col_done = x*col_done + y*c ; new c = -q*col_done + p*c  (det 1)
            for R in M:
                u, w = R[cols_done], R[c]
                R[cols_done], R[c] = x * u + y * w, -q * u + p * w
            for R in U:
                u, w = R[cols_done], R[c]
                R[cols_done], R[c] = x * u + y * w, -q * u + p * w
        if cols_done < ncols and M[row][cols_done] != 0:
            cols_done += 1
    return [[U[i][j] for i in range(ncols)] for j in range(cols_done, ncols)]


def kernel_mod1(rows: List[Sequence[Fraction]], ncols: int) -> List[Vec]:
    """Z-basis of {x in Z^ncols : r.x in Z for every rational row r}."""
    if not rows:
        return [[1 if i == j else 0 for j in range(ncols)] for i in range(ncols)]
    D = 1
    for r in rows:
        for x in r:
            D = D * Fraction(x).denominator // gcd(D, Fraction(x).denominator)
    g = len(rows)
    A = []
    for t, r in enumerate(rows):
        A.append([int(Fraction(x) * D) for x in r] + [-D if s == t else 0 for s in range(g)])
    K = int_kernel(A, ncols + g)
    return hnf([k[:ncols] for k in K], ncols)


def smith_basis(gens: List[Vec], periods: List[Vec], n: int) -> Tuple[List[Vec], List[int]]:
    """Decompose the finite group L/P (L spanned by gens + periods, P by periods) as a direct sum
    of cyclic groups.  Returns (basis elements in Z^n, orders), orders > 1."""
    L = hnf(gens + periods, n)
    P = hnf(periods, n)
    r = len(L)
    # coordinates of P's basis in the basis L: solve P_i = sum c_ij L_j (L upper triangular)
    coords = []
    for v in P:
        v = list(v)
        c = []
        for row in L:
            p = next(cc for cc in range(n) if row[cc])
            q = v[p] // row[p]
            c.append(q)
            v = [a - q * b for a, b in zip(v, row)]
        if any(v):
            raise ValueError("periods not contained in the lattice")
        coords.append(c)
    # Smith form of coords (rows = relations among L basis elements)
    M = [list(x) for x in coords]
    B = [list(x) for x in L]          # current basis of L (rows); relations M (rows) w.r.t. B
    m = len(M)
    diag = []
    t = 0
    while t < min(m, r):
        # find nonzero pivot with minimal |entry|
        best = None
        for i in range(t, m):
            for j in range(t, r):
                if M[i][j] and (best is None or abs(M[i][j]) < abs(M[best[0]][best[1]])):
                    best = (i, j)
        if best is None:
            break
        i, j = best
        M[t], M[i] = M[i], M[t]
        for R in M:
            R[t], R[j] = R[j], R[t]
        B[t], B[j] = B[j], B[t]
        done = False
        while not done:
            done = True
            for i in range(t + 1, m):                       # row reduction
                q = M[i][t] // M[t][t]
                if q:
                    M[i] = [a - q * b for a, b in zip(M[i], M[t])]
                if M[i][t]:
                    done = False
            for j in range(t + 1, r):                       # column reduction (changes basis B)
                q = M[t][j] // M[t][t]
                if q:
                    for R in M:
                        R[j] -= q * R[t]
                    B[t] = [a + q * b for a, b in zip(B[t], B[j])]
                if M[t][j]:
                    done = False
            if not done:
                best = None
                for i in range(t, m):
                    for j in range(t, r):
                        if M[i][j] and (best is None or abs(M[i][j]) < abs(M[best[0]][best[1]])):
                            best = (i, j)
                i, j = best
                M[t], M[i] = M[i], M[t]
                for R in M:
                    R[t], R[j] = R[j], R[t]
                B[t], B[j] = B[j], B[t]
                continue
            # divisibility condition
            for i in range(t + 1, m):
                for j in range(t + 1, r):
                    if M[i][j] % M[t][t]:
                        M[t] = [a + b for a, b in zip(M[t], M[i])]
                        done = False
                        break
                if not done:
                    break
        diag.append(abs(M[t][t]))
        t += 1
    basis, orders = [], []
    for k, d in enumerate(diag):
        if d > 1:
            basis.append(B[k])
            orders.append(d)
    return basis, orders


def subgroups(orders: Sequence[int], cap: int = 20000):
    """All subgroups of Z_{d_1} x ... x Z_{d_t} as HNF row lists (rows = generators), smallest first.
    Returns (list, complete flag)."""
    t = len(orders)
    out = []
    complete = True

    def divisors(d):
        return [x for x in range(1, d + 1) if d % x == 0]

    def rec(i, H):
        nonlocal complete
        if len(out) >= cap:
            complete = False
            return
        if i < 0:
            rows = [r for r in H if any(r)]
            Hn = hnf(rows + [[d if c == j else 0 for c in range(t)] for j, d in enumerate(orders)], t)
            if Hn == [list(r) for r in H]:
                # the lattice spanned by H contains the periods: a genuine subgroup
                out.append(Hn)
            return
        for h in divisors(orders[i]):
            row = [0] * t
            row[i] = h
            ranges = [range(H[j][j]) if j > i else [0] for j in range(t)]
            for vals in product(*[ranges[j] for j in range(i + 1, t)]):
                r = list(row)
                for j, v in zip(range(i + 1, t), vals):
                    r[j] = v
                H2 = list(H)
                H2[i] = r
                rec(i - 1, H2)

    rec(t - 1, [[0] * t for _ in range(t)])
    # de-duplicate (different raw choices can give the same HNF)
    seen, uniq = set(), []
    for H in out:
        k = tuple(map(tuple, H))
        if k not in seen:
            seen.add(k)
            uniq.append(H)
    order = lambda H: 1 if not t else _prod(orders) // _prod(H[k][k] for k in range(t))
    uniq.sort(key=order)
    return uniq, complete


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def subgroup_order(H: List[Vec], orders: Sequence[int]) -> int:
    return _prod(orders) // _prod(H[k][k] for k in range(len(orders)))
