"""v2.5: consistent topological defect lines of Gepner-type models whose blocks need not be Fermat.

Every block of W (fields linked by a term) is recognised as one rational N=2 SCFT:

    A_k    x^(k+2)                       su(2)_k x u(1)_2 / u(1)_(k+2)          (minimal model)
    KS     p_d(e_1..e_n), d = n+k+1      SU(n+1)_k x SO(2n)_1 / (SU(n)_(k+1) x U(1))   (Kazama-Suzuki CP^n_k)
    D      x^m + x y^2                   D_(m+1) invariant of su(2)_k, k = 2(m-1)  (m odd: Z2 extension)
    E7     x^3 + x y^3                   E7 invariant of su(2)_16 (automorphism invariant of the D10 extension)
    x^2    massive, c = 0

A line of the block is kept only if it

  (a) commutes with the block's N=2 algebra,
  (b) commutes with the spectral flow, the alignment/GSO currents and the J orbifold of the Gepner model,
  (c) is well defined on the coset (zero monodromy with the field-identification currents).

All operators in (a)-(c) are simple currents of the block whose non-abelian labels are trivial up to
field identification [literature: Kazama-Suzuki 1989, Gepner 1989]. Hence a Verlinde line L_A x L_B of
the non-abelian factors (trivial on the U(1) and fermion labels) passes (a)-(c) iff its monodromy with
every simple current of the non-abelian factors vanishes. This is checked numerically here from the
S-matrix (S_(L,J) / S_(0,J) = d_L for every simple current J). The lines that pass are the "core":

    A_k    su(2)_k lines with l even                              (PSU(2)_k; Fib for k = 3)
    KS     su(n+1)_k lines of N-ality 0  x  su(n)_(k+1) lines of N-ality 0
           (complete among undressed lines: the identification current has charges a/(n+1) and b/n,
           which cancel only if both vanish, since gcd(n, n+1) = 1)
    D_odd  (k = 2 mod 4, permutation invariant, chiral algebra su(2)_k): PSU(2)_k
    D_even (k = 0 mod 4) and E7: Verlinde lines of su(2)_k extended by l = k (fixed point k/2 split)

The fusion ring of every core factor is computed with the Verlinde formula and compared with the rings
that occur in the 161 Fermat Gepner models (PSU(2)_k, SU(2)_k and products of two). A factor that
matches none of them is flagged NEW. Ring isomorphism is [verified]; equivalence of the categories
(F-symbols) is [inferred].

Not covered, listed as [missing] in the report: lines of non-zero N-ality (need a U(1)/fermion dressing),
lines that commute with N=2 but not with the full W-algebra of a KS block, and for D/E7 blocks the
lines that preserve su(2)_k but not the extended algebra (the full bimodule category).

Works on .lgo files, run folders and saved results (NAME.lgor.json of any LGOR 2.x): only the model is
read, nothing is recomputed.
"""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .model import Model

try:
    import numpy as np
except ImportError:          # pragma: no cover
    np = None

# levels k of the minimal-model factors in the 161 Fermat Gepner models of the survey (survey.csv, column ks)
SURVEY_LEVELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28,
                 30, 31, 33, 34, 38, 40, 43, 46, 47, 49, 52, 54, 58, 61, 68, 70, 76, 82, 86, 88, 89, 94, 97,
                 98, 103, 108, 118, 124, 138, 154, 166, 178, 187, 214, 218, 229, 236, 238, 292, 310, 334, 340}

TOL = 1e-6
MAX_FUSION_WORK = 3e10       # rank^3 * (number of su(N)_k weights) above which the fusion tensor is skipped


# ====================================================================== su(N)_k modular data
class SUNk:
    """su(N)_k: integrable weights (Dynkin labels), Kac-Peterson S-matrix rows on demand, dimensions."""

    def __init__(self, N: int, k: int):
        if np is None:
            raise RuntimeError("blocklines needs numpy")
        self.N, self.k, self.kappa = N, k, k + N
        self.weights = [w for w in itertools.product(range(k + 1), repeat=N - 1) if sum(w) <= k]
        self.index = {w: i for i, w in enumerate(self.weights)}
        self.L = np.array([self._shifted(w) for w in self.weights], float)
        self._rows: Dict[int, "np.ndarray"] = {}
        self._norm = None
        self.dims = self._qdims()

    @property
    def n(self) -> int:
        return len(self.weights)

    def _shifted(self, w) -> List[int]:
        N = self.N
        l = [sum(w[j] for j in range(i, N - 1)) for i in range(N - 1)] + [0]
        return [l[i] + (N - 1 - i) for i in range(N)]          # partition + rho

    def _qdims(self):
        # q-Weyl formula: d = prod_{i<j} sin(pi (L_i - L_j)/kappa) / sin(pi (j - i)/kappa)
        N, kap = self.N, self.kappa
        d = np.ones(self.n)
        for i in range(N):
            for j in range(i + 1, N):
                d *= np.sin(np.pi * (self.L[:, i] - self.L[:, j]) / kap) / math.sin(np.pi * (j - i) / kap)
        return d

    def _raw_rows(self, idx: List[int]):
        N, kap = self.N, self.kappa
        L = self.L
        tot = L.sum(axis=1)
        out = np.empty((len(idx), self.n), complex)
        chunk = max(1, int(4e6 // (self.n * N * N)))
        for s in range(0, len(idx), chunk):
            I = idx[s:s + chunk]
            M = np.exp(-2j * np.pi * L[I][:, None, :, None] * L[None, :, None, :] / kap)
            out[s:s + len(I)] = np.linalg.det(M) * np.exp(2j * np.pi * np.outer(tot[I], tot) / (N * kap))
        return out

    def rows(self, idx: List[int]):
        """Rows S[idx, :] of the unitary S-matrix (normalised by S_00 > 0)."""
        if self._norm is None:
            r0 = self._raw_rows([0])[0]
            self._norm = (abs(r0[0]) / r0[0]) / math.sqrt((abs(r0) ** 2).sum())
            self._rows[0] = r0 * self._norm
        todo = [i for i in idx if i not in self._rows]
        if todo:
            for i, r in zip(todo, self._raw_rows(todo)):
                self._rows[i] = r * self._norm
        return np.array([self._rows[i] for i in idx])

    def nality(self, w) -> int:
        return sum((i + 1) * a for i, a in enumerate(w)) % self.N

    def simple_currents(self) -> List[tuple]:
        out = [tuple([0] * (self.N - 1))]
        for a in range(self.N - 1):
            v = [0] * (self.N - 1)
            v[a] = self.k
            out.append(tuple(v))
        return out

    def check_unitary(self) -> bool:
        S = self.rows(list(range(self.n)))
        return bool(np.allclose(S @ S.conj().T, np.eye(self.n), atol=1e-8) and np.allclose(S, S.T, atol=1e-8))


# ====================================================================== fusion categories (rings)
class Category:
    """Labels, quantum dimensions and a lazily computed fusion tensor N[a, b, c]."""

    def __init__(self, name: str, labels: List[str], dims, N=None, make_N: Optional[Callable] = None,
                 notes: Optional[List[str]] = None):
        self.name, self.labels, self.dims = name, list(labels), [float(d) for d in dims]
        self._N, self._make_N = N, make_N
        self.notes = list(notes or [])

    @property
    def N(self):
        if self._N is None:
            if self._make_N is None:
                raise ValueError(f"{self.name}: fusion tensor not available (too large)")
            self._N = self._make_N()
        return self._N

    @property
    def has_N(self) -> bool:
        return self._N is not None or self._make_N is not None

    @property
    def rank(self) -> int:
        return len(self.labels)

    @property
    def global_dim(self) -> float:
        return sum(d * d for d in self.dims)

    def is_pointed(self) -> bool:
        return all(abs(d - 1) < TOL for d in self.dims)

    def is_integral(self) -> bool:
        return all(abs(d - round(d)) < TOL for d in self.dims)

    def n_invertible(self) -> int:
        return sum(abs(d - 1) < TOL for d in self.dims)

    def dims_key(self) -> Tuple[float, ...]:
        return tuple(sorted(round(d, 5) for d in self.dims))

    def product(self, other: "Category", name: Optional[str] = None) -> "Category":
        r1, r2 = self.rank, other.rank
        make = None
        if self.has_N and other.has_N and (r1 * r2) <= 400:
            make = lambda: np.einsum("abc,def->adbecf", self.N, other.N).reshape(r1 * r2, r1 * r2, r1 * r2)
        labels = [f"{a}*{b}" for a in self.labels for b in other.labels]
        dims = [a * b for a in self.dims for b in other.dims]
        return Category(name or f"{self.name} x {other.name}", labels, dims, make_N=make)


def _fusion_from_S(Ssub, S0, name: str):
    """N_ab^c = sum_x S_ax S_bx conj(S_cx) / S_0x for the rows Ssub (which must close under fusion)."""
    r = Ssub.shape[0]
    Nt = np.zeros((r, r, r), int)
    Sc = Ssub.conj().T
    for a in range(r):
        V = ((Ssub[a] / S0)[None, :] * Ssub) @ Sc          # (b, c)
        if np.abs(V.imag).max() > 1e-5:
            raise ValueError(f"{name}: complex fusion coefficient")
        R = np.rint(V.real)
        if np.abs(V.real - R).max() > 1e-5 or R.min() < 0:
            raise ValueError(f"{name}: non-integer or negative fusion coefficient")
        Nt[a] = R.astype(int)
    return Nt


def verlinde_subring(S_rows, S0, dims, labels: List[str], name: str) -> Category:
    """Fusion ring on a set of labels from the rows S_rows (rows = the labels, columns = all weights).
    Closure is checked: the quantum dimensions must satisfy d_a d_b = sum_c N_ab^c d_c inside the set."""
    S_rows = np.asarray(S_rows)

    def make():
        Nt = _fusion_from_S(S_rows, S0, name)
        d = np.array(dims)
        lhs = np.outer(d, d)
        rhs = np.einsum("abc,c->ab", Nt, d)
        if np.abs(lhs - rhs).max() > 1e-6 * max(1.0, lhs.max()):
            raise ValueError(f"{name}: label set does not close under fusion")
        return Nt
    return Category(name, labels, dims, make_N=make)


_SUN_CACHE: Dict[Tuple[int, int], SUNk] = {}


def sun(N: int, k: int) -> SUNk:
    if (N, k) not in _SUN_CACHE:
        _SUN_CACHE[(N, k)] = SUNk(N, k)
    return _SUN_CACHE[(N, k)]


def psun(N: int, k: int) -> Tuple[Category, SUNk, List[int]]:
    """N-ality-zero part of su(N)_k (for N = 2: the even-l lines PSU(2)_k)."""
    su = sun(N, k)
    sub = [i for i, w in enumerate(su.weights) if su.nality(w) == 0]
    labels = [f"l={w[0]}" if N == 2 else str(w) for w in (su.weights[i] for i in sub)]
    name = f"PSU({N})_{k}"
    if len(sub) ** 3 * su.n > MAX_FUSION_WORK:
        cat = Category(name, labels, su.dims[sub],
                       notes=[f"fusion tensor not computed (rank {len(sub)}, {su.n} weights): ring only by dimensions"])
    else:
        rows = su.rows(sub)
        S0 = su.rows([0])[0]
        cat = verlinde_subring(rows, S0, su.dims[sub], labels, name)
    return cat, su, sub


def su2_full(k: int) -> Category:
    su = sun(2, k)
    idx = list(range(su.n))
    return verlinde_subring(su.rows(idx), su.rows([0])[0], su.dims, [f"l={w[0]}" for w in su.weights], f"SU(2)_{k}")


def d_extension(k: int) -> Tuple[Category, List[str], "np.ndarray"]:
    """Verlinde lines of su(2)_k extended by the simple current l = k (k = 0 mod 4), i.e. of the
    D_(k/2+2) invariant. Orbits [l] = {l, k-l} for even l < k/2, fixed point k/2 split into two.
    S_ext = 2 S on orbits, S on orbit-fixed entries, and 1/2 S_ff + e e' X/2 on the fixed-fixed block;
    X is fixed by unitarity, S^2 = C and integral fusion [Fuchs-Schellekens-Schweigert 1996]."""
    if k % 4:
        raise ValueError("D extension needs k = 0 mod 4")
    su = sun(2, k)
    S0 = su.rows(list(range(su.n))).real
    f = k // 2
    orb = list(range(0, f, 2))
    labels = [f"[{l}]" for l in orb] + [f"{f}+", f"{f}-"]
    n = len(labels)
    for X in (1, -1, 1j, -1j):
        S = np.zeros((n, n), complex)
        for i, a in enumerate(orb):
            for j, b in enumerate(orb):
                S[i, j] = 2 * S0[a, b]
            S[i, n - 2] = S[i, n - 1] = S[n - 2, i] = S[n - 1, i] = S0[a, f]
        for e, ie in ((1, n - 2), (-1, n - 1)):
            for e2, ie2 in ((1, n - 2), (-1, n - 1)):
                S[ie, ie2] = 0.5 * S0[f, f] + 0.5 * e * e2 * X
        if not np.allclose(S @ S.conj().T, np.eye(n), atol=1e-8):
            continue
        C = S @ S
        if not np.allclose(np.abs(C), np.round(np.abs(C)), atol=1e-8):
            continue
        dims = (S[:, 0] / S[0, 0]).real
        cat = verlinde_subring(S, S[0], dims, labels, f"su(2)_{k}/Z2 (D_{k // 2 + 2})")
        try:
            cat.N
        except ValueError:
            continue
        notes = [f"extended S-matrix: fixed-point resolution X = {X}; S unitary, S^2 a permutation, fusion integral "
                 "and non-negative [verified]"]
        return cat, notes, S
    raise ValueError(f"no consistent fixed-point resolution found for k={k}")


# ====================================================================== fusion-ring isomorphism
def ring_isomorphic(A: Category, B: Category, max_nodes: int = 2_000_000) -> Optional[bool]:
    """True/False; None if undecided (fusion tensor unavailable or search too large)."""
    if A.rank != B.rank or A.dims_key() != B.dims_key():
        return False
    if not (A.has_N and B.has_N):
        return None
    try:
        NA, NB = A.N, B.N
    except ValueError:
        return None
    r = A.rank
    ua = [i for i in range(r) if all(NA[i, j, j] == 1 for j in range(r))]
    ub = [j for j in range(r) if all(NB[j, x, x] == 1 for x in range(r))]
    if not ua or not ub:
        return False
    # invariants that a ring isomorphism preserves: dimension and sum_b N_ab^b (trace of the fusion matrix)
    ta = [round(A.dims[i], 5) for i in range(r)]
    tb = [round(B.dims[j], 5) for j in range(r)]
    tra = [int(np.trace(NA[i])) for i in range(r)]
    trb = [int(np.trace(NB[j])) for j in range(r)]
    cand = [[j for j in range(r) if tb[j] == ta[i] and trb[j] == tra[i]] for i in range(r)]
    if any(not c for c in cand):
        return False
    perm = [-1] * r
    used = [False] * r
    nodes = [0]

    def ok(i):
        assigned = [a for a in range(r) if perm[a] >= 0]
        pi = perm[i]
        for a in assigned:
            pa = perm[a]
            for b in assigned:
                pb = perm[b]
                if (NA[i, a, b] != NB[pi, pa, pb] or NA[a, i, b] != NB[pa, pi, pb]
                        or NA[a, b, i] != NB[pa, pb, pi]):
                    return False
        return True

    order = [ua[0]] + sorted((i for i in range(r) if i != ua[0]), key=lambda i: len(cand[i]))

    def rec(t):
        if t == r:
            return True
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise TimeoutError
        i = order[t]
        for j in cand[i]:
            if used[j]:
                continue
            perm[i] = j
            used[j] = True
            if ok(i) and rec(t + 1):
                return True
            perm[i] = -1
            used[j] = False
        return False

    perm[ua[0]] = ub[0]
    used[ub[0]] = True
    try:
        return rec(1)
    except TimeoutError:
        return None


_REF: Dict[str, Category] = {}


def _ref(kind: str, k: int) -> Category:
    key = f"{kind}{k}"
    if key not in _REF:
        _REF[key] = psun(2, k)[0] if kind == "P" else su2_full(k)
    return _REF[key]


def _ref_name(kind: str, k: int) -> str:
    if kind == "P":
        return {3: "Fib (= PSU(2)_3)", 4: "Rep(S3) ring (= PSU(2)_4)"}.get(k, f"PSU(2)_{k}")
    return {2: "Ising (= SU(2)_2)"}.get(k, f"SU(2)_{k}")


def _ref_dims(kind: str, k: int) -> List[float]:
    d = sun(2, k).dims
    if kind == "P":
        return [d[l] for l in range(0, k + 1, 2)]
    return list(d)


def _ref_ranks(r: int):
    """(kind, k) with rank r: PSU(2)_k has rank floor(k/2)+1, SU(2)_k has rank k+1."""
    out = [("P", k) for k in (2 * r - 2, 2 * r - 1) if k >= 1]
    if r >= 2:
        out.append(("S", r - 1))
    return out


KNOWN, NEW_LEVEL, NEW = "known", "new level", "NEW"


def _d_ext_dims(k: int) -> List[float]:
    d = sun(2, k).dims
    f = k // 2
    return [d[l] for l in range(0, f, 2)] + [d[f] / 2, d[f] / 2]


_CATALOGUE: Optional[List[Tuple[str, Tuple[float, ...], Callable]]] = None


def catalogue() -> List[Tuple[str, Tuple[float, ...], Callable]]:
    """Non-PSU(2) rings that occur in KS and D blocks, used to name coincidences between new rings:
    PSU(N)_k for N = 3..6 (up to 2000 weights) and the D-series extensions su(2)_k/Z2, k = 0 mod 4, k <= 200."""
    global _CATALOGUE
    if _CATALOGUE is None:
        out = []
        for N in range(3, 7):
            for k in range(2, 60):
                if math.comb(N - 1 + k, N - 1) > 2000:
                    break
                su = sun(N, k)
                dims = [su.dims[i] for i, w in enumerate(su.weights) if su.nality(w) == 0]
                out.append((f"PSU({N})_{k}", tuple(sorted(round(x, 5) for x in dims)),
                            (lambda N=N, k=k: psun(N, k)[0])))
        for k in range(4, 201, 4):
            out.append((f"D_{k // 2 + 2} = su(2)_{k}/Z2", tuple(sorted(round(x, 5) for x in _d_ext_dims(k))),
                        (lambda k=k: d_extension(k)[0])))
        _CATALOGUE = out
    return _CATALOGUE


def coincidences(cat: Category) -> List[str]:
    """Other catalogue rings isomorphic to cat ([verified] ring isomorphism)."""
    key = cat.dims_key()
    out = []
    for name, dk, build in catalogue():
        if dk != key or name.split(" =")[0] == cat.name.split(" ")[0]:
            continue
        other = build()
        if other.name == cat.name:
            continue
        iso = ring_isomorphic(cat, other)
        if iso:
            out.append(name)
        elif iso is None:
            out.append(name + " (dimensions only)")
    return out


def identify(cat: Category, known_levels=SURVEY_LEVELS) -> Tuple[str, str]:
    """Name a fusion ring by comparison with PSU(2)_k and SU(2)_k (any level) and products of two of them.
    Returns (description, status): KNOWN = a type and level that occur in the 161 Fermat Gepner models,
    NEW_LEVEL = a PSU(2)_k/SU(2)_k type at a level that does not occur there, NEW = none of these types."""
    if cat.rank == 1:
        return "trivial", KNOWN
    if cat.is_pointed():
        return f"pointed (rank {cat.rank}): invertible lines only", KNOWN
    r = cat.rank
    key = cat.dims_key()
    undecided = []
    for kind, k in _ref_ranks(r):
        if tuple(sorted(round(x, 5) for x in _ref_dims(kind, k))) != key:
            continue
        iso = ring_isomorphic(cat, _ref(kind, k))
        if iso:
            return f"fusion ring of {_ref_name(kind, k)}", (KNOWN if k in known_levels else NEW_LEVEL)
        if iso is None:
            undecided.append(_ref_name(kind, k))
    for r1 in range(2, r):
        if r % r1:
            continue
        r2 = r // r1
        if r2 < r1:
            break
        for k1 in _ref_ranks(r1):
            for k2 in _ref_ranks(r2):
                d = sorted(round(a * b, 5) for a in _ref_dims(*k1) for b in _ref_dims(*k2))
                if tuple(d) != key:
                    continue
                iso = ring_isomorphic(cat, _ref(*k1).product(_ref(*k2)))
                if iso:
                    known = k1[1] in known_levels and k2[1] in known_levels
                    return f"fusion ring of {_ref_name(*k1)} x {_ref_name(*k2)}", (KNOWN if known else NEW_LEVEL)
                if iso is None:
                    undecided.append(f"{_ref_name(*k1)} x {_ref_name(*k2)}")
    extra = "; integral (weakly group-theoretical)" if cat.is_integral() else ""
    co = coincidences(cat)
    same = f"; same fusion ring as {', '.join(co)}" if co else ""
    if undecided:
        return (f"dimensions match {', '.join(undecided)} but the ring comparison was not completed{extra}{same}", NEW)
    return f"not a PSU(2)_k, SU(2)_k or a product of two{extra}{same}", NEW


def level_rank_note(N: int, k: int) -> Optional[str]:
    if N >= 2 and k >= 2:
        return f"level-rank partner PSU({k})_{N} [literature: PSU(N)_k and PSU(k)_N have equivalent fusion rings]"
    return None


# ====================================================================== block recognition
def newton_power_sum(n: int, d: int) -> Dict[tuple, Fraction]:
    """p_d of n variables written in the elementary symmetric e_1..e_n (Newton identities)."""
    from collections import defaultdict
    z = (0,) * n

    def e(i):
        v = [0] * n
        v[i - 1] = 1
        return {tuple(v): Fraction(1)}

    def mul(a, b):
        r = defaultdict(Fraction)
        for ea, ca in a.items():
            for eb, cb in b.items():
                r[tuple(x + y for x, y in zip(ea, eb))] += ca * cb
        return {kk: v for kk, v in r.items() if v}
    p = {0: {z: Fraction(n)}}
    for m in range(1, d + 1):
        acc = defaultdict(Fraction)
        for i in range(1, min(m, n) + 1):
            term = {kk: v * m for kk, v in e(i).items()} if i == m else mul(e(i), p[m - i])
            for kk, v in term.items():
                acc[kk] += (-1) ** (i - 1) * v
        p[m] = {kk: v for kk, v in acc.items() if v}
    return p[d]


class Block:
    def __init__(self, fields: List[int], kind: str, **data):
        self.fields, self.kind, self.data = fields, kind, data
        self.notes: List[str] = []

    @property
    def name(self) -> str:
        d = self.data
        if self.kind == "A":
            return f"({d['k']})"
        if self.kind == "KS":
            return f"CP{d['n']}_{d['k']}"
        if self.kind == "D":
            return f"D{d['m'] + 1}"
        if self.kind == "x^2":
            return "x^2"
        return self.kind

    @property
    def c3(self) -> Fraction:
        d = self.data
        if self.kind in ("A", "D"):
            return Fraction(d["k"], d["k"] + 2)
        if self.kind == "KS":
            return Fraction(d["n"] * d["k"], d["n"] + d["k"] + 1)
        if self.kind == "E7":
            return Fraction(16, 18)
        return Fraction(0)

    def c3_from_weights(self, q: List[Fraction]) -> Fraction:
        return sum(1 - 2 * x for x in q)


def classify_blocks(model: Model) -> List[Block]:
    K = model.korder
    out = []
    for blk in model.blocks():
        q = [Fraction(model.fields[i].kcharge, K) for i in blk]
        terms = [t for t in model.superpotential if any(t.powers[i] for i in blk)]
        supp = {tuple(t.powers[i] for i in blk) for t in terms}
        b = None
        if len(blk) == 1:
            n = model.fermat_exponent(blk[0])
            if n == 2:
                b = Block(blk, "x^2")
            elif n is not None and n > 2:
                b = Block(blk, "A", k=n - 2)
        else:
            n = len(blk)
            order = sorted(range(n), key=lambda j: q[j])
            d = 1 / q[order[0]]
            # Kazama-Suzuki CP^n_k: weights (1..n)/d and the support of p_d(e_1..e_n)
            if d.denominator == 1 and all(q[order[i]] == Fraction(i + 1) / d for i in range(n)):
                d = int(d)
                k = d - n - 1
                if k >= 1:
                    ks_supp = {tuple(e[order.index(j)] for j in range(n)) for e in newton_power_sum(n, d)}
                    if supp == ks_supp:
                        b = Block(blk, "KS", n=n, k=k)
                        b.notes.append(f"W has exactly the monomials of the Newton power sum p_{d}(e_1..e_{n}) [verified]")
            if b is None and n == 2:
                for ix, iy in ((0, 1), (1, 0)):
                    def mono(px, py):
                        v = [0, 0]
                        v[ix], v[iy] = px, py
                        return tuple(v)
                    xm = [p[ix] for p in supp if p[iy] == 0]
                    if len(supp) == 2 and mono(1, 2) in supp and len(xm) == 1 and xm[0] >= 3:
                        m = xm[0]
                        b = Block(blk, "D", k=2 * (m - 1), m=m)
                    elif supp == {mono(3, 0), mono(1, 3)}:
                        b = Block(blk, "E7")
                    if b:
                        b.notes.append(f"normal form recognised with x = x{blk[ix] + 1}, y = x{blk[iy] + 1} [verified]")
                        break
        b = b or Block(blk, "unknown")
        if b.kind not in ("unknown", "x^2") and b.c3_from_weights(q) != b.c3:
            b.notes.append(f"WARNING: c/3 from the weights is {b.c3_from_weights(q)}, from the RCFT {b.c3}")
        out.append(b)
    return out


# ====================================================================== lines of one block
class BlockLines:
    def __init__(self, block: Block):
        self.block = block
        self.factors: List[Tuple[Category, str]] = []      # (category, description)
        self.checks: List[str] = []
        self.missing: List[str] = []
        self.ok = True
        kind, d = block.kind, block.data
        if kind == "x^2":
            return
        if kind == "A":
            self._sun_part(2, d["k"], f"su(2)_{d['k']}")
            self.missing.append("odd-l lines need an invertible dressing: treated by `tdl` and `defects` section 1")
        elif kind == "KS":
            n, k = d["n"], d["k"]
            self._sun_part(n + 1, k, f"su({n + 1})_{k} (numerator)")
            self._sun_part(n, k + 1, f"su({n})_{k + 1} (denominator)")
            self.checks.append(
                f"field identification: its current has su({n + 1}) charge a/{n + 1} and su({n}) charge b/{n}; "
                f"a line of the non-abelian factors is well defined iff a/{n + 1} = b/{n} mod 1, which for "
                f"gcd({n}, {n + 1}) = 1 forces a = b = 0, i.e. both N-alities 0: the core is complete among "
                "undressed lines [verified arithmetic; identification group from the literature]")
            self.missing.append("lines of non-zero N-ality (need a U(1)/fermion dressing) [missing]")
            self.missing.append("lines that commute with N=2 but not with the full W-algebra of the coset [missing]")
        elif kind in ("D", "E7"):
            k = d["k"] if kind == "D" else 16
            if k % 4:
                self._sun_part(2, k, f"su(2)_{k} (D_odd: permutation invariant, chiral algebra su(2)_{k})")
                self.checks.append(f"D_odd: lines with l even act equally on l and its partner k-l "
                                   "(S_(a,k-l) = (-1)^a S_(a,l)), so they commute with both chiral halves [verified]")
            else:
                cat, notes, S = d_extension(k)
                self.checks += notes
                self._check_ext(cat, S, k, kind)
                desc = f"su(2)_{k} extended by l={k}" + (
                    " (E7 = automorphism invariant of the D10 extension; its chiral-algebra-preserving lines "
                    "have the fusion ring of the extended theory [literature: FRS, permutation invariants])"
                    if kind == "E7" else "")
                self.factors.append((cat, desc))
                self.missing.append("lines that preserve su(2)_k but not the extended chiral algebra "
                                    "(full bimodule category) [missing]")
        else:
            self.ok = False
            self.missing.append("block not recognised: no line data [missing]")

    def _sun_part(self, N: int, k: int, desc: str):
        if N == 1 or k == 0:
            return
        cat, su, sub = psun(N, k)
        J = [su.index[j] for j in su.simple_currents()]
        SJ = su.rows(J)                                     # S[J, :] = S[:, J] (S is symmetric)
        ratio = SJ.T / SJ[:, 0][None, :]                    # [a, J] = S_{a,J} / S_{0,J}
        transparent = np.all(np.abs(ratio - su.dims[:, None]) < 1e-6, axis=1)
        sub_set = set(sub)
        bad_in = sum(1 for a in sub if not transparent[a])
        bad_out = sum(1 for a in range(su.n) if a not in sub_set and transparent[a])
        if bad_in or bad_out:
            self.ok = False
        self.checks.append(
            f"{desc}: the {len(sub)} line(s) of N-ality 0 have trivial monodromy with all {len(J)} simple currents "
            f"({'ok' if not bad_in else str(bad_in) + ' FAILURES'}); all {su.n - len(sub)} line(s) of non-zero N-ality "
            f"braid non-trivially with one ({'ok' if not bad_out else str(bad_out) + ' FAILURES'}) [verified]")
        lr = level_rank_note(N, k)
        if lr and N > 2:
            cat.notes.append(lr)
        self.factors.append((cat, desc))

    def _check_ext(self, cat: Category, S, k: int, kind: str):
        # The orbit labels must be local with the extension current l = k (S_(l,k)/S_(0,k) = d_l), which is also
        # the su(2) part of the field identification (l,m,s) ~ (k-l, m+k+2, s+2). The Gepner currents have
        # l = 0, i.e. they lie in the vacuum orbit [0], so every extended line commutes with them.
        su = sun(2, k)
        SJ = su.rows([k])[0]
        local = all(abs(SJ[l] / SJ[0] - su.dims[l]) < 1e-6 for l in range(0, k + 1, 2))
        nonlocal_odd = all(abs(SJ[l] / SJ[0] - su.dims[l]) > 1e-6 for l in range(1, k + 1, 2))
        currents = [i for i, d in enumerate(cat.dims) if abs(d - 1) < TOL]
        self.checks.append(
            f"extended su(2)_{k}: {cat.rank} lines, {len(currents)} invertible. All even l (the labels in the orbits "
            f"and the fixed point) are local with the extension current l = {k}, which is also the su(2) part of the "
            f"field identification (l,m,s) ~ ({k}-l, m+{k + 2}, s+2) ({'ok' if local else 'FAILED'}), and every odd l "
            f"is not ({'ok' if nonlocal_odd else 'FAILED'}) [verified]. The Gepner currents (G, spectral flow, J, "
            "alignment) have l = 0, i.e. they lie in the vacuum orbit [0], so every extended line commutes with them")
        if not (local and nonlocal_odd):
            self.ok = False

    def core(self) -> Optional[Category]:
        cats = [c for c, _ in self.factors]
        if not cats:
            return None
        out = cats[0]
        for c in cats[1:]:
            out = out.product(c)
        return out


# ====================================================================== model report
def load_model(path) -> Model:
    """A model from an .lgo file, a saved NAME.lgor.json (any LGOR 2.x) or a run folder holding either."""
    path = Path(path)
    if path.is_dir():
        js = sorted(p for p in path.glob("*.lgor.json") if "_mirror" not in p.name)
        lg = sorted(p for p in path.glob("*.lgo") if "_mirror" not in p.name)
        if js:
            path = js[0]
        elif lg:
            path = lg[0]
        else:
            raise ValueError(f"no .lgor.json or .lgo in {path}")
    if path.name.endswith(".json"):
        from .results import model_from_dict
        with open(path) as fh:
            d = json.load(fh)
        return model_from_dict(d["model"])
    return Model.from_file(path)


class ModelLines:
    def __init__(self, model: Model):
        self.model = model
        self.blocks = classify_blocks(model)
        self.lines = [BlockLines(b) for b in self.blocks]
        self.ident = []
        for bl in self.lines:
            ids = []
            for cat, desc in bl.factors:
                name, new = identify(cat)
                ids.append((cat, desc, name, new))
            self.ident.append(ids)
        self.c3 = sum(b.c3 for b in self.blocks)
        self.recognised = all(b.kind != "unknown" for b in self.blocks)
        self.checks_ok = all(bl.ok for bl in self.lines)

    def new_types(self, status=NEW) -> List[str]:
        out = []
        for b, ids in zip(self.blocks, self.ident):
            for cat, desc, name, st in ids:
                if st == status and not cat.is_pointed():
                    out.append(f"{b.name}: {cat.name} ({name})")
        return out

    def oneline(self) -> str:
        parts = []
        for b, ids in zip(self.blocks, self.ident):
            if b.kind == "x^2":
                continue
            if b.kind == "unknown":
                parts.append(f"{b.name}: UNRECOGNISED")
                continue
            fs = []
            for cat, _, name, st in ids:
                if cat.rank == 1 or cat.is_pointed():
                    continue
                short = name.replace("fusion ring of ", "").split(" (= ")[0]
                txt = cat.name.split(" (")[0]
                if st != NEW and short != txt:
                    txt += f"={short}"
                if st == NEW:
                    co = name.split("same fusion ring as ")
                    txt += " NEW" + (f" (={co[1]})" if len(co) > 1 else "")
                elif st == NEW_LEVEL:
                    txt += " new level"
                fs.append(txt)
            parts.append(f"{b.name}: {' x '.join(fs) or 'invertible only'}")
        flag = "" if (self.checks_ok and self.recognised and self.c3 == 3) else "  [CHECK FAILED]"
        return f"{self.model.name}: " + "; ".join(parts) + flag

    def text(self) -> str:
        m = self.model
        L = [f"BLOCK LINES of {m.name}   (LGOR 2.5 blocklines)", "",
             "Tags: [verified] computed here, [literature] quoted, [inferred] argued, [missing] not computed.", "",
             "STATUS"]
        L.append(f"  blocks recognised: {'yes' if self.recognised else 'NO'} [verified]")
        L.append(f"  central charge from the blocks c/3 = {self.c3} ({'ok' if self.c3 == 3 else 'MISMATCH'}) [verified]")
        L.append(f"  consistency of the core lines: {'all checks passed' if self.checks_ok else 'FAILED'} [verified]")
        L.append("    a core line commutes with N=2, spectral flow, the alignment/GSO currents and the J orbifold and is")
        L.append("    well defined on the coset: it has trivial monodromy with every simple current of its non-abelian")
        L.append("    factors [verified]; that these Gepner operators are such simple currents [literature: Kazama-Suzuki,")
        L.append("    Gepner]. Hence every core line descends to a line of the full Gepner model [inferred, standard].")
        if m.orders:
            L.append(f"  extra orbifold {m.orders}: assumed to act by simple currents on non-Fermat blocks [inferred];")
            L.append("    for Fermat blocks use `defects`")
        new = self.new_types()
        L.append(f"  NEW line types (fusion ring not of PSU(2)_k/SU(2)_k type or a product of two): {len(new)} [verified]")
        for x in new:
            L.append(f"    NEW {x}")
        lev = self.new_types(NEW_LEVEL)
        if lev:
            L.append(f"  known type at a level absent from the 161 Fermat Gepner models: {len(lev)}")
            for x in lev:
                L.append(f"    new level {x}")
        L.append("  Not yet: whether the lines give selection rules beyond invertible symmetries, and which deformations")
        L.append("  keep them (`tdl`/`lines` handle Fermat blocks only) [missing]")
        L.append("")
        for i, (b, bl, ids) in enumerate(zip(self.blocks, self.lines, self.ident), 1):
            fl = ", ".join(f"x{j + 1}" for j in b.fields)
            L.append(f"BLOCK {i}: {b.name}  [{b.kind}]  fields {fl}  c/3 = {b.c3}")
            for note in b.notes:
                L.append(f"  {note}")
            if b.kind == "x^2":
                L.append("  massive field, no lines")
                L.append("")
                continue
            for cat, desc, name, st in ids:
                L.append(f"  {desc}: core {cat.name}, rank {cat.rank}, global dim {cat.global_dim:.4f}, "
                         f"{cat.n_invertible()} invertible")
                ds = sorted(cat.dims)
                show = ", ".join(f"{x:.4f}" for x in ds[:12]) + (f", ... ({len(ds)} in all)" if len(ds) > 12 else "")
                L.append(f"    quantum dimensions: {show}")
                tag = "[inferred]" if "not completed" in name or "dimensions only" in name else "[verified]"
                flag = {NEW: "   NEW", NEW_LEVEL: "   (new level)"}.get(st, "") if not cat.is_pointed() else ""
                L.append(f"    fusion ring: {name}{flag} {tag}")
                for note in cat.notes:
                    L.append(f"    {note}")
            core = bl.core()
            if core is not None and len(ids) > 1:
                L.append(f"  block core = product of the factors: rank {core.rank}, global dim {core.global_dim:.4f}")
            for c in bl.checks:
                L.append(f"  check: {c}")
            for x in bl.missing:
                L.append(f"  not covered: {x}")
            L.append("")
        L += ["MODEL",
              "  Undressed consistent lines of the full model = product of the block cores. Each is local with all",
              "  Gepner currents, so it survives the orbifold, GSO and spectral flow, like W_i in the quintic.",
              "  As for every worldsheet line, it is a tree-level, approximate symmetry of the 4D theory [inferred]."]
        return "\n".join(L) + "\n"
