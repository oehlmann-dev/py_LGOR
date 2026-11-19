"""Small finite groups and group-theoretical fusion categories.

* permutation groups from generators (elements are tuples);
* C(G, K): the group-theoretical category of K-K bimodules in Vec_G (trivial cocycles).  It is
  the category of lines of the orbifold C/K when C has an anomaly-free symmetry G and K <= G is
  gauged.  C(G, G) = Rep(G), C(G, 1) = Vec_G.  Simple objects are pairs (double coset K g K,
  irreducible character of K cap g K g^-1); fusion is computed from K x K-equivariant characters
  (exact convolution formula, numerics only in the final rounding);
  for non-abelian stabilisers the irreducible characters come from Burnside's algorithm (numpy).
* character data of a commutative fusion ring (numpy), to compare an integral fusion ring with
  Rep(G) of small groups.
"""

from __future__ import annotations

import cmath
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

Perm = Tuple[int, ...]

try:
    import numpy as _np
except Exception:          # pragma: no cover
    _np = None


def have_numpy() -> bool:
    return _np is not None


def mul(a: Perm, b: Perm) -> Perm:
    """(a b)(x) = a(b(x))."""
    return tuple(a[x] for x in b)


def inv(a: Perm) -> Perm:
    r = [0] * len(a)
    for i, x in enumerate(a):
        r[x] = i
    return tuple(r)


def closure(gens: Sequence[Perm], n: int) -> List[Perm]:
    e = tuple(range(n))
    seen = {e}
    frontier = [e]
    gens = [tuple(g) for g in gens]
    while frontier:
        new = []
        for x in frontier:
            for g in gens:
                y = mul(g, x)
                if y not in seen:
                    seen.add(y)
                    new.append(y)
        frontier = new
    return sorted(seen)


def symmetric_group(n: int) -> List[Perm]:
    return sorted(itertools.permutations(range(n)))


def conj_classes(G: List[Perm]) -> List[List[Perm]]:
    left = set(G)
    out = []
    for g in G:
        if g not in left:
            continue
        cl = sorted({mul(mul(h, g), inv(h)) for h in G})
        for x in cl:
            left.discard(x)
        out.append(cl)
    return out


def is_abelian(K: List[Perm]) -> bool:
    return all(mul(a, b) == mul(b, a) for a in K for b in K)


# ----------------------------------------------------------------- characters
def abelian_characters(K: List[Perm]) -> List[Dict[Perm, complex]]:
    """All irreducible characters of an abelian permutation group."""
    e = tuple(range(len(K[0])))
    # greedy direct-product basis
    def order(g):
        o, x = 1, g
        while x != e:
            x, o = mul(g, x), o + 1
        return o
    basis, span = [], {e}
    for g in sorted(K, key=lambda x: -order(x)):
        if g in span:
            continue
        o = order(g)
        powers = [e]
        for _ in range(o - 1):
            powers.append(mul(g, powers[-1]))
        new = {mul(p, s) for p in powers for s in span}
        if len(new) == len(span) * o:
            basis.append((g, o))
            span = new
        if len(span) == len(K):
            break
    if len(span) != len(K):
        raise RuntimeError("no direct-product basis found")
    # coordinates of every element
    coords = {e: tuple(0 for _ in basis)}
    frontier = [e]
    while frontier:
        new = []
        for x in frontier:
            for t, (g, o) in enumerate(basis):
                y = mul(g, x)
                if y not in coords:
                    c = list(coords[x])
                    c[t] = (c[t] + 1) % o
                    coords[y] = tuple(c)
                    new.append(y)
        frontier = new
    chars = []
    for ks in itertools.product(*[range(o) for _, o in basis]):
        chars.append({x: cmath.exp(2j * math.pi * sum(k * c / o for k, c, (_, o) in zip(ks, coords[x], basis)))
                      for x in K})
    return chars


def irreducible_characters(G: List[Perm]) -> List[Dict[Perm, complex]]:
    """Irreducible characters (Burnside's algorithm).  Abelian groups need no numpy."""
    if is_abelian(G):
        return abelian_characters(G)
    if _np is None:
        raise RuntimeError("numpy is needed for characters of non-abelian groups")
    cls = conj_classes(G)
    r = len(cls)
    index = {}
    for i, cl in enumerate(cls):
        for x in cl:
            index[x] = i
    order = len(G)
    # class multiplication coefficients: C_i C_j = sum_k c_ijk C_k
    M = _np.zeros((r, r, r))
    for i, ci in enumerate(cls):
        for j, cj in enumerate(cls):
            cnt = [0] * r
            for x in ci:
                for y in cj:
                    cnt[index[mul(x, y)]] += 1
            for kk in range(r):
                M[i, j, kk] = cnt[kk] / len(cls[kk])
    rng = _np.random.default_rng(1)
    coef = rng.normal(size=r)
    A = sum(coef[i] * M[i] for i in range(r))      # A[j,k]: acting on class-sum basis
    w, V = _np.linalg.eig(A)
    chars = []
    sizes = _np.array([len(c) for c in cls])
    inv_class = [index[inv(cl[0])] for cl in cls]
    for t in range(r):
        v = V[:, t]                                  # proportional to omega_k = |C_k| chi(C_k)/chi(1)
        v = v / v[0]
        # chi(1)^2 = |G| / sum_k |C_k|^-1 omega_k omega_{k^-1}
        s = sum(v[kk] * v[inv_class[kk]] / sizes[kk] for kk in range(r))
        d = math.sqrt(abs(order / s))
        chi = {x: complex(d * v[index[x]] / sizes[index[x]]) for x in G}
        chars.append(chi)
    chars.sort(key=lambda c: (round(c[G[0]].real), ))
    return chars


# ----------------------------------------------------------------- C(G, K)
class GroupTheoretical:
    """Grothendieck ring of C(G, K) (trivial cocycles)."""

    def __init__(self, G: List[Perm], K: List[Perm]):
        self.G = G
        self.K = K
        self.Kset = set(K)
        n = len(G[0])
        self.e = tuple(range(n))
        # double cosets with representatives and coset maps u_x
        left = set(G)
        self.dcosets = []
        for g in G:
            if g not in left:
                continue
            orbit = {}
            for a in K:
                for b in K:
                    x = mul(mul(a, g), inv(b))
                    if x not in orbit:
                        orbit[x] = (a, b)
            for x in orbit:
                left.discard(x)
            self.dcosets.append((g, orbit))
        # simple objects
        self.simples = []
        for di, (g, orbit) in enumerate(self.dcosets):
            ginv = inv(g)
            Kg = [k for k in K if mul(mul(ginv, k), g) in self.Kset]      # K cap g K g^-1
            for ci, chi in enumerate(irreducible_characters(Kg)):
                self.simples.append((di, ci, Kg, chi))
        self.N = None

    def dims(self) -> List[float]:
        out = []
        for di, ci, Kg, chi in self.simples:
            g, orbit = self.dcosets[di]
            out.append(len(orbit) / len(self.K) * chi[self.e].real)
        return out

    def _char(self, idx):
        """chi(x; k1, k2) as a dict over x in the double coset and (k1, k2) in Stab(x)."""
        di, ci, Kg, chi = self.simples[idx]
        g, orbit = self.dcosets[di]
        out = {}
        for x, (a, b) in orbit.items():
            ainv = inv(a)
            for k1 in self.K:
                # stabiliser of x: k1 x k2^-1 = x  -> k2 = x^-1 k1 x
                k2 = mul(mul(inv(x), k1), x)
                if k2 not in self.Kset:
                    continue
                # pull back to g: u^-1 (k1, k2) u with u = (a, b): (a^-1 k1 a, b^-1 k2 b)
                h1 = mul(mul(ainv, k1), a)
                out[(x, k1)] = chi[h1]
        return out

    def fusion(self) -> List[List[Dict[int, int]]]:
        if self.N is not None:
            return self.N
        chars = [self._char(i) for i in range(len(self.simples))]
        nK = len(self.K)
        Ginv = {x: inv(x) for x in self.G}
        n = len(self.simples)
        N = [[{} for _ in range(n)] for _ in range(n)]
        for A in range(n):
            for B in range(n):
                prod = {}
                cA, cB = chars[A], chars[B]
                # chi_{A B}(g; k1, k2) = 1/|K| sum_{x: x^-1 k1 x in K} chi_A(x; k1, x^-1 k1 x) chi_B(x^-1 g; x^-1 k1 x, k2)
                for (x, k1), va in cA.items():
                    k = mul(mul(Ginv[x], k1), x)
                    for (y, kk), vb in cB.items():
                        if kk != k:
                            continue
                        gg = mul(x, y)
                        prod[(gg, k1)] = prod.get((gg, k1), 0) + va * vb / nK
                for C in range(n):
                    cC = chars[C]
                    s = sum(prod.get(key, 0) * v.conjugate() for key, v in cC.items())
                    m = s / nK ** 2 * 1.0
                    # normalisation: <chi, chi> = (1/|K|^2) sum_x sum_stab |chi|^2 ... see docstring
                    val = m.real
                    if abs(val) > 1e-6:
                        N[A][B][C] = int(round(val))
                        if abs(val - round(val)) > 1e-6:
                            raise RuntimeError("non-integer fusion coefficient")
        self.N = N
        return N

    def labels(self) -> List[str]:
        out = []
        for di, ci, Kg, chi in self.simples:
            g, orbit = self.dcosets[di]
            out.append(f"[{_cyc(g)}]{ci}")
        return out


def _cyc(p: Perm) -> str:
    seen, parts = set(), []
    for i in range(len(p)):
        if i in seen or p[i] == i:
            seen.add(i)
            continue
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(str(j + 1))
            j = p[j]
        parts.append("(" + " ".join(c) + ")")
    return "".join(parts) or "e"


# ----------------------------------------------------------------- fusion rings vs Rep(G)
def ring_character_data(N: List[List[Dict[int, int]]], dims: List[float]):
    """For a commutative fusion ring with structure constants N[a][b][c]: the 'class sizes'
    |G| / sum_a |chi_a(c)|^2 of its characters (= class sizes of G if the ring is Rep(G))."""
    if _np is None:
        return None
    n = len(N)
    if any(N[a][b] != N[b][a] for a in range(n) for b in range(n)):
        return None                       # not commutative: not the fusion ring of Rep(G)
    mats = []
    for a in range(n):
        M = _np.zeros((n, n))
        for b in range(n):
            for c, v in N[a][b].items():
                M[c, b] = v
        mats.append(M)
    rng = _np.random.default_rng(2)
    A = sum(rng.normal() * M for M in mats)
    w, V = _np.linalg.eig(A)
    order = sum(d * d for d in dims)
    sizes = []
    for t in range(n):
        v = V[:, t]
        # eigenvector of left multiplication: components along basis b are chi_b(c) up to scale
        lam = [(mats[a] @ v)[0] / v[0] if abs(v[0]) > 1e-9 else None for a in range(n)]
        if any(x is None for x in lam):
            return None
        s = sum(abs(x) ** 2 for x in lam)
        sizes.append(order / s)
    return round(order), sorted(round(x, 6) for x in sizes)


SMALL_GROUPS = {}


def _small_groups():
    if SMALL_GROUPS:
        return SMALL_GROUPS
    def add(name, gens, n):
        SMALL_GROUPS[name] = closure(gens, n)
    add("S3", [(1, 0, 2), (1, 2, 0)], 3)
    add("D4 (order 8)", [(1, 2, 3, 0), (0, 3, 2, 1)], 4)
    # Q8 as regular permutation representation
    q = {}
    units = ["1", "i", "j", "k", "-1", "-i", "-j", "-k"]
    def qm(a, b):
        sa, xa = (1, a) if not a.startswith("-") else (-1, a[1:])
        sb, xb = (1, b) if not b.startswith("-") else (-1, b[1:])
        table = {("1", x): (1, x) for x in "1ijk"}
        table.update({(x, "1"): (1, x) for x in "1ijk"})
        table.update({("i", "i"): (-1, "1"), ("j", "j"): (-1, "1"), ("k", "k"): (-1, "1"),
                      ("i", "j"): (1, "k"), ("j", "k"): (1, "i"), ("k", "i"): (1, "j"),
                      ("j", "i"): (-1, "k"), ("k", "j"): (-1, "i"), ("i", "k"): (-1, "j")})
        s, x = table[(xa, xb)]
        s *= sa * sb
        return x if s == 1 else "-" + x
    for gname in ("i", "j"):
        q[gname] = tuple(units.index(qm(gname, u)) for u in units)
    add("Q8", [q["i"], q["j"]], 8)
    add("D5 (order 10)", [(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)], 5)
    add("A4", [(1, 2, 0, 3), (1, 0, 3, 2)], 4)
    add("D6 (order 12)", [(1, 2, 3, 4, 5, 0), (0, 5, 4, 3, 2, 1)], 6)
    add("S4", [(1, 0, 2, 3), (1, 2, 3, 0)], 4)
    add("D8 (order 16)", [(1, 2, 3, 4, 5, 6, 7, 0), (0, 7, 6, 5, 4, 3, 2, 1)], 8)
    add("A5", [(1, 2, 0, 3, 4), (1, 2, 3, 4, 0)], 5)
    return SMALL_GROUPS


def class_size_data(G: List[Perm]):
    return len(G), sorted(float(len(c)) for c in conj_classes(G))


def match_rep_groups(order: int, sizes: List[float]) -> List[str]:
    """Names of library groups (and products G x H, G x abelian) with these class sizes."""
    out = []
    target = sorted(round(x) for x in sizes)
    if any(abs(x - round(x)) > 1e-4 for x in sizes):
        return out
    if all(x == 1 for x in target):
        return [f"abelian of order {order}"]
    lib = [(name, class_size_data(G)) for name, G in _small_groups().items()]
    lib = [(nm, o, sorted(round(x) for x in s)) for nm, (o, s) in lib]
    for nm, o, s in lib:
        if o == order and s == target:
            out.append(nm)
        elif order % o == 0 and sorted(s * (order // o)) == target:
            out.append(f"{nm} x (abelian of order {order // o})")
    for (n1, o1, s1), (n2, o2, s2) in itertools.combinations_with_replacement(lib, 2):
        if o1 * o2 == order and sorted(a * b for a in s1 for b in s2) == target:
            out.append(f"{n1} x {n2}")
    return out
