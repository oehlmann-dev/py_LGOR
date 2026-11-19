"""Candidate groups: abelian A with an automorphism sigma, G = A x| <sigma>, classes and irreps."""
import cmath
import itertools
import math
from collections import Counter
from fractions import Fraction as Fr

from .. import fingroups as fg


def _factor(n):
    f, p = Counter(), 2
    while p * p <= n:
        while n % p == 0:
            f[p] += 1
            n //= p
        p += 1
    if n > 1:
        f[n] += 1
    return f


def _partitions(e, maxpart=None):
    maxpart = maxpart or e
    if e == 0:
        yield ()
        return
    for k in range(min(e, maxpart), 0, -1):
        for rest in _partitions(e - k, k):
            yield (k,) + rest


def abelian_groups(n):
    """All abelian groups of order n as tuples of cyclic orders (prime-power form)."""
    per = [[tuple(p ** k for k in part) for part in _partitions(e)] for p, e in _factor(n).items()]
    return [sum(c, ()) for c in itertools.product(*per)] if per else [()]


class AbGroup:
    def __init__(self, mods):
        self.mods = tuple(mods)
        self.els = list(itertools.product(*[range(m) for m in self.mods]))
        self.idx = {x: i for i, x in enumerate(self.els)}
        self.n = len(self.els)

    def add(self, x, y):
        return tuple((a + b) % m for a, b, m in zip(x, y, self.mods))

    def scal(self, k, x):
        return tuple((k * a) % m for a, m in zip(x, self.mods))

    def name(self):
        return " x ".join("Z%d" % m for m in self.mods) if self.mods else "1"

    def automorphisms(self):
        """All automorphisms as permutations of element indices."""
        r = len(self.mods)
        choices = [[y for y in self.els if all(self.scal(m, y) == self.els[0] for _ in [0])] for m in self.mods]
        out = []
        for imgs in itertools.product(*choices):
            perm = []
            for x in self.els:
                y = self.els[0]
                for xi, im in zip(x, imgs):
                    y = self.add(y, self.scal(xi, im))
                perm.append(self.idx[y])
            if len(set(perm)) == self.n:
                out.append(tuple(perm))
        return out


def _perm_power(p, k):
    e = tuple(range(len(p)))
    r = e
    for _ in range(k):
        r = fg.mul(p, r)
    return r


def _perm_order(p):
    e = tuple(range(len(p)))
    o, x = 1, p
    while x != e:
        x, o = fg.mul(p, x), o + 1
    return o


MAX_AUT_ENUM = 200_000
SKIPPED = []          # (group, enumeration size) skipped by the guard


def candidate_actions(row, K):
    """All (A, sigma) with |A| = row[0] and #Fix(sigma^l) = row[l], up to conjugation in Aut(A)."""
    n = row[0]
    out = []
    if n <= 0 or any(r < 1 or n % r for r in row[1:]):     # Fix(sigma^l) is a subgroup: |Fix| divides |A|
        return out
    for mods in abelian_groups(n):
        A = AbGroup(mods)
        size = 1
        for m in A.mods:
            size *= sum(1 for y in A.els if A.scal(m, y) == A.els[0])
        if size > MAX_AUT_ENUM:
            SKIPPED.append((A.name(), size))
            continue
        auts = A.automorphisms()
        seen = set()
        for s in auts:
            if K % _perm_order(s):
                continue
            if any(sum(1 for i, j in enumerate(_perm_power(s, l)) if i == j) != row[l] for l in range(K)):
                continue
            if s in seen:
                continue
            seen |= {fg.mul(fg.mul(t, s), fg.inv(t)) for t in auts}
            out.append((A, s))
    return out


class SpaceGroup:
    """G = A x| <sigma> as affine permutations x -> a + sigma^j x of A."""

    def __init__(self, A, sigma):
        self.A, self.sigma = A, sigma
        self.m = _perm_order(sigma)
        self.sig = [_perm_power(sigma, j) for j in range(self.m)]
        self.el = {}
        for a in A.els:
            for j in range(self.m):
                self.el[(a, j)] = tuple(A.idx[A.add(a, A.els[self.sig[j][i]])] for i in range(A.n))
        self.G = sorted(set(self.el.values()))
        self.classes = fg.conj_classes(self.G)
        self.cindex = {x: i for i, cl in enumerate(self.classes) for x in cl}
        self.e = tuple(range(A.n))
        self.order = len(self.G)
        self._allow = {}
        self.chars = fg.irreducible_characters(self.G)

    def name(self):
        if self.m == 1:
            return self.A.name()
        known = {(12, 4): "A4", (6, 3): "S3", (8, 5): "D4", (18, 6): "(Z3xZ3) x| Z2", (24, 5): "S4"}
        nm = known.get((self.order, len(self.classes)))
        base = "%s x| Z%d" % (self.A.name(), self.m)
        return "%s = %s" % (base, nm) if nm else base

    def cls(self, a, j):
        return self.cindex[self.el[(a, j % self.m)]]

    def allowed(self, cls_tuple):
        key = tuple(sorted(cls_tuple))
        if key not in self._allow:
            cur = Counter({self.e: 1})
            for c in key:
                nxt = Counter()
                for g, mlt in cur.items():
                    for h in self.classes[c]:
                        nxt[fg.mul(g, h)] += mlt
                cur = nxt
            self._allow[key] = cur[self.e] > 0
        return self._allow[key]

    def allowed_abelian(self, cls_tuple):
        """allowed by every invertible (1-dim) line of G"""
        key = ("ab",) + tuple(sorted(cls_tuple))
        if key not in self._allow:
            ok = True
            for ch in self.chars:
                if abs(ch[self.e] - 1) > 1e-9:
                    continue
                v = 1
                for c in cls_tuple:
                    v *= ch[self.classes[c][0]]
                ok &= abs(v - 1) < 1e-9
            self._allow[key] = ok
        return self._allow[key]

    def orbits(self):
        """sigma-orbits on A = classes of pure translations, as class indices (orbit of 0 first)."""
        out = []
        for a in self.A.els:
            c = self.cls(a, 0)
            if c not in out:
                out.append(c)
        return out

    def irreps(self):
        """[(dim, {class: character})]"""
        res = []
        for ch in self.chars:
            d = round(ch[self.e].real)
            vals = {}
            for i, cl in enumerate(self.classes):
                v = ch[cl[0]]
                vals[i] = complex(round(v.real, 9), round(v.imag, 9))
            res.append((d, vals))
        res.sort(key=lambda t: t[0])
        return res


