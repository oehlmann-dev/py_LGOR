"""Genus-0 correlators of a Thom-Sebastiani product of A-type factors beyond 5 points, by WDVV reconstruction
(``lgor`` 3.12.1).

``lgor.frobenius.product_corr`` gives exact n = 3, 4, 5 correlators of a tensor product of A_{r-1} Frobenius
manifolds (tensor-product CohFT) but raises NotImplementedError for n >= 6 when two or more factors carry positive
degree.  Here the missing n-point correlators are obtained recursively from the WDVV equations

    sum_{S1 u S2 = S} sum_e  <a b e S1> <e* c d S2>  =  sum_{S1 u S2 = S} sum_e  <a c e S1> <e* b d S2>

(e* = dual basis element; the pairing is <e f 1>, which is a permutation matrix in the monomial basis).  For fixed
insertion multiset S the unknown (|S|+3)-point correlators enter linearly (through the terms with a 3-point
factor), everything else is known from lower n.  All equations for all (a,b,c,d) and S of the given size are
imposed; the result is accepted only if the linear system has a unique solution.  The 4- and 5-point data of
product_corr are re-checked against WDVV first (consistency test of the input).  Single-factor correlators
(one factor of positive degree) are exact (Saito flat coordinates of that factor) and are used as known values.
"""
from __future__ import annotations

import itertools
from collections import Counter
from fractions import Fraction as Fr

import sympy as sp

from ..frobenius import frob, product_corr


class ProductTFT:
    def __init__(self, rs):
        self.rs = list(rs)
        self.states = list(itertools.product(*[range(r - 1) for r in rs]))
        self.top = tuple(r - 2 for r in rs)
        self.dual = {s: tuple(r - 2 - a for r, a in zip(rs, s)) for s in self.states}
        self.known = {}
        self.max_n = 5
        self.q = {s: sum(Fr(a, r) for a, r in zip(s, rs)) for s in self.states}

    def _quads(self, S):
        """(a,b,c,d) with total charge q(abcdS) = |S| + 1 (only these give non-trivial WDVV equations)"""
        need = len(S) + 1 - sum(self.q[x] for x in S)
        for a, b, c in itertools.product(self.states, repeat=3):
            rest = need - self.q[a] - self.q[b] - self.q[c]
            for d in self.states:
                if self.q[d] == rest:
                    yield a, b, c, d

    # --- selection rules -----------------------------------------------------------------------------
    def degrees(self, sts):
        out = []
        for f, r in enumerate(self.rs):
            d = frob(r).degree(tuple(s[f] for s in sts))
            if d is None:
                return None
            out.append(d)
        return out

    def allowed(self, sts):
        d = self.degrees(sts)
        return d is not None and sum(d) == len(sts) - 3

    # --- correlators -----------------------------------------------------------------------------------
    def corr(self, sts):
        key = tuple(sorted(sts))
        if key in self.known:
            return self.known[key]
        n = len(key)
        if not self.allowed(key):
            return Fr(0)
        if n <= 5:
            v = product_corr(self.rs, list(key))
        else:
            d = self.degrees(key)
            if sum(1 for x in d if x > 0) <= 1:
                v = product_corr(self.rs, list(key)) if n <= 5 else self._single(key, d)
            else:
                raise KeyError("n=%d correlator %s not yet reconstructed (call extend(%d))" % (n, key, n))
        self.known[key] = v
        return v

    def _single(self, key, d):
        f = next((i for i, x in enumerate(d) if x > 0), 0)
        return frob(self.rs[f]).corr(tuple(s[f] for s in key))

    # --- WDVV ------------------------------------------------------------------------------------------
    def _splits(self, S):
        """all ordered splits of the labelled list S into (S1, S2)"""
        idx = range(len(S))
        for k in range(len(S) + 1):
            for A in itertools.combinations(idx, k):
                yield [S[i] for i in A], [S[i] for i in idx if i not in A]

    def _side(self, a, b, c, d, S, unk):
        """sum_{splits} sum_e <a b e S1><e* c d S2> as (constant, {unknown_key: coeff})"""
        const, lin = Fr(0), Counter()
        for S1, S2 in self._splits(S):
            for e in self.states:
                L = [a, b, e] + S1
                R = [self.dual[e], c, d] + S2
                if not self.allowed(L) or not self.allowed(R):
                    continue
                kl, kr = tuple(sorted(L)), tuple(sorted(R))
                ul, ur = kl in unk, kr in unk
                if ul and ur:
                    raise RuntimeError("quadratic unknowns")
                if ul:
                    lin[kl] += self.corr(R)
                elif ur:
                    lin[kr] += self.corr(L)
                else:
                    const += self.corr(L) * self.corr(R)
        return const, lin

    def check(self, n):
        """WDVV residuals of the (assumed known) correlators with |S| = n - 3; returns the number of failures"""
        bad = 0
        for S in itertools.combinations_with_replacement(self.states, n - 3):
            S = list(S)
            for a, b, c, d in self._quads(S):
                l, _ = self._side(a, b, c, d, S, set())
                r, _ = self._side(a, c, b, d, S, set())
                if l != r:
                    bad += 1
        return bad

    def extend(self, n):
        """reconstruct all allowed n-point correlators with >= 2 factors of positive degree"""
        unk = []
        for key in itertools.combinations_with_replacement(self.states, n):
            if key in self.known or not self.allowed(key):
                continue
            d = self.degrees(key)
            if sum(1 for x in d if x > 0) <= 1:
                self.known[key] = self._single(key, d)
            else:
                unk.append(key)
        unkset = set(unk)
        sym = {k: sp.Symbol("u%d" % i) for i, k in enumerate(unk)}
        eqs = set()
        for S in itertools.combinations_with_replacement(self.states, n - 3):
            S = list(S)
            for a, b, c, d in self._quads(S):
                l0, ll = self._side(a, b, c, d, S, unkset)
                r0, rl = self._side(a, c, b, d, S, unkset)
                expr = sp.Rational(l0 - r0) + sum(sp.Rational(v) * sym[k] for k, v in ll.items()) \
                    - sum(sp.Rational(v) * sym[k] for k, v in rl.items())
                expr = sp.expand(expr)
                if expr != 0:
                    eqs.add(expr)
        if not unk:
            return {"n": n, "unknowns": 0, "equations": len(eqs), "status": "nothing to reconstruct"}
        A, bvec = sp.linear_eq_to_matrix(list(eqs), [sym[k] for k in unk])
        rank = A.rank()
        status = "unique" if rank == len(unk) else "underdetermined (rank %d of %d)" % (rank, len(unk))
        sol = sp.linsolve((A, bvec), [sym[k] for k in unk])
        if not sol:
            return {"n": n, "unknowns": len(unk), "equations": len(eqs), "status": "INCONSISTENT"}
        sol = list(sol)[0]
        if rank == len(unk):
            for k, v in zip(unk, sol):
                v = sp.nsimplify(v)
                self.known[k] = Fr(int(sp.fraction(v)[0]), int(sp.fraction(v)[1]))
            self.max_n = max(self.max_n, n)
        return {"n": n, "unknowns": len(unk), "equations": len(eqs), "status": status,
                "nonzero": sum(1 for k in unk if self.known.get(k, 0) != 0)}

    def corr_fn(self):
        """drop-in replacement for product_corr in tft.certify (ns argument ignored)"""
        def f(ns, states):
            key = tuple(sorted(states))
            if len(key) > self.max_n:
                raise NotImplementedError
            return self.corr(key)
        return f


_TFT = {}


def get_tft(ns, nmax):
    """cached ProductTFT for exponents ns, reconstructed up to nmax points (raises if WDVV is not unique)"""
    key = tuple(ns)
    if key not in _TFT:
        _TFT[key] = ProductTFT(ns)
    T = _TFT[key]
    for n in range(T.max_n + 1, nmax + 1):
        r = T.extend(n)
        if r["status"] not in ("unique", "nothing to reconstruct"):
            raise RuntimeError("WDVV reconstruction at n=%d: %s" % (n, r["status"]))
        T.max_n = n
    return T


def check_solution(ns, SG, labels, tsec, tcls, mats, nmax):
    """Direct test of one orbit basis: max |class-forbidden correlator| for each n <= nmax (0 = line consistent).
    Use this rather than the solution count of certify at large nmax: the isolated-solution solver can miss
    solutions when there are many conditions (seen for (4,4) at n = 7, 8)."""
    from .tft import certify
    r = certify(ns, SG, labels, tsec, tcls, nmax=nmax)
    sub = {}
    for k, Mk in mats.items():
        Mk = sp.Matrix(Mk)
        for i in range(Mk.rows):
            for j in range(Mk.cols):
                sub[r["C"][k][i, j]] = Mk[i, j]
    return {n: max([abs(complex(sp.N(e.subs(sub)))) for e in cs] or [0]) for n, cs in sorted(r["conds"].items())}
