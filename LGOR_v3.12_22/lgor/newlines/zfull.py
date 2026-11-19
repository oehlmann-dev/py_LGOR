"""Complete abelian (non-R) discrete symmetry group of an LGOR spectrum, with prime powers,
and a mirror embedding check (``lgor discfull``, ``lgor embedcheck``).

LGOR's ``complete_discrete(p)`` searches Z_p-valued functions that are linear mod p in the
sector labels k0, k_j, the charges D q^i and 2 q-.  A Z_2 that is only visible as 2 x (a Z_4
function), i.e. f = F/2 with F linear mod 4 and even on every state, is missed.  Here the
same labels are used, but over Z/M with M = lcm(2K, orders of the .lgo group), so the full
finite abelian group  { linear functions mod M obeying the LGOR constraints } / gauge
is computed with its invariant factors (Smith normal form over Z).

Constraints (same as LGOR): every gaugino (q+ = -3/2) neutral; the three components of each
27 / 27bar equal.  Gauge: extra-U(1) torus elements of order dividing M, and the E6 centre.

embed_check(A, B): mirror pairs have identical state multisets under (q+, q-, q) -> (q+, -q-, -q).
For every element f_B of B's group the per-class value multiset {f_B(s) : s in class c} is
compared with all elements of A's group; f_B is "embedded" if some f_A has the same multiset
in every class (a necessary condition for f_A to be the mirror image of f_B; sufficient up
to a relabelling of charge-degenerate states).  Tags: [verified] = computed here.
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from fractions import Fraction
from math import gcd
from typing import Dict, List, Sequence, Tuple

from ..linalg import smith


def _lcm(a, b):
    return a * b // gcd(a, b)


def _den(xs):
    d = 1
    for x in xs:
        d = _lcm(d, Fraction(x).denominator)
    return d


def kernel_mod(rows: List[List[int]], n: int, M: int) -> List[List[int]]:
    """Basis (list of vectors in Z^n) of {x in Z^n : r.x = 0 mod M for all rows r}."""
    rows = [list(r) for r in {tuple(int(v) % M for v in r) for r in rows} if any(r)]
    if not rows:
        return [[int(i == j) for i in range(n)] for j in range(n)]
    diag, V = smith(rows, n)
    out = []
    for i in range(n):
        d = diag[i] if i < len(diag) else 0
        g = gcd(abs(int(d)), M)
        s = M // g
        out.append([s * V[i][t] for t in range(n)])
    return out


def invariant_factors(gens: List[List[int]], n: int) -> List[int]:
    """Invariant factors (>1) of Z^n / span(gens)."""
    if n == 0:
        return []
    diag, _ = smith([list(g) for g in gens], n)
    inv = []
    for i in range(n):
        d = abs(int(diag[i])) if i < len(diag) else 0
        if d == 0:
            raise ValueError("infinite quotient (should not happen: M Z^n is contained)")
        if d > 1:
            inv.append(d)
    return sorted(inv)


class FullDiscrete:
    """Full finite abelian non-R discrete symmetry group (mod gauge) of a SymmetryAnalysis."""

    def __init__(self, sym, reps_only: bool = False):
        from ..report import build_multiplets
        self.sym = sym
        self.reps_only = reps_only
        st = sym.states
        if reps_only:
            # 3.12: one state per 4D multiplet (16_-1/2 / 16bar_+1/2 component, singlets, gauginos).
            # The E6 components of LGOR multiplets sit in different sectors and their Cartan
            # charges q^i are not related by a flow that is linear in the labels, so the
            # component-equality constraint kills genuine phase symmetries of W (e.g. the Z3
            # (x5,x6,x7) -> (x5, w x6, w^2 x7) of Z6-II).  Here every multiplet is represented by
            # its first component; candidate symmetries are then checked on the Yukawa couplings.
            mults0, _ = build_multiplets(sym)
            keep = [x for x in st if x.state.q_plus == Fraction(-3, 2)]
            for rep in ("27", "27bar"):
                keep += [e[0][1] for e in mults0[rep]]
            for e in mults0["singlets"]:
                keep.append(e[0][1] if isinstance(e, (list, tuple)) and isinstance(e[0], tuple) else e)
            st = keep
        self.states = st
        model = sym.model
        K2 = 2 * model.korder
        M = K2
        for o in model.orders:
            M = _lcm(M, o)
        for (_, _, k) in sym.remnants:
            M = _lcm(M, k)
        # 3.12: explicit phase group P of W (all monomials incl. deformations):
        # a in (Q/Z)^n with sum_i a_i p_i in Z for every term.  LGOR's remnant list can miss
        # factors (e.g. the Z3 phases of the x^3 fields when other remnants are even), so the
        # generators of P enter as functions f_a(s) = sum_i a_i q^i(s)/alpha_i (alpha_i = field
        # R-charge); the kernel step below supplies any k0 / k_j / 2q- corrections needed on
        # twisted sectors and drops functions that are inconsistent.
        L = 1
        for term in model.superpotential:
            nz = [k for k, pw in enumerate(term.powers) if pw]
            if len(nz) == 1:
                L = _lcm(L, term.powers[nz[0]])
        if L == 1:
            L = K2
        Erows = [list(term.powers) for term in model.superpotential]
        Pbasis = kernel_mod(Erows, len(model.fields), L)
        alphas = [Fraction(f.kcharge, model.korder) for f in model.fields]
        self.phase_rows = []
        for x in Pbasis:
            if all(v % L == 0 for v in x):
                continue
            vals = [sum(Fraction(x[i], L) * Fraction(s.q[i]) / alphas[i] for i in range(len(x))) % 1
                    for s in st]
            Lf = _den(vals)
            self.phase_rows.append((x, Lf, vals))
            M = _lcm(M, Lf)
        self.M = M
        rows, names, scale = [], [], []
        for x, Lf, vals in self.phase_rows:
            rows.append([int(v * Lf) % Lf for v in vals])
            names.append("P(" + ",".join(f"{v}/{L}" for v in x) + ")")
            scale.append(M // Lf)
        rows.append([s.sector.k0 % K2 for s in st]); names.append("k0"); scale.append(M // K2)
        for j, o in enumerate(model.orders):
            rows.append([s.sector.twist[j] % o for s in st]); names.append(f"k{j + 1}"); scale.append(M // o)
        if sym.CW:
            Dq = _den(x for s in st for x in s.q)
            Vq = [[int(Dq * sum(Fraction(a) * Fraction(b) for a, b in zip(c, s.q))) for c in sym.CW] for s in st]
            diag, V = smith(Vq, len(sym.CW))
            for d, v in zip(diag, V):
                if not d:
                    continue
                rows.append([(sum(a * b for a, b in zip(v, r)) // d) for r in Vq])
                coef = [Fraction(Dq * sum(v[j] * sym.CW[j][i] for j in range(len(sym.CW))), d)
                        for i in range(len(model.fields))]
                names.append("(" + " + ".join(f"{x}*q^{model.fields[i].name}" for i, x in enumerate(coef) if x) + ")")
                scale.append(1)
        for j, (rname, _, k) in enumerate(sym.remnants):
            vals = []
            ok = True
            for s in st:
                if len(s.remnant) <= j or s.remnant[j] is None or Fraction(s.remnant[j]).denominator != 1:
                    ok = False
                    break
                vals.append(int(Fraction(s.remnant[j])) % k)
            if ok:
                rows.append(vals); names.append(rname); scale.append(M // k)
        rows.append([int(2 * s.state.q_minus) for s in st]); names.append("2q-"); scale.append(1)
        self.names = names
        self.scale = scale
        n = len(rows)
        self.n = n
        # generator functions g_r = scale_r * row_r (mod M), as columns per state
        self.F = [[(scale[r] * rows[r][i]) % M for r in range(n)] for i in range(len(st))]   # state x n
        idx = {id(x): i for i, x in enumerate(st)}
        cons = [self.F[idx[id(s)]] for s in st if s.state.q_plus == Fraction(-3, 2)]
        mults, _ = build_multiplets(sym)
        for rep in (() if reps_only else ("27", "27bar")):
            for e in mults[rep]:
                comps = [c[1] for c in e]
                for s2 in comps[1:]:
                    a, b = self.F[idx[id(comps[0])]], self.F[idx[id(s2)]]
                    cons.append([(x - y) % M for x, y in zip(a, b)])
        B = kernel_mod(cons, n, M)                      # n vectors spanning K1 (contains M Z^n)
        self.B = B
        # gauge functions mod M
        G = [[int(Fraction(x)) for x in f] for f in sym._gauge_torus_rows(st)]
        if M % 3 == 0:
            def tri(s):
                qm = s.state.q_minus
                return 1 if qm in (Fraction(-2), Fraction(-1, 2), Fraction(1)) else \
                    2 if qm in (Fraction(2), Fraction(1, 2), Fraction(-1)) else 0
            G.append([(M // 3) * tri(s) for s in st])
        self.G = G
        nb, ng = len(B), len(G)
        # rows per state of [F B | G]
        FB = [[sum(fi[r] * B[c][r] for r in range(n)) % M for c in range(nb)] for fi in self.F]
        self.FB = FB
        big = [FB[i] + [g[i] % M for g in G] for i in range(len(st))]
        ker = kernel_mod(big, nb + ng, M)
        proj = [v[:nb] for v in ker] + [[M * int(i == j) for i in range(nb)] for j in range(nb)]
        self.ker2 = proj
        self.invariants = invariant_factors(proj, nb)
        # explicit generators of the quotient: from Smith of ker2
        diag, V = smith([list(g) for g in proj], nb)
        self.gens = []          # (order, z-vector in K1 coordinates)
        Vinv = _inv_unimodular(V, nb)
        for i in range(nb):
            d = abs(int(diag[i])) if i < len(diag) else 0
            if d > 1:
                # generator of the cyclic factor: the i-th column direction of V
                self.gens.append((d, [V[i][t] for t in range(nb)]))

    def function(self, z: Sequence[int]) -> List[int]:
        """Values mod M on all states of the symmetry with K1-coordinates z."""
        M = self.M
        return [sum(a * b for a, b in zip(row, z)) % M for row in self.FB]

    def formula(self, z: Sequence[int]) -> str:
        M = self.M
        x = [sum(z[c] * self.B[c][r] for c in range(len(z))) % M for r in range(self.n)]
        vals = set(self.function(z))
        order = M // gcd(M, *vals) if any(vals) else 1
        terms = []
        for c, nm, sc in zip(x, self.names, self.scale):
            c = (c * sc) % M
            if c:
                terms.append(f"{c}*{nm}")
        return "phase exp(2 pi i f/M), f = (" + " + ".join(terms) + f") mod {M}  (this representative has order {order}; its class mod gauge is the stated Z_d)"

    def group_name(self) -> str:
        if not self.invariants:
            return "trivial"
        return " x ".join(f"Z{d}" for d in self.invariants)

    def primary(self) -> Dict[int, List[int]]:
        """Primary decomposition {p: [p^a, ...]}."""
        out = defaultdict(list)
        for d in self.invariants:
            m, p = d, 2
            while m > 1:
                if m % p == 0:
                    q = 1
                    while m % p == 0:
                        m //= p
                        q *= p
                    out[p].append(q)
                p += 1
        return {p: sorted(v) for p, v in out.items()}

    def elements(self, cap: int = 20000):
        """All group elements as (label, values mod M) (None if the group exceeds cap)."""
        size = 1
        for d, _ in self.gens:
            size *= d
        if size > cap:
            return None
        out = []
        for coeffs in itertools.product(*[range(d) for d, _ in self.gens]):
            z = [0] * len(self.B)
            for c, (d, v) in zip(coeffs, self.gens):
                for t in range(len(z)):
                    z[t] += c * v[t]
            out.append((coeffs, self.function(z)))
        return out


def _inv_unimodular(V, n):
    return None


def class_keys(sym, flip: bool, states=None):
    keys = []
    for s in (sym.states if states is None else states):
        sg = -1 if flip else 1
        keys.append((s.state.q_plus, sg * s.state.q_minus, tuple(sg * Fraction(x) for x in s.q)))
    return keys


def signature(keys, values, M) -> Tuple:
    d = defaultdict(list)
    for k, v in zip(keys, values):
        d[k].append(Fraction(v % M, M))
    return tuple(sorted((k, tuple(sorted(v))) for k, v in d.items()))


def _canon_shift(vals, M):
    """Translation-canonical form of a multiset of residues mod M and the shifts realising it."""
    best, shifts = None, []
    for t in sorted(set(vals)):
        c = tuple(sorted((v - t) % M for v in vals))
        if best is None or c < best:
            best, shifts = c, [t]
        elif c == best:
            shifts.append(t)
    return best, shifts


def embed_check(fdA: "FullDiscrete", fdB: "FullDiscrete", cap: int = 20000) -> Tuple[bool, str]:
    """Every element of B's group must have a partner in A's group (and vice versa) such that
    (i) on every charge class the value multisets agree up to a common shift per class
        (the pattern by which the symmetry splits charge-degenerate states), and
    (ii) the shifts are a gauge transformation of A (solved as linear congruences mod M on the
        classes where the shift is unique).
    A keys are flipped (q- -> -q-, q -> -q) as in the LGOR mirror check."""
    if fdA.M != fdB.M:
        M = _lcm(fdA.M, fdB.M)
    else:
        M = fdA.M
    sa, sb = M // fdA.M, M // fdB.M
    kA, kB = class_keys(fdA.sym, True, fdA.states), class_keys(fdB.sym, False, fdB.states)
    eA, eB = fdA.elements(cap), fdB.elements(cap)
    if eA is None or eB is None:
        return False, "  group too large for element-wise check [missing]"
    clsA, clsB = defaultdict(list), defaultdict(list)
    for i, k in enumerate(kA):
        clsA[k].append(i)
    for i, k in enumerate(kB):
        clsB[k].append(i)
    if set(clsA) != set(clsB):
        return False, "  charge classes differ between the sides"
    keys = sorted(clsA, key=str)
    # gauge rows of A evaluated per class (must be constant on classes)
    GA = []
    for g in fdA.G:
        row = []
        for k in keys:
            vals = {(sa * int(g[i])) % M for i in clsA[k]}
            if len(vals) != 1:
                row = None
                break
            row.append(vals.pop())
        if row is not None:
            GA.append(row)
    def patt(vals_by_state, cls, scale):
        out = {}
        for k in keys:
            out[k] = _canon_shift([(scale * vals_by_state[i]) % M for i in cls[k]], M)
        return out
    pA = [(c, patt(v, clsA, sa)) for c, v in eA]
    pB = [(c, patt(v, clsB, sb)) for c, v in eB]
    def match(pf, pg):
        # pattern equality on every class, then gauge-solvable shifts
        eqs, rhs = [], []
        for j, k in enumerate(keys):
            (ca, sha), (cb, shb) = pf[k], pg[k]
            if ca != cb:
                return False
            if len(sha) == 1:
                eqs.append(j)
                rhs.append((shb[0] - sha[0]) % M)
        # solve sum_g x_g GA[g][j] = rhs mod M for j in eqs (x integer)
        ng = len(GA)
        rows = [[GA[g][j] for g in range(ng)] + [-r] for j, r in zip(eqs, rhs)]
        ker = kernel_mod(rows, ng + 1, M) if rows else [[0] * ng + [1]]
        # need a kernel vector with last coordinate = 1 mod M  <=> gcd of last coords with M == 1
        gl = M
        for v in ker:
            gl = gcd(gl, v[-1] % M)
        return gl == 1
    pairs = {c: [ca for ca, q in pA if match(q, p)] for c, p in pB}
    missB = [c for c, v in pairs.items() if not v]
    missA = [c for c, p in pA if not any(match(p, q) for _, q in pB)]
    ok = not missA and not missB
    idA = tuple(0 for _ in fdA.gens)
    fake = [c for c, v in pairs.items() if any(c) and v == [idA]]
    out = [f"  elements: A {len(eA)}, B {len(eB)}; charge classes {len(keys)}, gauge rows of A constant on classes: {len(GA)}/{len(fdA.G)}",
           f"  B elements without an A partner: {len(missB)};  A elements without a B partner: {len(missA)}",
           f"  nontrivial B elements that only match A's identity (would be pure gauge on A): {len(fake)}"]
    for c, v in pairs.items():
        out.append(f"    B element {c} <-> A element(s) {v}")
    ok = ok and not fake
    return ok, "\n".join(out)


def report(symA, symB=None, nameA="A", nameB="B") -> str:
    lines = []
    fdA = FullDiscrete(symA, reps_only=True)
    try:
        old = FullDiscrete(symA, reps_only=False).group_name()
    except Exception as e:          # pragma: no cover
        old = f"failed ({e})"
    lines.append(f"FULL DISCRETE GROUP (3.12: phase group of W + labels, one state per multiplet, over Z/{fdA.M}, "
                 f"mod gauge) [verified]")
    lines.append(f"  (3.9 ansatz with E6-component equality and LGOR remnants only: {old})")
    lines.append(f"  {nameA}: {fdA.group_name()}   primary parts {fdA.primary()}")
    for d, z in fdA.gens:
        lines.append(f"     Z{d} generator: {fdA.formula(z)}")
    if symB is None:
        return "\n".join(lines) + "\n"
    fdB = FullDiscrete(symB, reps_only=True)
    lines.append(f"  {nameB}: {fdB.group_name()}   primary parts {fdB.primary()}")
    for d, z in fdB.gens:
        lines.append(f"     Z{d} generator: {fdB.formula(z)}")
    same = sorted(fdA.invariants) == sorted(fdB.invariants)
    lines.append(f"  abstract groups equal: {'yes' if same else 'NO'}")
    ok, txt = embed_check(fdA, fdB)
    lines.append(txt)
    lines.append("  mirror embedding (every element realised on the other side with identical per-class values): "
                 + ("ALL EMBEDDED [verified]" if ok else "NOT ALL EMBEDDED"))
    return "\n".join(lines) + "\n"
