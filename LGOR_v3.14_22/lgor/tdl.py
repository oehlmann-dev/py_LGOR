"""Verlinde lines (topological defect lines, TDLs) of Fermat Landau-Ginzburg orbifolds.

For W = sum_i x_i^{n_i} the Gepner point is the product of N=2 minimal models at levels
k_i = n_i - 2.  The Verlinde lines L_r (r = 0..k) of factor i act on a primary with su(2)_k
label l by S_{r,l}/S_{0,l}; their quantum dimension is d_r = sin((r+1)pi/(k+2))/sin(pi/(k+2)).
L_k is invertible ((-1)^l); for k = 3, L_2 is the Fibonacci line W of Cordova-Rizi.

This module provides

* ``state_labels``: the per-factor labels l_i of every massless LGOR state.  Massless states
  have right-movers in R ground states, so q_+^i = (l_i + 1)/(k_i + 2) - 1/2 mod 1, where q_+^i
  is the per-field right-moving charge.
* ``line_loci``: for each non-invertible line, the marginal (untwisted, orbifold-invariant)
  monomials it keeps, i.e. those on which it acts with eigenvalue +-d (x_i exponent 0 or k_i).
* ``classify_couplings``: the flat-coordinate 27^3.1 four-point functions dC_abc/dt_d of the
  untwisted chiral ring at the Fermat point (residue + DVV contact terms), compared with
  - the weak fusion rule: in each factor the su(2)_k fusion of the four labels contains 0 or k;
  - the strict fusion rule, which keeps track of the N=2 charges: chiral primaries are
    (a, a, 0) and the marginal descendant G^- Gbar^- phi_d is (d, d, 2).  In the factor carrying
    the marginal operator (exponents summing to 2k+2) the labels must fuse to 0, in every
    other factor to k.  For k even only the odd-r lines tell 0 from k.
  - the Fermat factorization zero: the special factor carries an identity entry.
* ``abelian_lattice_test``: whether any abelian (phase) symmetry, R or not, continuous or
  discrete, could explain the vanishing couplings (integer-span test).

Only Fermat superpotentials are supported.  Couplings involve untwisted 27s only.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction as Fr
from itertools import combinations_with_replacement, permutations, product
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Model
from .monomial import B as _B, BB as _BB, F as _F, BF as _BF


# ----------------------------------------------------------------------------- su(2)_k data
def line_dim(k: int, r: int, l: int = 0) -> float:
    """S_{r,l}/S_{0,l} of su(2)_k (equals d_r for l = 0)."""
    return math.sin(math.pi * (r + 1) * (l + 1) / (k + 2)) / math.sin(math.pi * (l + 1) / (k + 2))


def fuse(k: int, a: int, b: int) -> set:
    return set(range(abs(a - b), min(a + b, 2 * k - a - b) + 1, 2))


def channels(k: int, ls: Sequence[int]) -> set:
    ch = {ls[0]}
    for l in ls[1:]:
        ch = set().union(*[fuse(k, c, l) for c in ch])
    return ch


def weak_fusion_ok(k: int, ls: Sequence[int]) -> bool:
    return k <= 0 or bool(channels(k, ls) & {0, k})


def strict_fusion_ok(k: int, ls: Sequence[int], special: bool) -> bool:
    return k <= 0 or (0 if special else k) in channels(k, ls)


def label_class(k: int, l: int) -> int:
    return min(l, k - l)


# ----------------------------------------------------------------------------- model data
def fermat_exponents(model: Model) -> List[int]:
    ns = []
    for i in range(model.nfields):
        n = model.fermat_exponent(i)
        if n is None:
            raise ValueError("TDL analysis needs a Fermat superpotential W = sum x_i^n_i")
        ns.append(n)
    if any(len([p for p in t.powers if p]) != 1 for t in model.superpotential):
        raise ValueError("TDL analysis needs a Fermat superpotential (no mixed terms)")
    for i, (f, n) in enumerate(zip(model.fields, ns)):
        if Fr(f.kcharge, model.korder) != Fr(1, n):
            raise ValueError(f"field {i + 1}: charge {f.kcharge}/{model.korder} is not 1/{n}")
    return ns


def _qplus_fields(sec, mono) -> Tuple[Fr, ...]:
    m = sec.model
    n, N, K = m.nfields, m.N, m.korder
    out = []
    for i, f in enumerate(m.fields):
        a = Fr(f.kcharge, K)
        v = (a - Fr(1, 2)) + ((1 - a) * sec.Bt[i] + a * sec.Ft[i]) / Fr(N)
        v += a * (mono[_B * n + i] - mono[_BB * n + i] + mono[_F * n + i] - mono[_BF * n + i])
        out.append(v)
    return tuple(out)


def _frac_mod1(x: Fr) -> Fr:
    y = x - (x.numerator // x.denominator)
    return y - 1 if y > Fr(1, 2) else y


def state_labels(sec, st) -> set:
    """Set of per-factor label vectors l_i over the terms of a massless state (one element if well defined)."""
    m = sec.model
    ls = set()
    for _, mono in st.terms:
        qp = _qplus_fields(sec, mono)
        ls.add(tuple((_frac_mod1(qp[i]) + Fr(1, 2)) * Fr(m.korder, f.kcharge) - 1
                     for i, f in enumerate(m.fields)))
    return ls


def rep_of(st) -> str:
    qm, qp = st.q_minus, st.q_plus
    if qp == Fr(-3, 2):
        return "V" if qm == 0 else f"gaugino q-={qm}"
    return {Fr(-2): "27", Fr(2): "27bar", Fr(0): "1/78"}.get(qm, f"q-={qm}")


def massless_labels(spec) -> Tuple[Counter, int]:
    """Counter over (representation, label classes) of all massless states, and the number of
    states whose labels are not well defined."""
    model = spec.model
    K = [n - 2 for n in fermat_exponents(model)]
    reps, bad = Counter(), 0
    for b in spec.bunches:
        for sec in b.sectors:
            for st in sec.states():
                if st.q_plus not in (Fr(-1, 2), Fr(-3, 2)):
                    continue
                ls = state_labels(sec, st)
                l = next(iter(ls))
                if len(ls) != 1 or any(x.denominator != 1 or not 0 <= x <= K[i] for i, x in enumerate(l)):
                    bad += 1
                    continue
                reps[(rep_of(st), tuple(label_class(K[i], int(x)) for i, x in enumerate(l)))] += 1
    return reps, bad


def marginal_monomials(model: Model, invariant: bool = True) -> List[Tuple[int, ...]]:
    """Untwisted chiral-ring monomials of weight 1 (the untwisted 27s), optionally only those
    invariant under the extra discrete orbifold generators."""
    n = fermat_exponents(model)
    w = [Fr(1, x) for x in n]
    mons = [a for a in product(*[range(x - 1) for x in n]) if sum(ai * wi for ai, wi in zip(a, w)) == 1]
    if invariant:
        mons = [a for a in mons
                if all(sum(ai * f.charges[j] for ai, f in zip(a, model.fields)) % o == 0
                       for j, o in enumerate(model.orders))]
    return mons


def line_loci(model: Model, mons) -> List[Tuple[int, int, int, int, List[int]]]:
    """(factor, k, r, number of monomials kept, allowed exponents) for every non-invertible line."""
    n = fermat_exponents(model)
    out = []
    for i, ni in enumerate(n):
        k = ni - 2
        for r in range(1, k):
            d = line_dim(k, r)
            keep = [a for a in mons if abs(abs(line_dim(k, r, a[i])) - d) < 1e-9]
            out.append((i + 1, k, r, len(keep), sorted({a[i] for a in keep})))
    return out


# ----------------------------------------------------------------------------- flat 4-point
class FlatCouplings:
    """Flat-coordinate correlators of the untwisted chiral ring of W = sum x_i^{n_i} at the
    Fermat point (normalised so that <Hessian> = 1)."""

    def __init__(self, n: Sequence[int]):
        self.n = tuple(n)
        self.r = len(n)
        self.H = tuple(x - 2 for x in n)

    def add(self, *v):
        return tuple(map(sum, zip(*v)))

    def C0(self, *ms) -> int:
        return 1 if self.add(*ms) == self.H else 0

    def target(self, i):
        return tuple(self.H[j] + (self.n[i] if j == i else 0) for j in range(self.r))

    def four_point(self, d, a, b, c) -> Fr:
        """dC_abc/dt_d at t = 0: residue term plus the contact terms of DVV."""
        n, r = self.n, self.r
        s = self.add(a, b, c, d)
        t = Fr(0)
        for i in range(r):
            if d[i] and s == self.target(i):
                t -= Fr(d[i], n[i])
        for x, y, z in ((a, b, c), (b, a, c), (c, a, b)):
            sx = self.add(d, x)
            big = [i for i in range(r) if sx[i] >= n[i] - 1]
            if big:
                i = big[0]
                e = tuple(sx[j] - (n[i] if j == i else 0) for j in range(r))
                if min(e) >= 0:
                    t += Fr(sx[i] - n[i] + 1, n[i]) * self.C0(e, y, z)
        return t

    def quadruples(self, mons):
        """Unordered quadruples allowed by all phase / U(1) symmetries of the Fermat point."""
        mset = set(mons)
        quads = set()
        for a, b, c in combinations_with_replacement(sorted(mons), 3):
            for i in range(self.r):
                d = tuple(t - x - y - z for t, x, y, z in zip(self.target(i), a, b, c))
                if d in mset:
                    quads.add(tuple(sorted((a, b, c, d))))
        return quads

    def special_factor(self, q) -> int:
        s = self.add(*q)
        return next(j for j in range(self.r) if s[j] != self.H[j])


def classify_couplings(model: Model, mons) -> dict:
    n = fermat_exponents(model)
    K = [x - 2 for x in n]
    fc = FlatCouplings(n)
    stats, bypat, asym = Counter(), Counter(), 0
    for q in fc.quadruples(mons):
        i = fc.special_factor(q)
        vals = {fc.four_point(*p) for p in set(permutations(q))}
        asym += len(vals) > 1
        nz = any(v != 0 for v in vals)
        weak = all(weak_fusion_ok(K[j], [x[j] for x in q]) for j in range(len(n)))
        strict = all(strict_fusion_ok(K[j], [x[j] for x in q], j == i) for j in range(len(n)))
        ident = any(x[i] == 0 for x in q)
        stats[(weak, strict, ident, nz)] += 1
        bypat[(i + 1, K[i], tuple(sorted(x[i] for x in q)), weak, strict, nz)] += 1
    return {"stats": stats, "bypattern": bypat, "asymmetric": asym}


# ----------------------------------------------------------------------------- abelian test
def _hnf(rows, ncol):
    basis = {}

    def eg(x, y):
        if y == 0:
            return (1, 0)
        s, t = eg(y, x % y)
        return (t, s - (x // y) * t)

    for r in rows:
        r = r[:]
        while True:
            p = next((j for j in range(ncol) if r[j]), None)
            if p is None:
                break
            if p not in basis:
                basis[p] = r if r[p] > 0 else [-x for x in r]
                break
            b = basis[p]
            if r[p] % b[p] == 0:
                q = r[p] // b[p]
                r = [x - q * y for x, y in zip(r, b)]
            else:
                g = math.gcd(b[p], r[p])
                s, t = eg(b[p], r[p])
                new = [s * x + t * y for x, y in zip(b, r)]
                rr = [(r[p] // g) * x - (b[p] // g) * y for x, y in zip(b, r)]
                basis[p] = new if new[p] > 0 else [-x for x in new]
                r = rr
    return basis


def abelian_lattice_test(model: Model, mons) -> dict:
    """Fields: the untwisted 27s and the singlet components M_(d,i) = G_i^- Phi_d.  A phase
    symmetry (any abelian R or non-R symmetry) forbids a coupling v while allowing all non-zero
    couplings iff v - u0 is outside the integer span of the non-zero couplings' differences."""
    n = fermat_exponents(model)
    fc = FlatCouplings(n)
    mset = set(mons)
    idx = {("27", a): j for j, a in enumerate(mons)}
    for d in mons:
        for i in range(fc.r):
            if d[i]:
                idx[("M", d, i)] = len(idx)
    ncol = len(idx)

    def vec(ms, extra=None):
        v = [0] * ncol
        for a in ms:
            v[idx[("27", a)]] += 1
        if extra:
            v[idx[extra]] += 1
        return v

    allowed, zero = [], []
    for a, b, c in combinations_with_replacement(sorted(mons), 3):
        if fc.add(a, b, c) == fc.H:
            allowed.append(vec((a, b, c)))
    for d in mons:
        for i in range(fc.r):
            if not d[i]:
                continue
            tgt = tuple(t - x for t, x in zip(fc.target(i), d))
            for a, b in combinations_with_replacement(sorted(mons), 2):
                c = tuple(t - x - y for t, x, y in zip(tgt, a, b))
                if c not in mset or c < b:
                    continue
                (allowed if fc.four_point(d, a, b, c) != 0 else zero).append(vec((a, b, c), ("M", d, i)))
    if not allowed:
        return {"fields": ncol, "nonzero": 0, "zero": len(zero), "rank": 0, "explainable": len(zero)}
    u0 = allowed[0]
    basis = _hnf([[x - y for x, y in zip(u, u0)] for u in allowed[1:]], ncol)

    def in_span(v):
        r = v[:]
        for p in sorted(basis):
            if r[p]:
                b = basis[p]
                if r[p] % b[p]:
                    return False
                q = r[p] // b[p]
                r = [x - q * y for x, y in zip(r, b)]
        return not any(r)

    outside = sum(1 for v in zero if not in_span([x - y for x, y in zip(v, u0)]))
    return {"fields": ncol, "nonzero": len(allowed), "zero": len(zero), "rank": len(basis), "explainable": outside}


# ----------------------------------------------------------------------------- report
def tdl_text(spec, couplings: bool = True, lattice: bool = False) -> str:
    model = spec.model
    n = fermat_exponents(model)
    K = [x - 2 for x in n]
    L = [f"TDL ANALYSIS of {model.name}   W = Fermat, n = {n}, Gepner levels k = {K}", ""]
    reps, bad = massless_labels(spec)
    L.append("A. su(2)_k labels of the massless states (classes min(l, k-l) per factor)")
    L.append(f"   states with ill-defined labels: {bad}")
    for rep in ("V", "27", "27bar", "1/78"):
        c = sorted([(p, v) for (r, p), v in reps.items() if r == rep], key=lambda t: (-t[1], t[0]))
        L.append(f"   {rep:<6} {sum(v for _, v in c):5d}   most common: {c[:6]}")
    L += ["", "B. line eigenvalues S_{r,l}/S_{0,l} (r = k is invertible)"]
    for k in sorted(set(K)):
        for r in range(1, k + 1):
            L.append(f"   k={k} r={r} d={line_dim(k, r):.4f}{' (invertible)' if r == k else ''}: "
                     f"{[round(line_dim(k, r, l), 4) for l in range(k + 1)]}")
    allm = marginal_monomials(model, invariant=False)
    mons = marginal_monomials(model)
    n27 = sum(v for (r, p), v in reps.items() if r == "27")
    L += ["", f"C. untwisted marginal monomials: {len(allm)} of W, {len(mons)} invariant under the orbifold "
              f"(27s in the full spectrum: {n27})"]
    for fct, k, r, cnt, ex in line_loci(model, mons):
        L.append(f"   factor {fct} (k={k}) line r={r} (d={line_dim(k, r):.4f}) kept by {cnt} monomials "
                 f"(x{fct} exponents {ex})")
    if couplings:
        res = classify_couplings(model, mons)
        tot = sum(res["stats"].values())
        L += ["", f"D. untwisted 27^3.1 couplings allowed by all phase/U(1) symmetries: {tot}  "
                  f"(not totally symmetric: {res['asymmetric']})"]
        for (w, s, idn, nz), v in sorted(res["stats"].items()):
            L.append(f"   weak-fusion={'ok' if w else 'forbidden':<9} strict-fusion={'ok' if s else 'forbidden':<9} "
                     f"identity-in-special-factor={'yes' if idn else 'no':<3} nonzero={'yes' if nz else 'no':<3}: {v}")
        consistent = all((s == nz) for (w, s, idn, nz) in res["stats"])
        L.append("   RESULT: " + ("a coupling vanishes iff the strict fusion rule forbids it" if consistent
                                  else "strict fusion rule and the 4-point functions DISAGREE"))
        L.append("   by special factor (factor, k, sorted exponents there, weak ok, strict ok, nonzero): count")
        for key, v in sorted(res["bypattern"].items()):
            L.append(f"      {key}: {v}")
    if lattice:
        lt = abelian_lattice_test(model, mons)
        L += ["", "E. abelian lattice test (fields: untwisted 27s and singlets M_(d,i))",
              f"   fields {lt['fields']}, non-zero couplings {lt['nonzero']}, zero couplings {lt['zero']}, "
              f"rank {lt['rank']}",
              f"   zero couplings explainable by SOME abelian symmetry: {lt['explainable']} of {lt['zero']}"]
    L += ["", "Notes: couplings are first order in the modulus at the Fermat point, untwisted 27s only.",
          "Odd-r lines need dressing by invertible phases on the other factors (Cordova-Rizi)."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- surviving lines
def moduli_labels(spec) -> Dict[str, List[Tuple[Tuple[int, ...], str]]]:
    """Per-factor labels (l_i) and sector of every 27 (complex-structure modulus) and 27bar
    (Kähler modulus, including blow-up modes from twisted sectors) of a Fermat model."""
    K = [n - 2 for n in fermat_exponents(spec.model)]
    out = {"27": [], "27bar": []}
    for b in spec.bunches:
        for sec in b.sectors:
            for st in sec.states():
                if st.q_plus != Fr(-1, 2):
                    continue
                rep = rep_of(st)
                if rep not in out:
                    continue
                ls = state_labels(sec, st)
                l = next(iter(ls))
                if len(ls) != 1 or any(x.denominator != 1 or not 0 <= x <= K[i] for i, x in enumerate(l)):
                    raise ValueError(f"ill-defined labels for a {rep} in {sec.label}")
                out[rep].append((tuple(int(x) for x in l), sec.label))
    return out


def keeps(K: Sequence[int], rs: Sequence[int], l: Sequence[int]) -> Optional[int]:
    """Action of the product line (x)_i L_{r_i} on a modulus with labels l, relative to the
    quantum dimension: +1 (kept: eigenvalue = d), -1 (eigenvalue -d: kept only after an
    invertible dressing), or None (broken, |eigenvalue| < d)."""
    sign = 0
    for k, r, x in zip(K, rs, l):
        if r == 0 or k <= 0:
            continue
        if r == k:                       # invertible (-1)^l
            sign += x
        elif x == 0:
            pass
        elif x == k:
            sign += r
        else:
            return None
    return 1 if sign % 2 == 0 else -1


def line_survival(spec) -> dict:
    """Which marginal deformations (all 27s and 27bars, untwisted and twisted) keep which lines.

    A line kept by a set of exactly marginal deformations survives the whole family they
    generate (Cordova-Rizi criterion: the deforming operator is transparent, eigenvalue = d).
    Returned:
      factors:  per factor i and line r: (# 27 kept, # 27bar kept, # 27bar kept up to sign)
      supports: for every set S of factors carrying non-invertible lines, the moduli with
                l_i in {0, k_i} on S (kept by the product of lines on S, up to an invertible
                dressing), reported for the maximal S with a given kept set
      full:     product lines kept by every Kähler / every complex-structure modulus
    """
    from itertools import combinations
    K = [n - 2 for n in fermat_exponents(spec.model)]
    ml = moduli_labels(spec)
    nl = [i for i, k in enumerate(K) if k >= 2]
    fac = []
    for i in nl:
        k = K[i]
        for r in range(1, k):
            rs = [r if j == i else 0 for j in range(len(K))]
            row = {"factor": i + 1, "k": k, "r": r, "d": line_dim(k, r)}
            for rep in ("27", "27bar"):
                v = [keeps(K, rs, l) for l, _ in ml[rep]]
                row[rep] = sum(1 for x in v if x == 1)
                row[rep + "_sign"] = sum(1 for x in v if x == -1)
            fac.append(row)

    def kept(S, rep):
        return [j for j, (l, _) in enumerate(ml[rep]) if all(l[i] in (0, K[i]) for i in S)]

    mons = marginal_monomials(spec.model)

    def free(S):
        """untwisted marginal monomials without any x_i, i in S (decoupled deformations)"""
        return sum(1 for a in mons if all(a[i] == 0 for i in S))

    sup = {}
    for n in range(1, len(nl) + 1):
        for S in combinations(nl, n):
            kk = (tuple(kept(S, "27")), tuple(kept(S, "27bar")))
            if not kk[1] and not kk[0]:
                continue
            # keep only maximal supports for a given kept set
            if kk in sup and len(sup[kk]) >= len(S):
                continue
            sup[kk] = S
    supports = []
    for (k27, kb), S in sup.items():
        if any(set(S) < set(T) and set(kb) <= set(kb2) and set(k27) <= set(k272)
               for (k272, kb2), T in sup.items()):
            continue
        supports.append({"factors": [i + 1 for i in S], "27": len(k27), "27bar": len(kb),
                         "27_l0": sum(1 for j in k27 if all(ml["27"][j][0][i] == 0 for i in S)),
                         "27bar_l0": sum(1 for j in kb if all(ml["27bar"][j][0][i] == 0 for i in S)),
                         "27_decoupled": free(S),
                         "27bar_sectors": sorted({ml["27bar"][j][1] for j in kb}),
                         "27bar_labels": sorted({ml["27bar"][j][0] for j in kb})})
    supports.sort(key=lambda s: (-s["27bar"], -len(s["factors"])))
    full = {rep: [i + 1 for i in nl if len(kept((i,), rep)) == len(ml[rep]) and ml[rep]]
            for rep in ("27", "27bar")}
    breaker = [l for l, _ in ml["27bar"] if all(l[i] not in (0, K[i]) for i in nl)]
    return {"K": K, "n27": len(ml["27"]), "n27bar": len(ml["27bar"]), "n27_untwisted": len(mons), "factors": fac,
            "supports": supports, "full": full, "kahler_breaks_all": len(breaker)}


def survival_text(spec) -> str:
    s = line_survival(spec)
    model = spec.model
    L = [f"SURVIVING LINES of {model.name}   k = {s['K']}   27s: {s['n27']}   27bars: {s['n27bar']}", "",
         "A line survives a family of exactly marginal deformations iff every deforming modulus is",
         "transparent to it (eigenvalue = quantum dimension d; Cordova-Rizi). 27s are complex-structure",
         "moduli, 27bars Kähler moduli (twisted 27bars are blow-up modes). Labels come from the right-",
         "moving R charges of every massless state, so twisted moduli are included.", "",
         "A. single non-invertible lines: # moduli kept (eigenvalue +d) / kept up to sign (-d: needs",
         "   an invertible dressing; odd-r lines need the Cordova-Rizi dressing anyway)",
         "   factor   k   r       d     27 kept (-d)   27bar kept (-d)"]
    for f in s["factors"]:
        L.append(f"   {f['factor']:>6} {f['k']:>3} {f['r']:>3} {f['d']:>7.4f}   {f['27']:>7} ({f['27_sign']:>3})   "
                 f"{f['27bar']:>9} ({f['27bar_sign']:>3})")
    L += ["", "B. products of lines on a set S of factors: moduli with l_i in {0, k_i} on S (maximal S only).",
          "   In brackets: of these, moduli with l_i = 0 on S. 'exact' = untwisted marginal monomials with",
          "   no x_i, i in S: they deform only the other LG factors, so the lines on S (with their",
          "   dressing inside factor i) survive the whole finite family, to all orders.",
          "   S                       27 kept (l=0) exact   27bar kept (l=0)   sectors of the kept 27bars"]
    for x in s["supports"][:40]:
        L.append(f"   {str(x['factors']):<22} {x['27']:>8} ({x['27_l0']:>4}) {x['27_decoupled']:>5}"
                 f"   {x['27bar']:>10} ({x['27bar_l0']:>4})   {', '.join(x['27bar_sectors'][:6])}"
                 + (" ..." if len(x["27bar_sectors"]) > 6 else ""))
    if len(s["supports"]) > 40:
        L.append(f"   ... {len(s['supports']) - 40} more")
    L += ["", "C. summary",
          f"   Kähler moduli that break every non-invertible line: {s['kahler_breaks_all']} of {s['n27bar']}",
          f"   factors whose lines are kept by ALL Kähler moduli: {s['full']['27bar'] or 'none'}",
          f"   factors whose lines are kept by ALL complex-structure moduli: {s['full']['27'] or 'none'}",
          "   A line kept by a set of Kähler moduli survives the limit where only those are large",
          "   (e.g. blow-up modes large, the other Kähler moduli at their Gepner values).",
          "   For Kähler moduli run 'lines' on the mirror: kept 27bars that are untwisted monomials",
          "   there without the x_i of S are exact (decoupled) families on the mirror side."]
    return "\n".join(L) + "\n"
