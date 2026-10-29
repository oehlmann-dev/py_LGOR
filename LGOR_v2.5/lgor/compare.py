"""Compare the massless spectra of two models, e.g. a Fermat point and a deformation.

States are compared class by class, using the quantum numbers that are preserved
by the deformation: the sector, q-, q+, the R-charge, the discrete charges, and
the charges under the discrete remnants of broken U(1)s (computed with the
remnant generators of the deformed model, applied to the states of both models).

Within a class, the deformed model's states are projected onto the span of the
reference model's states (in the oscillator-monomial basis):

* surviving: the image of that projection;
* lost:      reference states orthogonal to everything that survives (e.g. the
             Goldstone bosons eaten by broken U(1)s, the broken gauginos);
* new:       deformed-model states with no component along the reference states.

Representatives of surviving states may still differ by terms that were
cohomologically trivial in the reference model.
"""

from __future__ import annotations

import datetime
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import linalg
from .spectrum import ModelSpectrum
from .symmetries import StateInfo, SymmetryAnalysis, _dot, _mod, fmt


# ------------------------------------------------------------------ rational linear algebra
def _rref_solve(G: List[List[Fraction]], rhs: List[Fraction]) -> List[Fraction]:
    n = len(G)
    M = [list(map(Fraction, row)) + [Fraction(r)] for row, r in zip(G, rhs)]
    for c in range(n):
        p = next(i for i in range(c, n) if M[i][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [M[i][n] for i in range(n)]


def _int_rows(vectors: List[List[Fraction]]) -> List[List[int]]:
    out = []
    for v in vectors:
        d = 1
        for x in v:
            d = d * Fraction(x).denominator // __import__("math").gcd(d, Fraction(x).denominator)
        out.append(linalg.primitive([int(x * d) for x in v]))
    return out


def _rank(vectors) -> int:
    if not vectors:
        return 0
    rows = _int_rows(vectors)
    return linalg.rank(rows, len(rows[0]))


class ClassDiff:
    def __init__(self, key, A: List[StateInfo], B: List[StateInfo]):
        self.key, self.A, self.B = key, A, B
        monos = sorted({m for s in A + B for _, m in s.state.terms})
        idx = {m: i for i, m in enumerate(monos)}

        def vec(s):
            v = [Fraction(0)] * len(monos)
            for c, m in s.state.terms:
                v[idx[m]] += c
            return v
        Av = [vec(s) for s in A]
        Bv = [vec(s) for s in B]
        k = len(Av)
        self.surviving: List[List[Fraction]] = []   # coefficient vectors over A
        self.lost: List[List[int]] = []
        self.new: List[List[int]] = []               # coefficient vectors over B
        if k == 0:
            self.new = [[int(i == j) for i in range(len(Bv))] for j in range(len(Bv))]
            return
        G = [[sum(a * b for a, b in zip(Av[i], Av[j])) for j in range(k)] for i in range(k)]
        self.G = G
        Y = []
        for b in Bv:
            rhs = [sum(x * y for x, y in zip(a, b)) for a in Av]
            Y.append(_rref_solve(G, rhs))
        # surviving subspace (in A-coefficients), independent rows
        surv = []
        for y in Y:
            if any(y) and _rank(surv + [y]) > len(surv):
                surv.append(y)
        self.surviving = surv
        # lost: z with z^T G y = 0 for all surviving y
        if surv:
            rows = [[sum(G[i][j] * y[j] for j in range(k)) for i in range(k)] for y in surv]
            self.lost = linalg.kernel(_int_rows(rows), k)
        else:
            self.lost = [[int(i == j) for i in range(k)] for j in range(k)]
        # new: combinations x of B states with A (B^T x) = 0
        if Bv:
            AB = [[sum(a * b for a, b in zip(Av[i], Bv[j])) for j in range(len(Bv))] for i in range(k)]
            self.new = linalg.kernel(_int_rows(AB), len(Bv)) if any(any(r) for r in AB) else \
                [[int(i == j) for i in range(len(Bv))] for j in range(len(Bv))]

    @property
    def changed(self) -> bool:
        return bool(self.lost or self.new or len(self.A) != len(self.B))


def _class_key(s: StateInfo, remnants, use_u1: bool, symR=None):
    """Preserved quantum numbers of a state.  R and discrete charges are evaluated with
    the generators of ``symR`` (the deformed model), which are symmetries of both models."""
    rem = tuple(_mod(_dot(c, q), k) for _, c, k in remnants for q in [s.q])
    if symR is not None and symR.n_R is not None:
        R = _mod(3 * s.sector.k0 - 2 * s.state.q_minus + _dot(symR.n_R, s.q), symR.M)
        disc = tuple(None if nj is None else _mod(s.sector.twist[j] + _dot(nj, s.q), o)
                     for j, (o, nj) in enumerate(zip(symR.model.orders, symR.n_disc)))
    else:
        R, disc = s.R, tuple(s.discrete)
    return (s.bunch_twist, s.sector.k0, s.state.q_minus, s.state.q_plus, R, disc, rem,
            tuple(s.u1) if use_u1 else ())


def compare(specA: ModelSpectrum, symA: SymmetryAnalysis, specB: ModelSpectrum, symB: SymmetryAnalysis,
            nameA: str = None, nameB: str = None, style: str = "modes") -> str:
    nameA = nameA or specA.model.name
    nameB = nameB or specB.model.name
    mA, mB = specA.model, specB.model
    if (mA.nfields, mA.korder, mA.orders) != (mB.nfields, mB.korder, mB.orders):
        raise ValueError("the two models must have the same fields, KOrder and discrete symmetries")
    remnants = symB.remnants or symA.remnants
    use_u1 = [list(c) for c in symA.U1] == [list(c) for c in symB.U1] and bool(symA.U1)
    classes: Dict[tuple, Tuple[list, list]] = OrderedDict()
    for tag, sym in (("A", symA), ("B", symB)):
        for s in sym.states:
            if s.state.q_plus > 0:
                continue
            key = _class_key(s, remnants, use_u1, symB)
            classes.setdefault(key, ([], []))[0 if tag == "A" else 1].append(s)
    diffs = [ClassDiff(k, a, b) for k, (a, b) in classes.items()]

    out = []
    bar = "=" * 100
    tA, tB = specA.total, specB.total
    out.append(f"LGOR comparison:  A = {nameA}   versus   B = {nameB}")
    out.append(f"(LGOR Python, {datetime.date.today()})")
    out.append("")

    def W(m):
        return [tuple(t.powers) + (t.coeff,) for t in m.superpotential]
    from .report import _poly_W
    out.append(f"A:  W = {_poly_W(mA)}")
    out.append(f"B:  W = {_poly_W(mB)}")
    out.append("")
    out.append("4D SPECTRUM                      A        B     B - A")

    def row(label, a, b):
        out.append(f"  {label:<28}{a:>5}    {b:>5}    {b - a:+5d}")
    nvA = len(symA.extra_vectors) - 1
    nvB = len(symB.extra_vectors) - 1
    row("E6 adjoint vectors", tA.n78[0], tB.n78[0])
    row("extra vectors", nvA, nvB)
    row("27", tA.n27[1], tB.n27[1])
    row("27bar", tA.n27bar[1], tB.n27bar[1])
    row("chiral 78", tA.n78[1], tB.n78[1])
    row("E6 singlets", tA.singlets[1], tB.singlets[1])
    out.append(f"  extra gauge group:  A: {symA.gauge_group}    B: {symB.gauge_group}")
    if symB.remnants:
        out.append(f"  B keeps the discrete remnants " + " x ".join(f"Z{k}" for _, _, k in symB.remnants)
                   + " of the broken U(1)s")
    out.append("")
    out.append("SYMMETRIES")
    out.append("  A:")
    out.extend("    " + l for l in symA.describe().splitlines())
    out.append("  B:")
    out.extend("    " + l for l in symB.describe().splitlines())
    out.append("")
    labels = []
    if remnants:
        labels.append("remnant charges (" + ", ".join(n for n, _, _ in remnants) + ")")
    out.append("COMPARISON BY CLASS of preserved quantum numbers: sector, q-, q+, R"
               + (", discrete charges" if mA.orders else "") + (", " + labels[0] if labels else "")
               + (", U(1) charges" if use_u1 else ""))
    out.append("  (q+ = +1/2, +3/2 are CPT conjugates and not listed)")
    nchanged = sum(d.changed for d in diffs)
    out.append(f"  {len(diffs)} classes, {nchanged} with differences; all other classes have the same number of states.")
    out.append("")

    def charges_A(s: StateInfo) -> str:
        parts = []
        if symA.U1:
            parts.append("U(1)=(" + ", ".join(fmt(x) for x in s.u1) + ")")
        if s.R is not None:
            parts.append(f"R={fmt(s.R_superfield if s.state.q_plus == Fraction(-1, 2) else s.R)}")
        return " ".join(parts)

    def poly_of(states: List[StateInfo], coeffs) -> str:
        terms = [(c, st) for c, st in zip(coeffs, states) if c]
        return " ".join(f"{'+' if c > 0 else '-'} {abs(c) if abs(c) != 1 else ''}[{i}]".replace("  ", " ")
                        for i, (c, st) in enumerate(terms, 1)), terms

    for d in diffs:
        if not d.changed:
            continue
        twist, k0, qm, qp, R, disc, rem, u1 = d.key
        sec = d.A[0].sector.label if d.A else d.B[0].sector.label
        kind = "VECTORS (gauginos)" if qp == Fraction(-3, 2) else "CHIRAL MULTIPLETS"
        from .report import classify
        first = d.A[0] if d.A else d.B[0]
        rep = classify(first, symA if d.A else symB)[1]
        if rep in ("Cartan gaugino",):
            rep = "Cartan gauginos (q- = 0)"
        out.append(bar)
        if R is None:
            Rtxt = "R = ?"
        elif qp == Fraction(-1, 2):
            Rtxt = f"R(Phi) = {fmt(_mod(R + symA.R_THETA, symA.M))}"
        else:
            Rtxt = f"R(gaugino) = {fmt(R)}"
        desc = f"{kind}, {rep}, sector {sec}, q- = {fmt(qm)}, q+ = {fmt(qp)}, {Rtxt}"
        if rem:
            desc += ", remnant charges (" + ",".join(fmt(x) for x in rem) + ")"
        out.append(desc)
        out.append(f"  A: {len(d.A)} states    B: {len(d.B)} states    surviving: {len(d.surviving)}"
                   f"    lost: {len(d.lost)}    new: {len(d.new)}")
        out.append(bar)
        out.append("  states of A in this class (charges in A):")
        for i, s in enumerate(d.A, 1):
            out.append(f"    [{i}] {charges_A(s):<34} {s.state.to_string(style)}")
        out.append("  states of B in this class:")
        for i, s in enumerate(d.B, 1):
            out.append(f"    <{i}> {s.state.to_string(style)}")
        if d.surviving:
            out.append("  surviving (projection of B onto A, in terms of the A states [i]):")
            for y in _int_rows(d.surviving):
                out.append("    " + " ".join(f"{'+' if c > 0 else '-'}{abs(c) if abs(c) != 1 else ''}[{i}]"
                                             for i, c in enumerate(y, 1) if c))
        # express the lost space through the broken U(1)s of A, if possible
        labelled = []
        if d.lost and symA.U1:
            cand = []
            if qp == Fraction(-3, 2):
                # Cartan gaugino of field i <-> q^i ; gaugino of U1_a = sum_i (coefficient of q^i) [state of field i]
                n = mA.nfields
                fld = []
                for st in d.A:
                    fs = {i for _, m in st.state.terms for i in range(n)
                          if any(m[kind * n + i] for kind in range(4))}
                    fld.append(fs.pop() if len(fs) == 1 else None)
                if None not in fld:
                    for name, c in zip(symA.u1_names, symA.U1):
                        cand.append((f"gaugino of {name}", [c[f] for f in fld]))
            else:
                for name, a in zip(symA.u1_names, range(len(symA.U1))):
                    cand.append((f"Goldstone eaten by {name}", [st.u1[a] for st in d.A]))
            k = len(d.A)
            ok = []
            for name, g in cand:
                if not any(g):
                    continue
                if all(sum(g[i] * d.G[i][j] * y[j] for i in range(k) for j in range(k)) == 0 for y in d.surviving):
                    ok.append((name, g))
            if ok and _rank([g for _, g in ok]) == len(d.lost) == len(ok):
                labelled = ok
        if labelled:
            out.append("  LOST, in terms of the broken U(1)s of A:")
            for name, g in labelled:
                gi = _int_rows([g])[0]
                combo = " ".join(f"{'+' if c > 0 else '-'}{abs(c) if abs(c) != 1 else ''}[{i}]"
                                 for i, c in enumerate(gi, 1) if c)
                out.append(f"    {combo:<40} {name}")
        elif d.lost:
            out.append("  LOST (combinations of A states orthogonal to everything that survives):")
            for z in d.lost:
                combo = " ".join(f"{'+' if c > 0 else '-'}{abs(c) if abs(c) != 1 else ''}[{i}]"
                                 for i, c in enumerate(z, 1) if c)
                u1s = {tuple(d.A[i].u1) for i, c in enumerate(z) if c}
                note = ""
                if symA.U1:
                    if len(u1s) == 1:
                        note = "   U(1) charges " + "(" + ", ".join(fmt(x) for x in u1s.pop()) + ")"
                    else:
                        note = "   (mixes states of different U(1) charge)"
                out.append(f"    {combo}{note}")
        if d.new:
            out.append("  NEW (combinations of B states with no component along A):")
            for z in d.new:
                out.append("    " + " ".join(f"{'+' if c > 0 else '-'}{abs(c) if abs(c) != 1 else ''}<{i}>"
                                             for i, c in enumerate(z, 1) if c))
        out.append("")

    # interpretation for the typical Higgsing pattern
    lost_vec = sum(len(d.lost) for d in diffs if d.key[3] == Fraction(-3, 2))
    lost_chi = sum(len(d.lost) for d in diffs if d.key[3] == Fraction(-1, 2))
    new_all = sum(len(d.new) for d in diffs)
    out.append("INTERPRETATION")
    out.append(f"  {lost_vec} vector multiplets and {lost_chi} chiral multiplets of A are lost, {new_all} states are new.")
    if lost_vec and lost_vec == lost_chi and not new_all:
        out.append("  This is the Higgs mechanism: every broken gauge boson eats one chiral multiplet. The lost chiral")
        out.append("  states are the Goldstone directions; the surviving combination in their class is the modulus")
        out.append("  that parametrises the deformation.")
    w = symA.warnings + symB.warnings
    if w:
        out.append("")
        out.append("WARNINGS")
        out.extend("  " + x for x in w)
    return "\n".join(out) + "\n"
