"""Non-Fermat deformations and their 4D interpretation.

A deformation W -> W + dW of the Landau-Ginzburg superpotential is a marginal
(complex structure) deformation of the worldsheet theory.  In 4D it is the VEV of
the E6-singlet modulus associated with dW, whose worldsheet state in the first
twisted sector T(1; 0..0) is

      sum_i  psi^i_{-(1-alpha_i/2)}  d_i(dW)(phi_{-alpha/2})  |1; 0..0>

(the G_{-1/2} descendant of the chiral primary dW).  This module

1. builds the deformed model and computes it (spectrum, symmetries),
2. compares it with the Fermat model (lost / surviving states, broken symmetries),
3. expresses that worldsheet state in the basis of massless 4D singlets of the
   Fermat model: this is the proposed VEV configuration,
4. checks D-flatness exactly for the extra U(1)s / Cartan generators,
5. checks F-flatness: (a) at trilinear order with all symmetry-allowed couplings,
   (b) to all orders with a symmetry argument (no superpotential term linear in a
   field and otherwise built from VEV fields is allowed), and an explicit scan of
   such terms up to a given order,
6. predicts the unbroken gauge group, the discrete remnants and the R-symmetry
   from the VEVs (4D Higgs mechanism) and compares with the worldsheet result.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from fractions import Fraction
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg
from .model import Model, SuperpotentialTerm
from .monomial import B, F
from .spectrum import ModelSpectrum, compute_model
from .symmetries import SymmetryAnalysis, _dot, _mod, fmt


# ------------------------------------------------------------------ parsing
def parse_terms(model: Model, text: str) -> List[SuperpotentialTerm]:
    """Parse e.g. '5 x1 x2 x3 x4 x5 + x1^3 x2^2' or '5*x1*x2*x3*x4*x5'.

    Fields are written x<name> with the field names of the model file.
    Coefficients must be integers (as in .lgo files).
    """
    names = {f.name: i for i, f in enumerate(model.fields)}
    terms = []
    text = text.replace("*", " ").replace("-", "+-")
    for chunk in [c.strip() for c in text.split("+") if c.strip()]:
        coeff = 1
        powers = [0] * model.nfields
        m = re.match(r"^(-?\d+)\s*(.*)$", chunk)
        if m and m.group(2) and not m.group(1).startswith("x"):
            coeff, chunk = int(m.group(1)), m.group(2)
        elif chunk.startswith("-"):
            coeff, chunk = -1, chunk[1:]
        found = re.findall(r"x([A-Za-z0-9_]+?)(?:\^(\d+))?(?=\s|$|x)", chunk.replace(" ", " "))
        rest = re.sub(r"x([A-Za-z0-9_]+?)(?:\^(\d+))?(?=\s|$|x)", "", chunk).strip()
        if rest:
            raise ValueError(f"cannot read '{rest}' in term '{chunk}' (write fields as x1, x2^3, ...)")
        if not found:
            raise ValueError(f"no fields in term '{chunk}'")
        for name, p in found:
            if name not in names:
                raise ValueError(f"unknown field x{name}; fields are " + ", ".join("x" + n for n in names))
            powers[names[name]] += int(p) if p else 1
        terms.append(SuperpotentialTerm(coeff, powers))
    return terms


def term_string(model: Model, t: SuperpotentialTerm) -> str:
    mono = " ".join(f"x{model.fields[i].name}" + (f"^{p}" if p > 1 else "") for i, p in enumerate(t.powers) if p)
    return (f"{t.coeff} " if t.coeff != 1 else "") + mono


def deformed_model(model: Model, terms: Sequence[SuperpotentialTerm], name: Optional[str] = None) -> Model:
    new = Model(name or (model.name + "_deformed"), model.korder, list(model.orders),
                [type(f)(f.name, f.kcharge, list(f.charges)) for f in model.fields],
                [SuperpotentialTerm(t.coeff, list(t.powers)) for t in model.superpotential] +
                [SuperpotentialTerm(t.coeff, list(t.powers)) for t in terms])
    problems = new.check()
    if problems:
        raise ValueError("the deformation is not allowed: " + "; ".join(problems))
    return new


# ------------------------------------------------------------------ VEV proposal
class VevProposal:
    def __init__(self, specF: ModelSpectrum, symF: SymmetryAnalysis, terms: Sequence[SuperpotentialTerm],
                 symD: Optional[SymmetryAnalysis] = None, max_order: int = 10):
        from .report import build_multiplets
        self.specF, self.symF, self.symD = specF, symF, symD
        self.model = specF.model
        self.terms = list(terms)
        self.max_order = max_order
        self.notes: List[str] = []
        mults, _ = build_multiplets(symF)
        self.labels = {}
        for rep in ("27", "27bar"):
            for i, e in enumerate(mults[rep], 1):
                self.labels[id(e[0][1].state.terms)] = f"{rep}-{i:03d}"
        for i, s in enumerate(mults["singlets"], 1):
            self.labels[id(s.state.terms)] = f"1-{i:03d}"
        self.singlets = mults["singlets"]
        self._find_vevs()
        if self.vevs:
            self._d_flatness()
            self._f_flatness()
            self._predictions()

    # worldsheet state of the modulus -------------------------------------------
    def _find_vevs(self):
        n = self.model.nfields
        twist0 = tuple([0] * self.model.ndiscrete)
        sec = self.specF.sector(1, twist0)
        target: Dict[tuple, Fraction] = defaultdict(Fraction)
        for t in self.terms:
            for i in range(n):
                if t.powers[i] > 0:
                    v = [0] * (4 * n)
                    for j in range(n):
                        v[B * n + j] = t.powers[j]
                    v[B * n + i] -= 1
                    v[F * n + i] += 1
                    target[tuple(v)] += t.coeff * t.powers[i]
        target = {m: c for m, c in target.items() if c}
        self.target_text = sec.format_poly([(int(c), m) for m, c in sorted(target.items())], "modes")
        # locate the complex / q+ space and the Fermat states there
        allc = [s for s in self.symF.states
                if s.sector is sec and s.state.q_plus == Fraction(-1, 2) and s.state.q_minus == 0]
        # only states connected to the target through shared monomials matter for the projection
        support = set(target)
        candidates, rest = [], list(allc)
        changed = True
        while changed:
            changed = False
            keep = []
            for st in rest:
                ms = {m for _, m in st.state.terms}
                if ms & support:
                    candidates.append(st)
                    support |= ms
                    changed = True
                else:
                    keep.append(st)
            rest = keep
        monos = sorted({m for s in candidates for _, m in s.state.terms} | set(target))
        idx = {m: i for i, m in enumerate(monos)}

        def vec(terms):
            v = [Fraction(0)] * len(monos)
            for c, m in terms:
                v[idx[m]] += c
            return v
        A = [vec(s.state.terms) for s in candidates]
        t = vec([(c, m) for m, c in target.items()])
        self.vevs: List[Tuple[str, Fraction, object]] = []
        if not A:
            self.notes.append("no massless singlets in T(1) of the Fermat model: no VEV proposal possible")
            return
        k = len(A)
        G = [[sum(a * b for a, b in zip(A[i], A[j])) for j in range(k)] for i in range(k)]
        rhs = [sum(a * b for a, b in zip(Ai, t)) for Ai in A]
        from .compare import _rref_solve
        y = _rref_solve(G, rhs)
        resid = [ti - sum(y[i] * A[i][j] for i in range(k)) for j, ti in enumerate(t)]
        self.residual_norm = sum(r * r for r in resid)
        self.target_norm = sum(x * x for x in t)
        for yi, s in zip(y, candidates):
            if yi:
                self.vevs.append((self.labels.get(id(s.state.terms), "?"), yi, s))
        if self.residual_norm:
            frac = self.residual_norm / self.target_norm
            self.notes.append(f"the modulus state is not entirely in the span of the Fermat singlets "
                              f"(relative residual {float(frac):.3g}); the residual is Qbar-exact or massive")

    # D-flatness -----------------------------------------------------------------
    def _d_flatness(self):
        sym = self.symF
        self.D = [sum(y * y * Fraction(s.u1[a]) for _, y, s in self.vevs) for a in range(len(sym.U1))]
        self.d_flat = all(d == 0 for d in self.D)
        # gauge invariant holomorphic monomial built from the VEV fields (D-flat direction)
        rows = [[int(s.u1[a]) for _, _, s in self.vevs] for a in range(len(sym.U1))]
        self.invariant = None
        if rows:
            ker = linalg.kernel(rows, len(self.vevs))
            pos = [v for v in ker if all(x > 0 for x in v)]
            if not pos and ker:
                tot = [sum(v[i] for v in ker) for i in range(len(self.vevs))]
                if all(x > 0 for x in tot):
                    pos = [tot]
            if pos:
                self.invariant = pos[0]
        if self.symF.factors:
            self.notes.append("D-flatness of the non-Abelian extra factors is checked for their Cartan generators only")

    # F-flatness -----------------------------------------------------------------
    def _charges(self, s):
        return (tuple(Fraction(x) for x in s.u1), Fraction(s.R_superfield), tuple(s.discrete), tuple(s.remnant))

    def _f_flatness(self):
        sym = self.symF
        M = sym.M
        RW = 2 * sym.R_THETA
        vch = [self._charges(s) for _, _, s in self.vevs]
        vlabels = [l for l, _, _ in self.vevs]
        # (a) trilinear couplings with >= 2 VEV legs
        from .yukawa import YukawaAnalysis
        self.yuk = YukawaAnalysis(sym)
        vset = set(vlabels)
        self.dangerous_cubic = []
        for name, res in self.yuk.results.items():
            for t in res["allowed"]:
                nv = sum(f.label in vset for f in t)
                if nv >= 2:
                    self.dangerous_cubic.append(tuple(f.label for f in t))
        # (b) R-symmetry argument to all orders
        rs = [c[1] for c in vch]
        g = M
        for r in rs:
            g = gcd(g, int(r) % M)
        reach = {_mod(RW - m * g, M) for m in range(M // g + 1)} if g else {_mod(RW, M)}
        rset = {Fraction(s.R_superfield) for s in self.singlets}
        self.R_allows = sorted(reach & rset)
        self.f_flat_all_orders_by_R = not self.R_allows
        # (c) explicit scan: X * prod v^n, X any E6 singlet, 2 <= sum n <= max_order
        by_charge = defaultdict(list)
        for i, s in enumerate(self.singlets, 1):
            by_charge[self._charges(s)].append(f"1-{i:03d}")
        k = len(vch)
        r = len(vch[0][0])
        nd = len(vch[0][2])
        nr = len(vch[0][3])
        orders = list(self.model.orders)
        rem = [kk for _, _, kk in sym.remnants]
        self.dangerous_scan = Counter()
        self.dangerous_examples = {}
        for total in range(2, self.max_order + 1):
            for comb in itertools.combinations_with_replacement(range(k), total):
                u = tuple(-sum(vch[j][0][a] for j in comb) for a in range(r))
                R = _mod(RW - sum(vch[j][1] for j in comb), M)
                d = tuple((-sum(vch[j][2][a] for j in comb)) % orders[a] for a in range(nd))
                e = tuple((-sum(vch[j][3][a] for j in comb)) % rem[a] for a in range(nr))
                for key, labs in by_charge.items():
                    if key[0] == u and key[1] == R and tuple(x % o for x, o in zip(key[2], orders)) == d \
                            and tuple(x % o for x, o in zip(key[3], rem)) == e:
                        self.dangerous_scan[total + 1] += len(labs)
                        self.dangerous_examples.setdefault(total + 1, (labs[0], [vlabels[j] for j in comb]))
        self.f_flat = not self.dangerous_cubic and (self.f_flat_all_orders_by_R or not self.dangerous_scan)

    # 4D predictions ---------------------------------------------------------------
    def _predictions(self):
        sym = self.symF
        r = len(sym.U1)
        V = [[int(Fraction(s.u1[a])) for a in range(r)] for _, _, s in self.vevs]
        self.broken_rank = linalg.rank(V, r) if r else 0
        self.unbroken_cartans = r - self.broken_rank
        # roots: unbroken iff orthogonal to every VEV weight (metric induced by q-space)
        self.unbroken_roots = []
        self.n_roots = sum(len(f.roots) for f in sym.factors)
        if sym.factors:
            from .compare import _rref_solve
            G = [[_dot(a, b) for b in sym.U1] for a in sym.U1]
            ws = [[Fraction(x) for x in s.u1] for _, _, s in self.vevs]
            Gw = [_rref_solve(G, w) for w in ws]               # G^-1 w
            for f in sym.factors:
                for root in f.roots:
                    if all(sum(Fraction(x) * y for x, y in zip(root, gw)) == 0 for gw in Gw):
                        self.unbroken_roots.append(root)
        self.pred_vectors = len(self.unbroken_roots) + self.unbroken_cartans
        self.pred_lost = (r + self.n_roots) - self.pred_vectors
        # name of the unbroken group
        rs = set(self.unbroken_roots)
        parent = list(range(len(self.unbroken_roots)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        roots = self.unbroken_roots
        for i, a in enumerate(roots):
            for j in range(i):
                b = roots[j]
                if tuple(x + y for x, y in zip(a, b)) in rs or tuple(x - y for x, y in zip(a, b)) in rs:
                    parent[find(i)] = find(j)
        comps = defaultdict(list)
        for i, a in enumerate(roots):
            comps[find(i)].append(a)
        from .symmetries import _name_simple, _rational_rank
        names, rk = [], 0
        for c in comps.values():
            k = _rational_rank(c)
            rk += k
            names.append(_name_simple(len(c), k))
        ab = self.unbroken_cartans - rk
        if ab > 0:
            names.append("U(1)" + (f"^{ab}" if ab > 1 else ""))
        nonab = sorted([x for x in names if not x.startswith("U(1)")], key=lambda x: x)
        self.pred_group = " x ".join(nonab + [x for x in names if x.startswith("U(1)")]) or "none"
        # masses from trilinear couplings with one VEV leg:  W ⊃ y <v> X Y
        self.mass_rank = {}
        import random
        rng = random.Random(12345)
        P = 2 ** 61 - 1
        vval = {lab: int(y.numerator) * pow(int(y.denominator), -1, P) % P for lab, y, _ in self.vevs}
        for cat, key in (("1.1.1 (singlets only)", "singlets"), ("27.27bar.1", "27x27bar")):
            res = self.yuk.results.get(cat)
            if not res:
                continue
            entries = defaultdict(int)
            for t in res["allowed"]:
                labs = [f.label for f in t]
                nv = sum(l in vval for l in labs)
                if nv != 1:
                    continue
                v = next(l for l in labs if l in vval)
                rest = [l for l in labs if l not in vval]
                c = rng.randrange(1, P) * vval[v] % P
                if key == "singlets":
                    x, y = rest
                    if x == y:
                        entries[(x, x)] = (entries[(x, x)] + 2 * c) % P
                    else:
                        entries[(x, y)] = (entries[(x, y)] + c) % P
                        entries[(y, x)] = (entries[(y, x)] + c) % P
                else:
                    x = next(l for l in rest if l.startswith("27-"))
                    y = next(l for l in rest if l.startswith("27bar"))
                    entries[(x, y)] = (entries[(x, y)] + c) % P
            rows = sorted({a for a, _ in entries})
            cols = sorted({b for _, b in entries})
            M = [[entries.get((a, b), 0) for b in cols] for a in rows]
            self.mass_rank[key] = _rank_mod(M, P)
        # discrete remnant (basis independent, modulo unbroken continuous rotations)
        self.pred_remnant = sym.residual_group([s.q for _, _, s in self.vevs])
        self.pred_remnant_order = 1
        for d in self.pred_remnant:
            self.pred_remnant_order *= d
        self.ws_remnant = None
        if self.symD is not None:
            K = self.model.korder
            vs = set()
            for term in self.symD.model.superpotential:
                for k in range(self.model.nfields):
                    if term.powers[k] > 0:
                        v = tuple(Fraction(term.powers[i] * self.model.fields[i].kcharge, K) - (1 if i == k else 0)
                                  for i in range(self.model.nfields))
                        if any(v):
                            vs.add(v)
            self.ws_remnant = self.symD.residual_group(sorted(vs))
        # R-symmetry: is there an R-symmetry (R mixed with gauge rotations) leaving the VEVs neutral?
        vstates = [s for _, _, s in self.vevs]
        self.R_mixing = sym.r_mixing_for(vstates)
        self.R_preserved = self.R_mixing is not None
        M = sym.M
        g = M
        for st in vstates:
            g = gcd(g, int(_mod(st.R_superfield, M)))
        self.R_remaining = M if self.R_preserved else g

    # ------------------------------------------------------------------ output
    def text(self, compare_stats: Optional[dict] = None) -> str:
        sym = self.symF
        out = ["PROPOSED VEV CONFIGURATION (in the Fermat model)"]
        out.append("  deformation: dW = " + " + ".join(term_string(self.model, t) for t in self.terms))
        out.append("  worldsheet modulus  sum_i psi_i d_i(dW) |1>  =  " + self.target_text)
        if not self.vevs:
            out.extend("  " + n for n in self.notes)
            return "\n".join(out) + "\n"
        g = 0
        for _, y, _ in self.vevs:
            g = gcd(g, int(y.numerator)) if y.denominator == 1 else 0
        out.append(f"  = linear combination of {len(self.vevs)} massless 4D singlets of the Fermat model:")
        for lab, y, s in self.vevs:
            ch = "(" + ", ".join(fmt(x) for x in s.u1) + ")" if sym.U1 else ""
            out.append(f"     <{lab}> ~ {fmt(y):>5}    U(1)={ch}  R(Phi)={fmt(s.R_superfield)}   {s.state.to_string()}")
        out.append("  i.e. the deformation parameter is the VEV  <Phi> = epsilon * (coefficients above).")
        out.append("")
        out.append("  D-FLATNESS (extra U(1)s / Cartans):  D_a = sum_i Q_a(i) |<Phi_i>|^2")
        if sym.U1:
            out.append("    " + ", ".join(f"D[{n}] = {fmt(d)}" for n, d in zip(sym.u1_names, self.D)))
            out.append("    -> " + ("D-flat" if self.d_flat else "NOT D-flat with these magnitudes"))
            if self.invariant:
                mon = " ".join(f"{l}" + (f"^{p}" if p > 1 else "") for (l, _, _), p in zip(self.vevs, self.invariant))
                out.append(f"    gauge-invariant holomorphic monomial of the VEV fields: {mon}  (D-flat direction)")
        else:
            out.append("    no extra gauge symmetry: trivially D-flat")
        out.append("")
        out.append("  F-FLATNESS:  F_X = dW/dX = 0 at the VEV for every chiral field X")
        out.append(f"    (a) trilinear couplings with two or more VEV legs: "
                   + ("none" if not self.dangerous_cubic else f"{len(self.dangerous_cubic)}: " +
                      ", ".join(" ".join(t) for t in self.dangerous_cubic[:5])))
        RW = 2 * sym.R_THETA
        if self.f_flat_all_orders_by_R:
            out.append(f"    (b) all orders: every term X * <Phi>^n has R = R(X) + sum R(<Phi>) = {RW} mod {sym.M} only if")
            out.append(f"        R(X) is one of the values reachable from R(W) by the VEV R-charges; no E6 singlet has such")
            out.append(f"        an R-charge -> no such term is allowed -> F-flat to all orders (exact, by the R-symmetry)")
        else:
            out.append(f"    (b) the R-symmetry alone does not forbid X * <Phi>^n (singlets with R in {[fmt(x) for x in self.R_allows]})")
        if self.dangerous_scan:
            out.append(f"    (c) explicit scan up to order {self.max_order + 1}: allowed terms X*<Phi>^n by order: "
                       + ", ".join(f"order {k}: {v}" for k, v in sorted(self.dangerous_scan.items())))
            for k, (x, vs) in sorted(self.dangerous_examples.items())[:3]:
                out.append(f"        e.g. {x} * " + " ".join(vs))
        else:
            out.append(f"    (c) explicit scan up to order {self.max_order + 1}: no allowed term X*<Phi>^n")
        out.append("    -> " + ("F-flat" if self.f_flat else "F-flatness NOT guaranteed by the symmetries"))
        out.append("")
        out.append("  4D HIGGS MECHANISM: prediction from the VEVs vs. worldsheet result of the deformed model")
        D = self.symD

        def row(lab, p, w):
            ok = "" if w is None else ("ok" if str(p) == str(w) else "MISMATCH")
            out.append(f"    {lab:<46} predicted {str(p):<22}" + ("" if w is None else f" worldsheet {str(w):<22} {ok}"))
        nvecD = None if D is None else len(D.extra_vectors) - 1
        row("unbroken extra gauge group", self.pred_group, None if D is None else D.gauge_group)
        row("extra vector multiplets", self.pred_vectors, nvecD)
        row("broken generators (= eaten singlets)", self.pred_lost,
            None if D is None else (len(sym.extra_vectors) - 1) - nvecD)
        ms = self.mass_rank.get("singlets", 0)
        m27 = self.mass_rank.get("27x27bar", 0)
        out.append(f"    {'mass terms <v> X Y (generic rank)':<46} singlets: {ms}   27 x 27bar pairs: {m27}")
        if compare_stats:
            row("lost vectors (diff)", self.pred_lost, compare_stats['lost_vectors'])
            pred = self.pred_lost + ms
            w = compare_stats['lost_chiral']
            if self.symF.factors:
                ok = "ok (bound)" if self.pred_lost <= w <= pred else "MISMATCH"
                out.append(f"    {'lost chiral singlets = eaten + massive (diff)':<46} predicted {('<= ' + str(pred)):<22}"
                           f" worldsheet {str(w):<22} {ok}")
                out.append(f"    {'':<46} (non-Abelian factors: the generic mass-matrix rank ignores the Clebsch-Gordan")
                out.append(f"    {'':<46}  relations between couplings, so it is an upper bound; eaten: {self.pred_lost},"
                           f" massive: {w - self.pred_lost})")
            else:
                row("lost chiral singlets = eaten + massive (diff)", pred, w)
        if D is not None:
            row("27 / 27bar multiplets", f"{self.specF.total.n27[1] - m27} / {self.specF.total.n27bar[1] - m27}",
                f"{D.spec.total.n27[1]} / {D.spec.total.n27bar[1]}")
        fmtg = lambda g: (" x ".join(f"Z{d}" for d in g) or "none")
        row("discrete remnant (mod unbroken gauge group)", fmtg(self.pred_remnant),
            None if self.ws_remnant is None else fmtg(self.ws_remnant))
        if self.R_preserved:
            mix = "" if not any(self.R_mixing) else " (R mixed with gauge rotation theta=(" + \
                ",".join(fmt(x) for x in self.R_mixing) + "))"
            out.append(f"    {'R-symmetry':<46} predicted Z{sym.M}^R unbroken{mix}")
        else:
            out.append(f"    {'R-symmetry':<46} predicted broken to Z{self.R_remaining}")
        if D is not None:
            out.append(f"    {'':<46} worldsheet: Q_R = 3 k0 - 2 q- + n.q mod {D.M} exists "
                       f"(all gauginos R = 3: {'yes' if all(s.R == 3 for s in D.states if s.state.q_plus == Fraction(-3, 2)) else 'NO'})")
        for n in self.notes:
            out.append("  note: " + n)
        return "\n".join(out) + "\n"


def _rank_mod(M, P):
    M = [row[:] for row in M]
    rank = 0
    ncol = len(M[0]) if M else 0
    for c in range(ncol):
        piv = next((i for i in range(rank, len(M)) if M[i][c] % P), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        inv = pow(M[rank][c], -1, P)
        M[rank] = [x * inv % P for x in M[rank]]
        for i in range(len(M)):
            if i != rank and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % P for a, b in zip(M[i], M[rank])]
        rank += 1
    return rank


# ------------------------------------------------------------------ driver
class Deformation:
    """Deform a computed model, compute the deformed one, compare and propose VEVs."""

    def __init__(self, specF: ModelSpectrum, symF: SymmetryAnalysis, terms, name=None, processes=1, progress=None):
        if isinstance(terms, str):
            terms = parse_terms(specF.model, terms)
        self.terms = terms
        self.specF, self.symF = specF, symF
        self.model = deformed_model(specF.model, terms, name)
        self.spec = compute_model(self.model, processes=processes, progress=progress)
        self.sym = SymmetryAnalysis(self.spec)
        from .compare import compare, ClassDiff
        self.compare_text = compare(specF, symF, self.spec, self.sym, specF.model.name, self.model.name)
        # lost counts from the comparison
        lv = self.compare_text.split("INTERPRETATION")[1].split("\n")[1] if "INTERPRETATION" in self.compare_text else ""
        m = re.search(r"(\d+) vector multiplets and (\d+) chiral multiplets of A are lost", lv)
        self.stats = {"lost_vectors": int(m.group(1)), "lost_chiral": int(m.group(2))} if m else None
        self.vev = VevProposal(specF, symF, terms, self.sym)

    def text(self) -> str:
        return self.compare_text + "\n" + self.vev.text(self.stats)
