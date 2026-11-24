"""General topological defect lines of Fermat Gepner models (``lgor defects``).

Five analyses, each result tagged
    [verified]    computed here, exactly or with a stated numerical check
    [literature]  a standard result quoted from the literature (named), not re-derived here
    [inferred]    follows from verified results plus an argument that is not checked by code
    [missing]     not computed by this version

1. Survival of the Verlinde lines of the minimal-model factors in the full heterotic model:
   monodromy with the spacetime supercharge / E6 current beta_0, with the supercurrents beta_i
   and with the orbifold currents, including dressing by invertible (abelian) lines.
2. Self-duality (Tambara-Yamagami) defects: subgroups H of the phase symmetries with C/H = C up
   to left N=2 charge conjugation and/or a permutation of identical factors.  Exact comparison
   of the torus partition functions (simple-current modular invariants) plus an LGOR comparison
   of the massless spectra.
3. Group-theoretical lines: integral fusion subcategories of the surviving lines compared with
   Rep(G), and the categories C(G, K) for permutation symmetries G of identical factors.
4. Lines that preserve the chiral algebras: genuine (ambichiral) lines versus alpha^+-/alpha^-
   induced pairs, fixed points of the current group, number of simple defects sum Z_ij^2.
5. F-symbols (su(2)_k quantum 6j, pentagon checked), known categories, anomalies/gaugeability.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

from . import fingroups as fg
from . import fsymbols as fs
from . import zgroups as zg
from .model import Field, Model
from .rcft import Current, Factor, GepnerData

V, LIT, INF, MISS = "[verified]", "[literature]", "[inferred]", "[missing]"


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _fmt_phase(a, ns):
    return "(" + ", ".join(str(Fr(x % n, n)) if x % n else "0" for x, n in zip(a, ns)) + ")"


def _solve_affine(rows: List[List[Fr]], consts: List[Fr], nv: int) -> Optional[List[int]]:
    """Integer x with consts + rows.x in Z^g (componentwise), or None."""
    g = len(rows)
    if g == 0:
        return [0] * nv
    D = 1
    for r, c in zip(rows, consts):
        for x in list(r) + [c]:
            D = _lcm(D, Fr(x).denominator)
    A = []
    for t, (r, c) in enumerate(zip(rows, consts)):
        A.append([int(Fr(x) * D) for x in r] + [-D if s == t else 0 for s in range(g)] + [int(Fr(c) * D)])
    K = zg.int_kernel(A, nv + g + 1)
    # combine kernel vectors to get last coordinate 1
    acc = None
    for v in K:
        if acc is None:
            if v[-1]:
                acc = v
            continue
        a, b = acc[-1], v[-1]
        if b == 0:
            continue
        gg, x, y = zg._xgcd(a, b)
        acc = [x * p + y * q for p, q in zip(acc, v)]
    if acc is None or abs(acc[-1]) != 1:
        return None
    if acc[-1] == -1:
        acc = [-x for x in acc]
    return acc[:nv]


class DefectAnalysis:
    def __init__(self, model: Model, spec=None, max_subgroups: int = 5000, max_perms: int = 120,
                 lgor_check: bool = True, max_lgor_checks: int = 6, processes: int = 1):
        self.model = model
        self.gd = GepnerData(model)
        self.spec = spec
        self.max_subgroups = max_subgroups
        self.max_perms = max_perms
        self.lgor_check = lgor_check
        self.max_lgor_checks = max_lgor_checks
        self.processes = processes
        self.status: List[Tuple[str, str, str]] = []      # (check, tag, short result)
        self.data: Dict[str, object] = {}

    # =================================================================== section 1
    def _dressing_conditions(self, p: Sequence[int], with_orbifold: bool):
        """Variables: for every factor f a dressing (m_f, s_f = 2 t_f).  Even-k factors carry a
        line of parity p_f, so m_f = p_f + 2 u_f; odd-k factors have m_f free (the class has
        representatives of both parities).  Returns (rows, consts, decode)."""
        gd = self.gd
        r = gd.r
        nv = 2 * r                      # (u_f or m_f, t_f)
        rows, consts = [], []

        def m_coeff(f):
            return (2, p[f]) if gd.levels[f] % 2 == 0 else (1, 0)

        currents = [gd.beta0] + ([*gd.orbifold] if with_orbifold else [])
        for J in currents:
            row = [Fr(0)] * nv
            c = Fr(0)
            for f in range(r):
                mJ, sJ = J.ms[f]
                n = gd.ns[f]
                a, b = m_coeff(f)
                row[2 * f] += Fr(mJ * a, 2 * n)
                c += Fr(mJ * b, 2 * n)
                row[2 * f + 1] += -Fr(sJ * 2, 4)
            rows.append(row)
            consts.append(c)

        def decode(x):
            out = []
            for f in range(r):
                a, b = m_coeff(f)
                m = (a * x[2 * f] + b) % (2 * gd.ns[f])
                s = (2 * x[2 * f + 1]) % 4
                out.append((m, s))
            return out
        return rows, consts, nv, decode

    def section1(self) -> str:
        gd = self.gd
        r = gd.r
        out = ["1. SURVIVAL OF VERLINDE LINES IN THE FULL MODEL", "-" * 100]
        out.append("Factors (N=2 minimal models): " + ", ".join(f"x{i + 1}: k={k}" for i, k in enumerate(gd.levels)))
        out.append("Currents: beta_0 = spacetime supercharge (right) / E6 spinor currents (left); beta_i = supercurrent of")
        out.append("factor i; orbifold: " + (", ".join(f"{J.name} phases {_fmt_phase(J.phase, gd.ns)}" for J in [gd.jcur] + gd.orbifold)))
        out.append("")
        out.append("Criterion " + LIT + " (Verlinde lines of a simple-current extension/orbifold; Fuchs-Runkel-Schweigert,")
        out.append("Boeckenhauer-Evans-Kawahigashi): a line a of the product theory commutes with a current J iff its monodromy")
        out.append("charge Q_J(a) is an integer.  Physical requirement used here " + INF + ": the line must commute with the")
        out.append("worldsheet supercurrents (s even) and with beta_0, otherwise it rotates the spacetime supercharge or the")
        out.append("16 of SO(10) inside E6 by a phase.  Monodromy with an orbifold current only decides whether the line")
        out.append("descends as ONE symmetric line (integer charge) or as a PAIR alpha^+(a), alpha^-(a) that differ by a")
        out.append("quantum-symmetry line of the orbifold " + LIT + " (alpha-induction).")
        out.append("")
        # parity analysis
        even = [f for f in range(r) if gd.levels[f] % 2 == 0 and gd.levels[f] > 0]
        out.append("Lines per factor (su(2)_k label l up to l ~ k-l):")
        for f, F in enumerate(gd.factors):
            cls = sorted({F.lclass(l) for l in range(F.k + 1)})
            desc = []
            for l in cls:
                d = F.dim(l)
                par = "both parities" if F.k % 2 else ("even" if l % 2 == 0 else "odd")
                desc.append(f"l={l} (d={d:.4f}, {par})")
            out.append(f"  x{f + 1} (k={F.k}): " + "; ".join(desc))
        out.append("")
        results = {}
        allp = list(itertools.product([0, 1], repeat=len(even)))
        for pe in allp:
            p = [0] * r
            for f, v in zip(even, pe):
                p[f] = v
            row1 = self._dressing_conditions(p, False)
            row2 = self._dressing_conditions(p, True)
            x1 = _solve_affine(row1[0], row1[1], row1[2])
            x2 = _solve_affine(row2[0], row2[1], row2[2])
            results[pe] = (None if x1 is None else row1[3](x1), None if x2 is None else row2[3](x2))
        self.data["parity"] = results
        self.data["even_factors"] = even
        out.append("Lines whose su(2) labels are all even (and every line of an odd-k factor, which has an even")
        out.append("representative) survive with trivial dressing as single symmetric lines " + V + ".")
        if even:
            s1 = [pe for pe, (a, b) in results.items() if a is not None and any(pe)]
            s2 = [pe for pe, (a, b) in results.items() if b is not None and any(pe)]
            def gens(S):
                basis = []
                span = {tuple([0] * len(even))}
                for v in sorted(S, key=lambda v: (sum(v), v)):
                    if v in span:
                        continue
                    basis.append(v)
                    span |= {tuple((a + b) % 2 for a, b in zip(v, w)) for w in span}
                return basis, span
            b1, sp1 = gens(s1)
            b2, sp2 = gens(s2)
            out.append(f"Odd su(2) labels sit in the even-k factors {', '.join('x%d' % (f + 1) for f in even)}.  The parity patterns")
            out.append(f"(which of these factors carry an odd l) that can be dressed form a group " + V + ":")
            out.append(f"  commuting with supercurrents and beta_0: {len(sp1)} of {2 ** len(even)} patterns; generators:")
            for v in b1:
                al = results[v][0]
                fac = [even[i] for i, x in enumerate(v) if x]
                kind = "single line" if results[v][1] is not None else "alpha^+/alpha^- pair (orbifold-nonlocal)"
                out.append(f"    odd l in {', '.join('x%d' % (f + 1) for f in fac)}: {kind}; a dressing (m,s) per factor: {al}")
            if not b1:
                out.append("    none: every line with an odd label in an even-k factor rotates the supercharge/E6 " + V)
            out.append(f"  also local with all orbifold currents (single symmetric lines): {len(sp2)} patterns")
            self.status.append(("1 odd-label lines (even k) that survive", V, f"{len(sp1)} of {2 ** len(even)} parity patterns"))
        else:
            out.append("No even-k factor with k > 0: no odd-label constraint.")
            self.status.append(("1 odd-label lines (even k) that survive", V, "no even-k factor"))
        # count of non-invertible class vectors surviving
        tot_sing, tot_pair = 0, 0
        cls_lists = []
        for f, F in enumerate(gd.factors):
            cls_lists.append(sorted({F.lclass(l) for l in range(F.k + 1)}))
        for L in itertools.product(*cls_lists):
            pe = tuple(L[f] % 2 for f in even)
            a, b = results[pe] if even else (True, True)
            if a is None:
                continue
            if all(gd.factors[f].dim(L[f]) < 1 + 1e-9 for f in range(r)):
                continue
            if b is not None:
                tot_sing += 1
            else:
                tot_pair += 1
        out.append("")
        out.append(f"Surviving non-invertible line types (su(2) class vectors with d > 1, up to invertible dressing) " + V + ":")
        out.append(f"  single symmetric lines: {tot_sing};  alpha^+/alpha^- pairs: {tot_pair}")
        self.status.append(("1 surviving non-invertible line types", V, f"{tot_sing} single + {tot_pair} pairs"))
        # fusion per factor
        out.append("")
        out.append("Fusion " + V + " (su(2)_k fusion in l, addition of (m, s); labels (l,m,s) canonical, (0,m,s) invertible):")
        for k in sorted(set(gd.levels)):
            if k == 0:
                continue
            F = Factor(k)
            bare = lambda l: F.canon(l, l % 2, 0)
            cls = sorted({F.lclass(l) for l in range(k + 1)})
            nonin = [l for l in cls if F.dim(l) > 1 + 1e-9]
            if not nonin:
                out.append(f"  k={k}: only invertible lines")
                continue
            for a in nonin[:4]:
                for b in nonin[:4]:
                    if b < a:
                        continue
                    res = F.fuse(bare(a), bare(b))
                    out.append(f"  k={k}: {bare(a)} x {bare(b)} = " + " + ".join(str(c) for c in res))
        # explicit product example for each generator
        if even:
            for v in b1[:4]:
                al = results[v][0]
                odd = {even[i] for i, x in enumerate(v) if x}
                line = self._line_from([1 if f in odd else 0 for f in range(r)], al)
                if line is None:
                    continue
                comps = gd.fuse(line, gd.conj(line))
                ok = all(self._is_local(c) for c in comps)
                out.append(f"  example X = {line} (d = {gd.dim(line):.4f}):  X x Xbar has {sum(comps.values())} simple components, "
                           f"all commuting with supercurrents and beta_0: {ok} " + V)
        out.append("")
        out.append("Not done here " + MISS + ": eigenvalues of dressed lines on the LGOR massless states (for undressed")
        out.append("lines see `lgor tdl`); lines that do not come from Verlinde lines of the product (see sections 2-4).")
        return "\n".join(out)

    def _is_local(self, line) -> bool:
        gd = self.gd
        if any(s % 2 for (_, _, s) in line):
            return False
        return gd.Q(gd.beta0, line) == 0

    # =================================================================== section 2
    def _phase_quotient(self):
        """Phase group Ph = {a : sum a_f/n_f in Z} modulo the orbifold group O (j + .lgo generators)."""
        gd = self.gd
        r, ns = gd.r, gd.ns
        Ph = zg.kernel_mod1([[Fr(1, n) for n in ns]], r)
        per = [[n if i == f else 0 for i in range(r)] for f, n in enumerate(ns)]
        Ogens = [list(gd.jcur.phase)] + [list(J.phase) for J in gd.orbifold]
        O = zg.hnf(Ogens + per, r)
        hPh = zg.hnf(Ph, r)
        # O cap Ph
        coeffs = zg.kernel_mod1([[sum(Fr(row[f], ns[f]) for f in range(r)) for row in O]], len(O))
        OPh = zg.hnf([[sum(c[i] * O[i][f] for i in range(len(O))) for f in range(r)] for c in coeffs], r)
        basis, orders = zg.smith_basis(hPh, OPh, r)
        j_in = zg.contains(hPh, gd.jcur.phase, r)
        return basis, orders, OPh, j_in

    def _pairs_lattice(self, currents: List[List[int]]):
        """Simple-current modular invariant of the internal theory, NS sector, as a lattice of pairs
        (alpha, mu) in Z^{2r} x Z^{2r}; alpha = (m_f, t_f) left with s_f = 2 t_f, mu right.
        Kreuzer-Schellekens with the symmetric torsion X = B/2 (B the monodromy bilinear form):
            mu = alpha + sum b_a J_a,   B(J_c, alpha) + sum_a X_ca b_a in Z for every c.
        The Ramond sector follows from the NS sector by spectral flow in both theories."""
        gd = self.gd
        r, ns = gd.r, gd.ns
        D = 2 * r
        g = len(currents)
        cur = [[(J[i] if i % 2 == 0 else J[i] // 2) for i in range(D)] for J in currents]   # (m, t)

        def B(u, v):
            q = Fr(0)
            for f in range(r):
                q += Fr(u[2 * f] * v[2 * f], 2 * ns[f]) - u[2 * f + 1] * v[2 * f + 1]
            return q
        rows = []
        for c in range(g):
            row = [Fr(0)] * (D + g)
            Jc = cur[c]
            for f in range(r):
                row[2 * f] = Fr(Jc[2 * f], 2 * ns[f])
                row[2 * f + 1] = Fr(-Jc[2 * f + 1])
            for a in range(g):
                row[D + a] = B(Jc, cur[a]) / 2
            rows.append(row)
        Lam = zg.kernel_mod1(rows, D + g)
        gens = []
        for v in Lam:
            al = v[:D]
            mu = list(al)
            for a in range(g):
                for i in range(D):
                    mu[i] += v[D + a] * cur[a][i]
            gens.append(al + mu)
        per = []
        for f in range(r):
            for i, p in ((2 * f, 2 * ns[f]), (2 * f + 1, 2)):
                e = [0] * (2 * D)
                e[i] = p
                per.append(e)
                e2 = [0] * (2 * D)
                e2[D + i] = p
                per.append(e2)
        return zg.hnf(gens + per, 2 * D)

    def _base_currents(self) -> List[List[int]]:
        gd = self.gd
        out = []
        for J in gd.align + [gd.jcur] + gd.orbifold:
            v = []
            for m, s in J.ms:
                v += [m, s]
            out.append(v)
        return out

    def _phase_current(self, a) -> List[int]:
        v = []
        for x in a:
            v += [2 * x, 0]
        return v

    def _apply_left(self, P, omega: bool, perm: Sequence[int]):
        """Image of the pair lattice under (alpha, mu) -> (phi(alpha), mu)."""
        r = self.gd.r
        D = 2 * r
        out = []
        for row in P:
            al, mu = row[:D], row[D:]
            new = [0] * D
            for f in range(r):
                g = perm[f]
                new[2 * g] = al[2 * f]
                new[2 * g + 1] = al[2 * f + 1]
            if omega:
                new = [-x for x in new]
            out.append(new + mu)
        return zg.lattice_key(out, 2 * D)

    def _perms(self):
        gd = self.gd
        blocks = {}
        for f, k in enumerate(gd.levels):
            blocks.setdefault(k, []).append(f)
        groups = [list(itertools.permutations(b)) for b in blocks.values()]
        total = 1
        for g in groups:
            total *= len(g)
        if total > self.max_perms:
            return [tuple(range(gd.r))], False
        out = []
        for combo in itertools.product(*groups):
            perm = list(range(gd.r))
            for b, img in zip(blocks.values(), combo):
                for src, dst in zip(b, img):
                    perm[src] = dst
            out.append(tuple(perm))
        return out, True

    def section2(self) -> str:
        gd = self.gd
        r, ns = gd.r, gd.ns
        out = ["2. SELF-DUALITY (TAMBARA-YAMAGAMI) DEFECTS FROM HALF GAUGING", "-" * 100]
        basis, orders, OPh, j_in = self._phase_quotient()
        qord = 1
        for o in orders:
            qord *= o
        perms, allperms = self._perms()
        base = self._base_currents()
        P_C = self._pairs_lattice(base)
        keys = {}
        for om in (False, True):
            for perm in perms:
                keys.setdefault(self._apply_left(P_C, om, perm), []).append((om, perm))
        out.append("Test " + V + ": the torus partition function of the gauged theory (simple-current invariant, all states,")
        out.append("not only massless; NS sector, the R sector follows by spectral flow " + INF + ") equals that of C with an")
        out.append("automorphism phi applied to the left movers.  phi: identity or left N=2 charge conjugation omega, times")
        out.append(f"{len(perms)} permutations of identical factors" + ("" if allperms else " (too many: only the identity " + MISS + ")") + ".")
        out.append("Convention: Kreuzer-Schellekens invariants with symmetric torsion X = B/2.  It reproduces the Greene-Plesser")
        out.append("mirrors of LGOR models (unit tests) " + V + "; that LGOR's orbifolds carry exactly this torsion is " + INF + ".")
        out.append("")
        found = []
        # (a) gauging phase symmetries
        out.append("(a) Gauging a subgroup H of the phase symmetries x_f -> exp(2 pi i a_f/n_f) x_f, sum a_f/n_f in Z,")
        out.append("    modulo the orbifold group: Q = " + (" x ".join(f"Z{o}" for o in orders) or "trivial") + f" (order {qord}) " + V)
        if not j_in:
            out.append("    note: J itself is not in this phase group (sum q_i not an integer); quotient taken by O cap Ph.")
        if orders:
            subs, complete = zg.subgroups(orders, cap=self.max_subgroups)
            out.append(f"    subgroups H scanned: {len(subs) - 1}" + ("" if complete else f" (capped at {self.max_subgroups}; the rest " + MISS + ")"))
            for H in subs:
                gens = []
                for row in H:
                    if not any(c % o for c, o in zip(row, orders)):
                        continue
                    a_ = [0] * r
                    for i, c in enumerate(row):
                        for f in range(r):
                            a_[f] += c * basis[i][f]
                    gens.append([x % n for x, n in zip(a_, ns)])
                if not gens:
                    continue
                Hord = zg.subgroup_order(H, orders)
                P_H = self._pairs_lattice(base + [self._phase_current(x) for x in gens])
                key = tuple(map(tuple, P_H))
                if key in keys:
                    found.append(("phase", Hord, gens, keys[key]))
        # (b) gauging quantum symmetries (removing orbifold generators of the .lgo file)
        out.append("(b) Gauging the quantum symmetry dual to a set of the .lgo orbifold generators (undoing that orbifold):")
        if gd.orbifold:
            for size in range(1, len(gd.orbifold) + 1):
                for S in itertools.combinations(range(len(gd.orbifold)), size):
                    keep = [J for i, J in enumerate(gd.orbifold) if i not in S]
                    cur = []
                    for J in gd.align + [gd.jcur] + keep:
                        v = []
                        for m_, s_ in J.ms:
                            v += [m_, s_]
                        cur.append(v)
                    key = tuple(map(tuple, self._pairs_lattice(cur)))
                    if key in keys:
                        Hord = 1
                        for i in S:
                            Hord *= self.model.orders[i]
                        found.append(("quantum", Hord, list(S), keys[key]))
            out.append(f"    subsets of the {len(gd.orbifold)} generators scanned: {2 ** len(gd.orbifold) - 1}; "
                       "general subgroups of the orbifold group and mixed phase/quantum gaugings " + MISS)
        else:
            out.append("    the model has no orbifold generators besides J")
        self.data["selfdual"] = found
        out.append("")
        if not found:
            out.append("No gauging gives back phi(C): no self-duality defect of these types " + V + ".")
            self.status.append(("2 self-duality defects (TY)", V, "none"))
            return "\n".join(out)
        out.append(f"Self-dual gaugings found: {len(found)}")
        for kind, Hord, gens, phis in found:
            sq = math.isqrt(Hord) ** 2 == Hord
            names = []
            for om, perm in phis:
                names.append(("omega" if om else "id") + ("" if perm == tuple(range(r)) else " o perm" + str(tuple(p + 1 for p in perm))))
            if kind == "phase":
                out.append(f"  |H| = {Hord}, H generated by the phases " + ", ".join(_fmt_phase(x, ns) for x in gens))
            else:
                out.append(f"  |H| = {Hord}, H = quantum symmetry dual to the orbifold generators " +
                           ", ".join(f"#{i + 1} (Z{self.model.orders[i]})" for i in gens))
            out.append(f"     gauged theory = phi(C) for phi in {{{', '.join(names[:6])}{', ...' if len(names) > 6 else ''}}} " + V)
            out.append(f"     duality defect D = phi o (half gauging of H): <D> = sqrt({Hord}) = {math.sqrt(Hord):.4f}, "
                       f"D x Dbar = sum_(h in H) eta_h " + LIT + " (Tambara-Yamagami; Choi-Cordova-Hsin-Lam-Shao 2021)")
            if any(om for om, _ in phis):
                out.append("     with omega: D flips the left N=2 charges, i.e. exchanges 27 <-> 27bar in 4D " + INF)
            out.append("     " + ("|H| is a perfect square: TY(H) may be group-theoretical and gaugeable, depending on the "
                                  "bicharacter chi and tau " + MISS if sq else
                                  "|H| is not a perfect square: D is intrinsically non-invertible (not group-theoretical) and has no "
                                  "fiber functor (cannot be gauged) " + LIT + " (Tambara 2000; Gelaki-Naidu-Nikshych 2009)"))
            out.append("     self-conjugacy of D, bicharacter chi and sign tau of the TY category " + MISS)
        self.status.append(("2 self-duality defects (TY)", V, f"{len(found)}: orders " + ", ".join(str(h) for _, h, _, _ in found)))
        if self.lgor_check:
            out.append("")
            out.append("LGOR cross-check of the massless spectra (independent of the modular-invariant test):")
            nchk = 0
            for kind, Hord, gens, phis in found:
                if nchk >= self.max_lgor_checks:
                    out.append(f"  (further gaugings not checked, limit {self.max_lgor_checks}) " + MISS)
                    break
                nchk += 1
                try:
                    msg = self._lgor_massless(kind, gens, phis)
                except Exception as e:      # keep going
                    msg = f"  |H| = {Hord}: LGOR check could not run ({e}) " + MISS
                out.append(msg)
        else:
            out.append("LGOR massless cross-check skipped (--no-lgor) " + MISS)
        return "\n".join(out)

    def _orbifold_model(self, gens) -> Model:
        """C/H as an LGOR model: add independent generators of H (direct product with the old group)."""
        gd = self.gd
        m = self.model
        r, ns = gd.r, gd.ns
        per = [[n if i == f else 0 for i in range(r)] for f, n in enumerate(ns)]
        Ogens = [list(gd.jcur.phase)] + [list(J.phase) for J in gd.orbifold]
        Olat = zg.hnf(Ogens + per, r)
        b, o = zg.smith_basis(gens, Olat, r)
        # O elements mod periods (to adjust lifts)
        O_el = set()
        frontier = [tuple([0] * r)]
        O_el.add(frontier[0])
        while frontier:
            new = []
            for x in frontier:
                for g in Ogens:
                    y = tuple((a + c) % n for a, c, n in zip(x, g, ns))
                    if y not in O_el:
                        O_el.add(y)
                        new.append(y)
            frontier = new
        orders = list(m.orders)
        charges = [list(f.charges) for f in m.fields]
        for vec, d in zip(b, o):
            lift = None
            for oe in sorted(O_el):
                cand = [(v + e) % n for v, e, n in zip(vec, oe, ns)]
                if all((d * c) % n == 0 for c, n in zip(cand, ns)):
                    lift = cand
                    break
            if lift is None:
                raise ValueError("H does not split off the orbifold group (no direct-product generator)")
            c = [lift[f] * d // ns[f] for f in range(r)]
            c[-1] -= sum(c)
            orders.append(d)
            for f in range(r):
                charges[f].append(c[f])
        fields = [Field(f.name, f.kcharge, ch) for f, ch in zip(m.fields, charges)]
        new = Model(f"{m.name}_H", m.korder, orders, fields, list(m.superpotential))
        probs = new.check()
        if probs:
            raise ValueError("; ".join(probs))
        from .mirror import generator_overlap
        w = generator_overlap(new)
        if w:
            raise ValueError(w)
        return new

    def _spectrum(self):
        if self.spec is None:
            self.spec = self.model.compute(processes=self.processes)
        return self.spec

    def _ungauged_model(self, removed) -> Model:
        m = self.model
        keep = [j for j in range(m.ndiscrete) if j not in removed]
        fields = [Field(f.name, f.kcharge, [f.charges[j] for j in keep]) for f in m.fields]
        return Model(f"{m.name}_ungauged", m.korder, [m.orders[j] for j in keep], fields, list(m.superpotential))

    def _lgor_massless(self, kind, gens, phis) -> str:
        from .symmetries import SymmetryAnalysis
        gd = self.gd
        r = gd.r
        spec = self._spectrum()
        symC = SymmetryAnalysis(spec)
        newm = self._orbifold_model(gens) if kind == "phase" else self._ungauged_model(gens)
        specH = newm.compute(processes=self.processes)
        symH = SymmetryAnalysis(specH)
        keyH = Counter((s.state.q_plus, s.state.q_minus, tuple(s.q)) for s in symH.states)
        ok_phi = []
        for om, perm in phis:
            sg = -1 if om else 1
            kc = Counter()
            for s in symC.states:
                q = [None] * r
                for f in range(r):
                    q[perm[f]] = sg * s.q[f]
                kc[(s.state.q_plus, sg * s.state.q_minus, tuple(q))] += 1
            if kc == keyH:
                ok_phi.append((om, perm))
        tC, tH = spec.total, specH.total
        if kind == "phase":
            nm = f"H = <{', '.join(_fmt_phase(a, gd.ns) for a in gens)}>"
        else:
            nm = "H = quantum symmetry of generators " + ", ".join(f"#{i + 1}" for i in gens)
        res = (f"  {nm}: C has (27, 27bar) = ({tC.n27[1]}, {tC.n27bar[1]}), C/H has ({tH.n27[1]}, {tH.n27bar[1]}); "
               f"all massless states of C/H = phi(states of C): "
               f"{'yes for ' + str(len(ok_phi)) + ' of ' + str(len(phis)) + ' phi' if ok_phi else 'NO'} " + V)
        if kind == "phase":
            lifts = []
            for g in gens:                       # lift with sum a_f/n_f = 0 exactly (insensitive to vacuum shifts)
                g = list(g)
                tot = sum(Fr(a, n) for a, n in zip(g, gd.ns))
                g[-1] -= int(tot * gd.ns[-1])
                lifts.append(g)
            inv = [st for st in symC.states if all(sum(a * x for a, x in zip(g, st.q)) % 1 == 0 for g in lifts)]
            res += (f"\n     H-invariant massless states of C: {len(inv)} of {len(symC.states)} (D maps them to phi-images "
                    "with factor sqrt|H| and annihilates the others " + LIT + ")")
        return res

    # =================================================================== section 3
    def section3(self) -> str:
        gd = self.gd
        r = gd.r
        out = ["3. GROUP-THEORETICAL LINES", "-" * 100]
        out.append("(a) Integral fusion subcategories generated by surviving lines, compared with Rep(G):")
        if not fg.have_numpy():
            out.append("    numpy not available: character data of fusion rings not computed " + MISS)
        cands = []
        results = self.data.get("parity", {})
        even = self.data.get("even_factors", [])
        cls = [sorted({F.lclass(l) for l in range(F.k + 1)}) for F in gd.factors]
        # single-factor and two-factor candidates with integer dimension
        seen_types = set()
        for f in range(r):
            for l in cls[f]:
                d = gd.factors[f].dim(l)
                if d > 1.5 and abs(d - round(d)) < 1e-9:
                    cands.append(((f, l),))
        for f, g in itertools.combinations(range(r), 2):
            for l in cls[f]:
                for m in cls[g]:
                    d = gd.factors[f].dim(l) * gd.factors[g].dim(m)
                    if gd.factors[f].dim(l) > 1 + 1e-9 and gd.factors[g].dim(m) > 1 + 1e-9 and abs(d - round(d)) < 1e-9:
                        cands.append(((f, l), (g, m)))
        shown = 0
        for cand in cands:
            typ = tuple(sorted((gd.levels[f], l) for f, l in cand))
            if typ in seen_types:
                continue
            seen_types.add(typ)
            X = self._dressed(cand, results, even)
            if X is None:
                continue
            facs = [f for f, l in cand]
            ks = [gd.levels[f] for f in facs]
            start = tuple(l for f, l in cand)
            labels = self._su2_closure(ks, start)
            idx = {a: i for i, a in enumerate(labels)}
            N = [[{} for _ in labels] for _ in labels]
            for a in labels:
                for b in labels:
                    for c, m in self._su2_fuse(ks, a, b).items():
                        N[idx[a]][idx[b]][idx[c]] = m
            dims = [math.prod(Factor(k).dim(l) for k, l in zip(ks, a)) for a in labels]
            integral = all(abs(d - round(d)) < 1e-9 for d in dims)
            dl = Counter(round(d, 4) for d in dims)
            txt = (f"    {self._name(cand)} (surviving as X = {X}): su(2) part of the generated subcategory: rank {len(labels)}, "
                   f"dims {dict(sorted(dl.items()))}, global dim {sum(d * d for d in dims):.1f}")
            if integral and fg.have_numpy():
                rd = fg.ring_character_data(N, dims)
                if rd:
                    mt = fg.match_rep_groups(*rd)
                    txt += (f"; class sizes {[int(round(x)) for x in rd[1]]}; same fusion ring as Rep(G) for G in "
                            f"{mt or 'none of the library groups (S3, D4, Q8, D5, A4, D6, S4, D8, A5 and products)'} " + V)
                else:
                    txt += "; ring not commutative or eigen-decomposition failed"
            elif not integral:
                txt += "; not integral: not Rep(G) " + V
            out.append(txt)
            shown += 1
            if shown >= 12:
                out.append("    (more candidates not shown)")
                break
        if not cands:
            out.append("    no surviving line with integer dimension in one or two factors " + V)
        out.append("    Whether the category (not just the fusion ring) is Rep(G): F-symbols not compared " + MISS)
        out.append("")
        # (b) permutation symmetries
        out.append("(b) Permutation symmetries of identical factors and the group-theoretical categories C(G, K):")
        blocks = {}
        for f, k in enumerate(gd.levels):
            blocks.setdefault(k, []).append(f)
        big = max(blocks.values(), key=len)
        m = len(big)
        out.append(f"    largest block of identical factors: k={gd.levels[big[0]]}, {m} factors -> G = S{m} permutes them")
        out.append("    (a symmetry of the Fermat model if the orbifold group is permutation invariant " + INF + ")")
        if m < 2:
            out.append("    no permutation symmetry")
        elif m > 5:
            out.append("    S_m with m > 5 not computed " + MISS)
        else:
            G = fg.symmetric_group(m)
            Ks = self._cyclic_subgroups(G)
            out.append("    gauging K in C gives C/K, whose lines form C(G, K) " + LIT + " (Ostrik; Bhardwaj-Tachikawa);")
            out.append("    spectrum of the permutation orbifold C/K " + MISS + " (LGOR has no permutation orbifolds); anomaly of K "
                       + MISS + " (assumed trivial)")
            for name, K in Ks + [("G = S%d" % m, G)]:
                try:
                    if len(K) == len(G):
                        chars = fg.irreducible_characters(G)
                        dims = [round(c[G[0]].real) for c in chars]
                        txt = f"    K = {name}: C(G,G) = Rep(S{m}), dims {sorted(dims)} " + V
                    else:
                        T = fg.GroupTheoretical(G, K)
                        dims = T.dims()
                        N = T.fusion()
                        dl = Counter(round(x, 3) for x in dims)
                        non = sum(1 for x in dims if x > 1 + 1e-9)
                        txt = (f"    K = {name}: rank {len(dims)}, dims {dict(sorted(dl.items()))}, non-invertible {non}, "
                               f"global dim {sum(x * x for x in dims):.0f} (= |G|) " + V)
                        if len(dims) <= 8:
                            labs = T.labels()
                            for i in range(len(dims)):
                                for j in range(i, len(dims)):
                                    if dims[i] > 1 + 1e-9 or dims[j] > 1 + 1e-9:
                                        terms = [f"{mlt if mlt > 1 else ''}{labs[c]}" for c, mlt in sorted(N[i][j].items())]
                                        txt += f"\n        {labs[i]} x {labs[j]} = " + " + ".join(terms)
                except RuntimeError as e:
                    txt = f"    K = {name}: {e} " + MISS
                out.append(txt)
        self.status.append(("3 group-theoretical lines", V, f"{shown} integral-subring candidates; permutation block S{m}"))
        return "\n".join(out)

    @staticmethod
    def _su2_fuse(ks, a, b):
        out = {(): 1}
        for k, x, y in zip(ks, a, b):
            new = {}
            for c in range(abs(x - y), min(x + y, 2 * k - x - y) + 1, 2):
                for pre, mlt in out.items():
                    new[pre + (c,)] = new.get(pre + (c,), 0) + mlt
            out = new
        return out

    def _su2_closure(self, ks, start):
        ring = {tuple(0 for _ in ks), start}
        frontier = list(ring)
        while frontier:
            new = []
            for a in frontier:
                for c in self._su2_fuse(ks, a, start):
                    if c not in ring:
                        ring.add(c)
                        new.append(c)
            frontier = new
        return sorted(ring)

    def _cyclic_subgroups(self, G):
        out, seen = [], set()
        for g in G:
            K = fg.closure([g], len(g))
            if len(K) == 1:
                continue
            ctype = tuple(sorted(len(c) for c in self._cycles(g)))
            if ctype in seen:
                continue
            seen.add(ctype)
            out.append((f"<{fg._cyc(g)}> = Z{len(K)}", K))
        return out

    @staticmethod
    def _cycles(p):
        seen, cyc = set(), []
        for i in range(len(p)):
            if i in seen:
                continue
            c, j = [], i
            while j not in seen:
                seen.add(j)
                c.append(j)
                j = p[j]
            cyc.append(c)
        return cyc

    def _name(self, cand):
        return " x ".join(f"l={l} in x{f + 1} (k={self.gd.levels[f]})" for f, l in cand)

    def _dressed(self, cand, results, even):
        gd = self.gd
        r = gd.r
        L = [0] * r
        for f, l in cand:
            L[f] = l
        pe = tuple(L[f] % 2 for f in even)
        if even:
            if pe not in results or results[pe][0] is None:
                return None
            al = results[pe][0]
        else:
            al = [(0, 0)] * r
        line = self._line_from(L, al)
        return line if line is not None and self._is_local(line) else None

    def _line_from(self, L, al):
        """Product line with su(2) classes L and abelian dressing al = [(m, s)], or None."""
        out = []
        for F, l, (m, s) in zip(self.gd.factors, L, al):
            if (l + m + s) % 2:
                if F.k % 2 == 0:
                    return None
                l = F.k - l
            out.append(F.canon(l, m, s))
        return tuple(out)

    def _closure(self, X, cap=400):
        gd = self.gd
        ring = {gd.canon(tuple((0, 0, 0) for _ in range(gd.r)))}
        frontier = [X, gd.conj(X)]
        gensX = [X, gd.conj(X)]
        ring |= set(frontier)
        frontier = list(ring)
        while frontier:
            new = []
            for a in frontier:
                for g in gensX:
                    for c in gd.fuse(a, g):
                        if c not in ring:
                            ring.add(c)
                            new.append(c)
                            if len(ring) > cap:
                                return None
            frontier = new
        return ring

    # =================================================================== section 4
    def section4(self) -> str:
        gd = self.gd
        r, ns = gd.r, gd.ns
        out = ["4. LINES PRESERVING THE CHIRAL ALGEBRAS (alpha-induction, simple-current formulas)", "-" * 100]
        out.append("Classification " + LIT + " (Boeckenhauer-Evans-Kawahigashi; Fuchs-Runkel-Schweigert): for a simple-current")
        out.append("extension/orbifold every Verlinde line a of the product gives alpha^+(a) and alpha^-(a); alpha^+(a) = alpha^-(a)")
        out.append("(a genuine Verlinde line of the extended theory) iff a is local with all currents.  One of alpha^+/alpha^-")
        out.append("preserves the left, the other the right extended algebra; each fuses like the product: alpha(a)alpha(b) = alpha(ab).")
        out.append("Heterotic reading " + INF + ": local with beta_0 and beta_i -> preserves spacetime SUSY and E6; lines non-local")
        out.append("only with orbifold currents come as alpha^+/alpha^- pairs differing by a quantum-symmetry line (section 1).")
        out.append("")
        # fixed points of the current group
        def fixed_points(currents_with_s0, label):
            D = 2 * r + 1
            gens = []
            for J in currents_with_s0:
                v = []
                for m, s in J.ms:
                    v += [m, s]
                gens.append(v + [J.s0])
            per = []
            for f in range(r):
                for i, p in ((2 * f, 2 * ns[f]), (2 * f + 1, 4)):
                    e = [0] * D
                    e[i] = p
                    per.append(e)
            e = [0] * D
            e[-1] = 4
            per.append(e)
            H = zg.hnf(gens + per, D)
            even = [f for f in range(r) if gd.levels[f] % 2 == 0 and gd.levels[f] > 0]
            hits = []
            for size in range(1, len(even) + 1):
                for S in itertools.combinations(even, size):
                    t = [0] * D
                    for f in S:
                        t[2 * f] = ns[f]
                        t[2 * f + 1] = 2
                    if zg.contains(H, t, D):
                        hits.append(S)
            return hits
        phys = [gd.beta0] + gd.betas + [gd.jcur] + gd.orbifold
        hits = fixed_points(phys, "physical")
        hits_int = fixed_points(gd.align + [gd.jcur] + gd.orbifold, "internal")
        if hits or hits_int:
            out.append("Fixed points " + V + ": the current group contains the field-identification shift on factors")
            for S in (hits or hits_int)[:6]:
                out.append(f"  {', '.join('x%d' % (f + 1) for f in S)}: labels with l = k/2 there are fixed; such orbits split into several")
            out.append("  lines (fixed-point resolution) " + MISS + "; counts below then are lower bounds " + INF)
        else:
            out.append("Fixed points: none (the current group acts freely on the labels) " + V)
        self.status.append(("4 fixed points of the current group", V, "none" if not (hits or hits_int) else f"{len(hits or hits_int)} sets of factors"))
        # number of simple defects of the internal theory preserving the product algebra
        base = self._base_currents()
        P = self._pairs_lattice(base)
        D = 2 * r
        detP = 1
        for i, row in enumerate(P):
            detP *= row[i]
        detPer = 1
        for n in ns:
            detPer *= 2 * n * 2
        nP = detPer * detPer // detP
        # parity image of alpha (NS sector: label parity l = m mod 2)
        span = {tuple([0] * r)}
        for row in P:
            v = tuple(row[2 * f] % 2 for f in range(r))
            span |= {tuple((a + b) % 2 for a, b in zip(v, w)) for w in span}
        NL = 0
        for p in span:
            c = 1
            for f, k in enumerate(gd.levels):
                c *= sum(1 for l in range(k + 1) if l % 2 == p[f])
            NL += c
        total = nP // len(span) * NL // (2 ** r)
        nlabels = 1
        for F in gd.factors:
            nlabels *= F.n * (F.k + 1)
        out.append("")
        out.append(f"Internal (2,2) theory (LG orbifold by J and the .lgo group, NS-NS sector): number of simple topological")
        out.append(f"defects preserving the bosonic chiral algebra of every factor on both sides = sum_(ij) Z_ij^2 = {total} " + V)
        out.append("  (formula " + LIT + f" Fuchs-Runkel-Schweigert; valid without fixed points; NS Verlinde lines of the product: {nlabels})")
        self.status.append(("4 number of simple defects (sum Z^2)", V if not hits_int else INF, str(total)))
        out.append("Explicit fusion of all of these (non-local alpha^+ alpha^- products, bimodule lines) " + MISS +
                   "; fusion of alpha^+ and of alpha^- separately = product fusion " + LIT)
        return "\n".join(out)

    # =================================================================== section 5
    def section5(self) -> str:
        gd = self.gd
        out = ["5. F-SYMBOLS AND ANOMALIES", "-" * 100]
        for k in sorted(set(gd.levels)):
            if k == 0:
                continue
            out.append(f"k = {k}: {fs.known_category(k)} " + LIT)
            if k <= 12:
                S = fs.SU2k(k)
                err = S.pentagon_check(samples=150 if k <= 6 else 40)
                out.append(f"    su(2)_{k} F-symbols (quantum 6j, unitary gauge): pentagon satisfied to {err:.1e} on random samples " + V)
                ind = [S.fs_indicator(a) for a in range(k + 1)]
                out.append(f"    Frobenius-Schur indicators of l = 0..{k}: {ind} " + V)
            else:
                out.append(f"    F-symbols available via lgor.fsymbols.SU2k({k}); pentagon not checked for k > 12 " + MISS)
            alg = fs.algebra_objects(k)
            out.append("    algebra objects beyond the trivial one: " + ("; ".join(alg) if alg else "none") + " " + LIT)
            if k > 1:
                out.append("    the full su(2)_k line category has non-integer dimensions: no fiber functor, cannot be gauged as a whole " + LIT)
            if k == 3:
                F = fs.fib_from_su2_3()
                phi = (1 + 5 ** 0.5) / 2
                out.append(f"    Fibonacci F^(ttt)_t = [[{F[0][0]:.6f}, {F[0][1]:.6f}], [{F[1][0]:.6f}, {F[1][1]:.6f}]]; "
                           f"gauge-invariant entry F[1,1] = 1/phi = {1 / phi:.6f}: {abs(F[0][0] - 1 / phi) < 1e-12} " + V)
        out.append("")
        # invertible local Verlinde lines: (0, 2a, 2t) per factor
        out.append("Invertible Verlinde lines that survive (l = 0: phases (0,2a,0) and phase x (-1)^F_f (0,2a,2)):")
        r, ns = gd.r, gd.ns
        D = 2 * r
        rows = []
        for J in [gd.beta0] + gd.orbifold:
            row = []
            for f in range(r):
                mJ, sJ = J.ms[f]
                row += [Fr(mJ * 2, 2 * ns[f]), -Fr(sJ * 2, 4)]
            rows.append(row)
        # variables (a_f, t_f): line (0, 2a_f, 2t_f)
        Lloc = zg.kernel_mod1(rows, D)
        per = []
        for f in range(r):
            for i, p in ((2 * f, ns[f]), (2 * f + 1, 2)):
                e = [0] * D
                e[i] = p
                per.append(e)
        # quotient by internal currents lying in this lattice: j, orbifold phases, (0,0,2)_i (0,0,2)_j pairs
        cur = []
        for J in [gd.jcur] + gd.orbifold + gd.align:
            v = []
            for m, s in J.ms:
                v += [m // 2, s // 2]
            cur.append(v)
        Hl = zg.hnf(Lloc, D)
        cur_in = [c for c in cur if zg.contains(Hl, c, D)]
        basis, orders = zg.smith_basis(Lloc, zg.hnf(cur_in + per, D) if (cur_in + per) else per, D)
        spins = []
        for v in basis:
            h = Fr(0)
            for f in range(r):
                a, t = v[2 * f], v[2 * f + 1]
                h += -Fr((2 * a) ** 2, 4 * ns[f]) + Fr((2 * t) ** 2, 8)
            spins.append(h % 1)
        ok = fs.pointed_anomaly(orders, spins)
        out.append(f"  group modulo the internal currents: " + (" x ".join(f"Z{o}" for o in orders) or "trivial") + " " + V)
        out.append("  (lines differing by a current with spacetime part act as each other times spacetime fermion number " + INF + ")")
        for v, o, h, a in zip(basis, orders, spins, ok):
            lab = tuple((0, 2 * v[2 * f] % (2 * ns[f]), 2 * v[2 * f + 1] % 4) for f in range(r))
            out.append(f"  generator {lab}: order {o}, spin h = {h}, N h {'in' if a else 'NOT in'} Z -> "
                       f"{'anomaly free' if a else 'anomalous'} " + V + " (criterion " + LIT + ")")
        out.append("  mixed anomalies between generators are covered by the criterion only on this basis " + INF)
        # quantum-symmetry lines of the .lgo orbifold (not Verlinde lines of the product theory)
        def phase_group(currents):
            gens = [tuple(Fr(m, 2 * ns[f]) % 1 for f, (m, s) in enumerate(J.ms)) for J in currents]
            grp = {tuple(Fr(0) for _ in range(r))}
            frontier = list(grp)
            while frontier:
                new = []
                for x in frontier:
                    for g in gens:
                        y = tuple((a + b) % 1 for a, b in zip(x, g))
                        if y not in grp:
                            grp.add(y)
                            new.append(y)
                frontier = new
            return len(grp)
        nq = phase_group([gd.jcur] + gd.orbifold) // phase_group([gd.jcur]) if gd.orbifold else 1
        if nq > 1:
            out.append(f"  quantum-symmetry lines of the .lgo orbifold: dual group of order {nq} (orbifold group modulo j "
                       + V + "), invertible and anomaly free (gauging them undoes the orbifold) " + LIT)
            out.append("  these are not Verlinde lines of the product theory and are not in the group above; "
                       "their mixed anomaly with it " + MISS)
        self.status.append(("5 invertible surviving lines", V, "phase " + (" x ".join(f"Z{o}" for o in orders) or "trivial") +
                            f", {sum(1 for x in ok if not x)} anomalous generator(s)" +
                            (f"; plus quantum lines of order {nq}" if nq > 1 else "")))
        out.append("")
        sd = self.data.get("selfdual")
        if sd:
            out.append("Duality defects of section 2: see the gaugeability statement there " + LIT + "; TY data chi, tau " + MISS)
        out.append("F-symbols of dressed product lines (su(2)_k F x abelian 3-cocycles of the (m,s) parts) " + MISS)
        return "\n".join(out)

    # =================================================================== report
    def text(self, sections=(1, 2, 3, 4, 5)) -> str:
        body = []
        fns = {1: self.section1, 2: self.section2, 3: self.section3, 4: self.section4, 5: self.section5}
        for s in sections:
            try:
                body.append(fns[s]())
            except Exception as e:      # report and continue with the next section
                body.append(f"{s}. section failed: {type(e).__name__}: {e} " + MISS)
                self.status.append((f"{s} (whole section)", MISS, f"error: {e}"))
        head = [f"LGOR DEFECT LINES for model {self.model.name}", "=" * 100,
                "Tags: [verified] computed here; [literature] quoted standard result; [inferred] argued, not checked by code;",
                "      [missing] not computed in this version.", "",
                "STATUS"]
        for chk, tag, res in self.status:
            head.append(f"  {chk:<48} {tag:<13} {res}")
        head.append("")
        return "\n".join(head) + "\n" + "\n\n".join(body) + "\n"
