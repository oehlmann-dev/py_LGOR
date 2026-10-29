"""A single twisted sector T_{k0; k1..kN}: twists, vacuum, massless states, Qbar cohomology.

This is the port of Sector.cpp / SField.cpp / Complex.cpp.  Integer conventions
are the same as in the C++ code: every twist is an integer in units of 1/N,
with N = model.N ("OrderHCF").  Energies are in units of 1/N, and the left/right
R-charges q-, q+ are in units of 1/(2 KOrder).

Within a sector the states with fixed q- form a complex (one ``Complex`` per q-
value: 4 in Ramond sectors, 5 in NS sectors).  Each complex has 6 spaces with
q+ = -5/2 ... 5/2, linked by Qbar.  The physical states are the cohomology at
q+ = -3/2 ... 3/2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg
from .model import Model
from .monomial import (B, BB, BF, F, KIND_NAMES, KIND_NAMES_PHYS, KIND_NAMES_TEX,
                       Mono, choose, holomorphic, product, regular_nonzero, zero)


def tdiv(a: int, b: int) -> int:
    """Integer division truncating toward zero, like C/C++ ``/``."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


Poly = List[Tuple[int, Mono]]       # list of (coefficient, monomial)


@dataclass
class State:
    """A Qbar-cohomology representative: a polynomial in oscillators acting on the sector vacuum."""
    sector: "Sector"
    qminus_index: int      # index of the complex (q- value)
    qplus_index: int       # 0..3  <->  q+ = -3/2 .. 3/2
    terms: Poly

    @property
    def q_minus(self) -> Fraction:
        if self.sector.ramond:
            return Fraction(2 * self.qminus_index - 3, 2)
        return Fraction(self.qminus_index - 2)

    @property
    def q_plus(self) -> Fraction:
        return Fraction(2 * self.qplus_index - 3, 2)

    def to_string(self, style: str = "modes") -> str:
        return self.sector.format_poly(self.terms, style)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"<State q-={self.q_minus} q+={self.q_plus} in {self.sector.label}: {self.to_string()}>"


@dataclass
class Complex:
    """States of fixed q- in one sector: 6 spaces (q+ = -5/2..5/2) linked by Qbar."""
    parts: List[List[Mono]] = field(default_factory=lambda: [[] for _ in range(6)])
    cohomology: List[List[Poly]] = field(default_factory=lambda: [[] for _ in range(4)])


class Sector:
    def __init__(self, model: Model, k0: int, twist: Sequence[int]):
        self.model = model
        self.k0 = k0
        self.twist = tuple(twist)
        self.ramond = (k0 % 2 == 0)
        self.complexes: List[Complex] = [Complex() for _ in range(4 if self.ramond else 5)]
        self.warnings: List[str] = []
        self.gravitini: List[Fraction] = []    # q+ of the gravitino / E8 x SO(10) gaugino states
        self.qbar: List[Tuple[int, Mono]] = []
        self.qbar_squared_violations = 0
        self.computed = False
        # v2.4: LGOR_GENERIC_ZERO_MODES=1 forces untruncated zero modes for every block (test of the block bases)
        import os
        self.generic_zero_modes = os.environ.get("LGOR_GENERIC_ZERO_MODES") == "1"

    # ------------------------------------------------------------------ labels
    @property
    def label(self) -> str:
        return "T(" + "; ".join([str(self.k0)] + ([", ".join(map(str, self.twist))] if self.twist else [])) + ")"

    # ------------------------------------------------------------------ twists
    def compute_twists(self):
        m = self.model
        N = m.N
        n = m.nfields
        self.Bt = [0] * n
        self.BBt = [0] * n
        self.Ft = [0] * n
        self.BFt = [0] * n
        for i, fld in enumerate(m.fields):
            kiqi = sum(c * t * of for c, t, of in zip(fld.charges, self.twist, m.order_f))
            bt = (kiqi + tdiv(self.k0 * fld.kcharge * m.korder_f, 2)) % N
            ft = (kiqi + tdiv(self.k0 * (fld.kcharge - m.korder) * m.korder_f, 2)) % N
            if ft > 0:
                ft -= N
            self.Bt[i] = bt
            self.Ft[i] = ft
            self.BBt[i] = 0 if bt == 0 else N - bt
            self.BFt[i] = 0 if ft == 0 else -N - ft

    def compute_vacuum(self):
        m = self.model
        N, K = m.N, m.korder
        energy = tdiv(-N * N * 5, 4)
        ql = tdiv(-K * N * 3, 2)
        qr = tdiv(-K * N * 3, 2)
        qu = [0] * m.ndiscrete
        for i, fld in enumerate(m.fields):
            bt, ft = self.Bt[i], self.Ft[i]
            energy += bt * (N - bt) + ft * (N + ft)
            ql += (fld.kcharge - K) * ft - fld.kcharge * bt
            qr -= (fld.kcharge - K) * bt - fld.kcharge * ft
            for j in range(m.ndiscrete):
                qu[j] += fld.charges[j] * (ft - bt + N)
        if self.ramond:
            energy = 0
        if energy % (2 * N) != 0:
            self.warnings.append(f"vacuum energy {energy} not divisible by {2 * N}")
        if (2 * ql) % N or (2 * qr) % N:
            self.warnings.append("vacuum R-charges not properly quantized")
        self.energy = tdiv(energy, 2 * N)
        self.qL = tdiv(ql, N // 2)
        self.qR = tdiv(qr, N // 2)
        self.qu = []
        for j in range(m.ndiscrete):
            if qu[j] % N:
                self.warnings.append(f"discrete vacuum charge {j} not properly quantized")
            self.qu.append(tdiv(qu[j], N) % m.orders[j])

    # ------------------------------------------------------------------ states
    def _zero_mode_boson_max(self, i: int) -> int:
        """Highest power of a bosonic zero mode phi_i kept.

        Higher powers are Qbar-exact (they sit in the Jacobian ideal x_i^{n_i-1}).
        The C++ code used OrderHCF/2/q_i - 2, which equals n_i - 2 only when every
        discrete order divides 2*KOrder; here n_i = KOrder/q_i is used directly.
        """
        fld = self.model.fields[i]
        return self.model.korder // fld.kcharge - 2

    def find_massless_neutral_monomials(self):
        m = self.model
        n, N, K = m.nfields, m.N, m.korder
        E = self.energy
        if E > 0:
            return
        # nonzero modes: (index into the 4n monomial vector, energy cost, max power)
        modes = []
        for i in range(n):
            if self.Bt[i] != 0:
                modes.append((B * n + i, self.Bt[i], None))
                modes.append((BB * n + i, self.BBt[i], None))
        for i in range(n):
            if self.Ft[i] != 0:
                modes.append((F * n + i, -self.BFt[i], 1))
                modes.append((BF * n + i, -self.Ft[i], 1))
        zero_bosons = [i for i in range(n) if self.Bt[i] == 0]
        zero_fermions = [i for i in range(n) if self.Ft[i] == 0]
        # Fermat W: the Jacobian ring of the fixed fields is the box phi_i^a, a <= n_i - 2, and the
        # psibar zero modes of those fields are paired up by Qbar, so both are truncated by hand.
        # General W (v2.4): keep every phi zero-mode power allowed by the q+ window and the psibar
        # zero modes; the Koszul part psibar_0^i d_i W of Qbar then produces the Jacobian ring of
        # W restricted to the fixed fields in the cohomology.
        # Blocks of W (fields coupled by a term). A block whose fields all have boson zero modes
        # decouples: its zero modes are replaced by a monomial basis of its Jacobian ring and its
        # psibar zero modes are dropped (Fermat: the box a <= n_i - 2). A block with only some
        # fields fixed keeps all window-allowed zero-mode powers and the psibar zero modes; the
        # Koszul part of Qbar then does the reduction ("generic" fields).
        zset = set(zero_bosons)
        box_blocks = []          # (fields, list of exponent tuples)
        generic_fields = set()
        for blk in m.blocks():
            fixed = [i for i in blk if i in zset]
            if not fixed:
                continue
            if self.generic_zero_modes or len(fixed) < len(blk):
                generic_fields.update(fixed)
                continue
            if len(blk) == 1 and m.fermat_exponent(blk[0]) is not None:
                i = blk[0]
                box_blocks.append(([i], [(a,) for a in range(self._zero_mode_boson_max(i) + 1)]))
                continue
            basis = m.jacobian_basis(blk)
            if basis is None:
                self.warnings.append(f"W restricted to fields {[i + 1 for i in blk]} is degenerate; untruncated zero modes used")
                generic_fields.update(fixed)
            else:
                box_blocks.append((list(blk), basis))
        generic = bool(generic_fields)
        zb_max = None
        zf_max = {i: (1 if (self.Bt[i] != 0 or i in generic_fields) else 0) for i in zero_fermions}

        targets = {0, -N // 2, -N}
        found: List[Tuple[List[int], int]] = []
        budget = -E
        vec = [0] * (4 * n)

        def rec(start: int, used: int):
            if E + used in targets:
                found.append((vec.copy(), E + used))
            for idx in range(start, len(modes)):
                pos, cost, mx = modes[idx]
                if used + cost > budget:
                    continue
                if mx is not None and vec[pos] >= mx:
                    continue
                vec[pos] += 1
                rec(idx, used + cost)
                vec[pos] -= 1

        rec(0, 0)

        from itertools import product as iproduct
        zf_ranges = [range(zf_max[i] + 1) for i in zero_fermions]
        Kc = [f.kcharge for f in m.fields]

        gen_list = [i for i in zero_bosons if i in generic_fields]
        pos = {i: j for j, i in enumerate(zero_bosons)}

        def zb_vectors(qr_partial):
            """Zero-mode boson powers: product of the block bases, times (generic fields) all powers
            with q+ index <= 5 (each phi_0^i raises qr by 2 K_i)."""
            for combo in iproduct(*[b for _, b in box_blocks]):
                cur = [0] * len(zero_bosons)
                used = 0
                for (blk, _), e in zip(box_blocks, combo):
                    for i, a in zip(blk, e):
                        cur[pos[i]] = a
                        used += 2 * Kc[i] * a
                if not gen_list:
                    yield tuple(cur)
                    continue
                room = 5 * K - qr_partial - used
                if room < 0:
                    continue

                def rec_zb(j, left):
                    if j == len(gen_list):
                        yield tuple(cur)
                        return
                    i = gen_list[j]
                    step = 2 * Kc[i]
                    a = 0
                    while a * step <= left:
                        cur[pos[i]] = a
                        yield from rec_zb(j + 1, left - a * step)
                        a += 1
                    cur[pos[i]] = 0

                yield from rec_zb(0, room)

        def partial_qr(v):
            qr = self.qR
            for i in range(n):
                qr += 2 * (v[i] - v[n + i] + v[2 * n + i] - v[3 * n + i]) * Kc[i]
            return qr

        for base, tester_energy in found:
            for zf in iproduct(*zf_ranges):
                vz = base.copy()
                for i, p in zip(zero_fermions, zf):
                    vz[BF * n + i] = p
                qrp = partial_qr(vz) if generic else 0
                for zb in zb_vectors(qrp):
                    v = base.copy()
                    for i, p in zip(zero_fermions, zf):
                        v[BF * n + i] = p
                    for i, p in zip(zero_bosons, zb):
                        v[B * n + i] = p
                    # neutrality under the discrete symmetries
                    ok = True
                    for j in range(m.ndiscrete):
                        c = self.qu[j]
                        for i in range(n):
                            c += (v[i] - v[n + i] + v[2 * n + i] - v[3 * n + i]) * m.fields[i].charges[j]
                        if c % m.orders[j]:
                            ok = False
                            break
                    if not ok:
                        continue
                    ql = self.qL
                    qr = self.qR
                    for i in range(n):
                        bdiff = v[i] - v[n + i]
                        fdiff = v[2 * n + i] - v[3 * n + i]
                        ql += 2 * bdiff * Kc[i] + 2 * fdiff * (Kc[i] - K)
                        qr += 2 * (bdiff + fdiff) * Kc[i]
                    if self.ramond and (ql + K) % (2 * K):
                        continue
                    if not self.ramond and ql % (2 * K):
                        continue
                    if (qr + K) % (2 * K):
                        continue
                    if tester_energy == -N:
                        self.gravitini.append(Fraction(qr, 2 * K))
                        continue
                    QuR = tdiv(qr + 5 * K, 2 * K)
                    QuL = tdiv(ql + (3 if self.ramond else 4) * K, 2 * K)
                    if QuL < 0 or QuR < 0 or QuL >= len(self.complexes) or QuR > 5:
                        if not generic:
                            self.warnings.append(f"state outside the charge window (q- index {QuL}, q+ index {QuR})")
                        continue
                    self.complexes[QuL].parts[QuR].append(tuple(v))
        for cx in self.complexes:
            for p in cx.parts:
                p.sort()

    # ------------------------------------------------------------------ Qbar
    def mono_energy(self, v: Sequence[int]) -> int:
        n = self.model.nfields
        e = 0
        for i in range(n):
            e += v[i] * self.Bt[i] + v[n + i] * self.BBt[i] - v[2 * n + i] * self.BFt[i] - v[3 * n + i] * self.Ft[i]
        return e

    def create_qbar(self):
        """Qbar = psibar^i d_i W, expanded in modes (port of Sector::CreateQBar)."""
        m = self.model
        n, N = m.nfields, m.N
        terms: List[Tuple[int, Mono]] = []

        def add_with_destroyers(basic: List[int], coeff0: int):
            nd = tdiv(self.mono_energy(basic), N)
            if nd < 0:
                self.warnings.append("Qbar term with negative number of destroyers skipped")
                return
            for combo in combinations_with_replacement(range(n), nd):
                t = basic.copy()
                for k in combo:
                    t[k] -= 1
                    t[n + k] -= 1
                if holomorphic(t, n):
                    c = coeff0
                    for k in range(n):
                        c *= choose(t[k] - t[n + k], t[k])
                    terms.append((c, tuple(t)))

        for term in m.superpotential:
            for j in range(n):
                if term.powers[j] <= 0:
                    continue
                basic = [0] * (4 * n)
                for k in range(n):
                    basic[k] = term.powers[k]
                basic[j] -= 1
                if self.Ft[j] == 0:
                    basic[3 * n + j] -= 1
                else:
                    basic[2 * n + j] += 1
                coeff0 = term.powers[j] * term.coeff
                add_with_destroyers(basic, coeff0)
                if self.Ft[j] != 0:
                    basic[2 * n + j] -= 1
                    basic[3 * n + j] -= 1
                    add_with_destroyers(basic, coeff0)
        self.qbar = terms

    def apply_qbar(self, mono: Mono) -> Dict[Mono, int]:
        n = self.model.nfields
        out: Dict[Mono, int] = defaultdict(int)
        for k, t in self.qbar:
            c, res = product(mono, t, n)
            if regular_nonzero(res, n):
                out[res] += c * k
        return out

    # ------------------------------------------------------------------ cohomology
    def _qbar_matrices(self, cx: Complex):
        """M[i]: matrix of Qbar from parts[i] to parts[i+1] (rows = parts[i+1])."""
        mats = []
        missing = 0
        for i in range(5):
            src, dst = cx.parts[i], cx.parts[i + 1]
            index = {mono: r for r, mono in enumerate(dst)}
            rows = [[0] * len(src) for _ in dst]
            for c, mono in enumerate(src):
                for res, val in self.apply_qbar(mono).items():
                    if val == 0:
                        continue
                    r = index.get(res)
                    if r is None:
                        missing += 1
                    else:
                        rows[r][c] += val
            mats.append(rows)
        if missing:
            self.warnings.append(f"{missing} Qbar image terms not in the state basis (\"Basis Vector Missing\")")
        return mats

    def _check_qbar_squared(self, cx: Complex, mats) -> int:
        bad = 0
        for i in range(4):
            A, Bm = mats[i + 1], mats[i]
            if not A or not Bm or not cx.parts[i]:
                continue
            prod = linalg.matmul(A, Bm, len(cx.parts[i + 1]), len(cx.parts[i]))
            bad += sum(1 for row in prod for x in row if x)
        return bad

    def _complex_cohomology(self, cx: Complex, check: bool):
        parts = cx.parts
        mats = self._qbar_matrices(cx)
        if check:
            self.qbar_squared_violations += self._check_qbar_squared(cx, mats)
        sizes = [len(p) for p in parts]
        keep = [[True] * s for s in sizes]
        coh: List[List[Poly]] = [[] for _ in range(4)]

        def row_nonzeros(M, r):
            return [c for c, x in enumerate(M[r]) if x]

        def col_nonzeros(M, c):
            return [r for r, row in enumerate(M) if row[c]]

        # 1. monomials that are neither hit nor mapped anywhere are cohomology by themselves
        for i in range(1, 5):
            for j in range(sizes[i]):
                hit = any(mats[i - 1][j])
                maps = bool(sizes[i + 1]) and any(row[j] for row in mats[i])
                if not hit and not maps:
                    keep[i][j] = False
                    coh[i - 1].append([(1, parts[i][j])])
        # 2. monomials paired one-to-one by Qbar cancel
        for i in range(4):
            for j in range(sizes[i]):
                col = col_nonzeros(mats[i], j)
                if len(col) == 1:
                    r = col[0]
                    if row_nonzeros(mats[i], r) == [j]:
                        keep[i][j] = False
                        keep[i + 1][r] = False
        kept = [[j for j in range(sizes[i]) if keep[i][j]] for i in range(6)]
        if not any(kept):
            cx.cohomology = coh
            return
        # 3. remaining problem: harmonic representatives = ker(Qbar) ∩ ker(Qbar^T)
        R = []
        for i in range(5):
            R.append([[mats[i][r][c] for c in kept[i]] for r in kept[i + 1]])
        R.append([])
        for i in range(6):
            rows = list(R[i]) if i < 5 else []
            if i > 0:
                lower = R[i - 1]   # shape |kept[i]| x |kept[i-1]|
                for c in range(len(kept[i - 1])):
                    rows.append([lower[r][c] for r in range(len(kept[i]))])
            ncols = len(kept[i])
            if ncols == 0:
                continue
            ker = linalg.kernel(rows, ncols)
            if not ker:
                continue
            if i in (0, 5):
                self.warnings.append(f"found Qbar-cohomology at q+ = {'-' if i == 0 else '+'}5/2")
                continue
            basis = [parts[i][j] for j in kept[i]]
            for vec in ker:
                coh[i - 1].append([(x, basis[k]) for k, x in enumerate(vec) if x])
        cx.cohomology = coh

    # ------------------------------------------------------------------ driver
    def compute(self, check: bool = True) -> "Sector":
        self.compute_twists()
        self.compute_vacuum()
        self.find_massless_neutral_monomials()
        self.create_qbar()
        for cx in self.complexes:
            self._complex_cohomology(cx, check)
        if self.qbar_squared_violations:
            self.warnings.append(f"Qbar^2 != 0 on {self.qbar_squared_violations} matrix entries")
        self.computed = True
        return self

    # ------------------------------------------------------------------ access
    def counts(self) -> List[List[int]]:
        """counts[q- index][q+ index] = number of cohomology states."""
        return [[len(cx.cohomology[j]) for j in range(4)] for cx in self.complexes]

    def raw_dimensions(self) -> List[List[int]]:
        return [[len(p) for p in cx.parts] for cx in self.complexes]

    def q_minus_values(self) -> List[Fraction]:
        if self.ramond:
            return [Fraction(2 * k - 3, 2) for k in range(4)]
        return [Fraction(k - 2) for k in range(5)]

    def states(self, q_minus=None, q_plus=None) -> List[State]:
        out = []
        for a, cx in enumerate(self.complexes):
            for b in range(4):
                for terms in cx.cohomology[b]:
                    s = State(self, a, b, terms)
                    if q_minus is not None and s.q_minus != Fraction(q_minus):
                        continue
                    if q_plus is not None and s.q_plus != Fraction(q_plus):
                        continue
                    out.append(s)
        return out

    @property
    def vacuum(self) -> Dict[str, object]:
        K = self.model.korder
        return {
            "sector": self.label,
            "energy": Fraction(self.energy, self.model.N),
            "q_minus": Fraction(self.qL, 2 * K),
            "q_plus": Fraction(self.qR, 2 * K),
            "discrete_charges": tuple(self.qu),
            "ramond": self.ramond,
        }

    def mode_number(self, kind: int, i: int) -> Fraction:
        """Mode index of oscillator ``kind`` of field i (minus its energy)."""
        N = self.model.N
        e = {B: self.Bt[i], BB: self.BBt[i], F: -self.BFt[i], BF: -self.Ft[i]}[kind]
        return Fraction(-e, N)

    def oscillator_table(self) -> List[Dict[str, object]]:
        rows = []
        for i, fld in enumerate(self.model.fields):
            rows.append({"field": fld.name,
                         **{KIND_NAMES_PHYS[k]: self.mode_number(k, i) for k in range(4)}})
        return rows

    # ------------------------------------------------------------------ printing
    def format_mono(self, mono: Mono, style: str = "modes") -> str:
        n = self.model.nfields
        parts = []
        for kind in range(4):
            for i in range(n):
                p = mono[kind * n + i]
                if p == 0:
                    continue
                name = self.model.fields[i].name
                if style == "lgor":
                    s = f"{KIND_NAMES[kind]}{name}"
                    if p != 1:
                        s += f"^{p}"
                elif style == "latex":
                    mode = self.mode_number(kind, i)
                    s = f"{KIND_NAMES_TEX[kind]}^{{{name}}}_{{{mode}}}"
                    if p != 1:
                        s = "(" + s + f")^{{{p}}}"
                else:
                    mode = self.mode_number(kind, i)
                    s = f"{KIND_NAMES_PHYS[kind]}{name}[{mode}]"
                    if p != 1:
                        s += f"^{p}"
                parts.append(s)
        if not parts:
            return "1"
        return (r"\," if style == "latex" else " ").join(parts)

    def vacuum_ket(self, style: str = "modes") -> str:
        inside = str(self.k0) + ("; " + ", ".join(map(str, self.twist)) if self.twist else "")
        if style == "latex":
            return r"\left|" + inside + r"\right\rangle"
        return "|" + inside + ">"

    def format_poly(self, terms: Poly, style: str = "modes") -> str:
        if not terms:
            return "0"
        out = []
        for idx, (c, mono) in enumerate(terms):
            body = self.format_mono(mono, style)
            sign = "-" if c < 0 else "+"
            mag = abs(c)
            sep = r"\," if style == "latex" else " "
            if body == "1":
                piece = str(mag)
            elif mag == 1:
                piece = body
            else:
                piece = f"{mag}{sep}{body}"
            if idx == 0:
                out.append(("-" if c < 0 else "") + piece)
            else:
                out.append(f" {sign} {piece}")
        poly = "".join(out)
        if len(terms) == 1 and terms[0][1] == (0,) * len(terms[0][1]):
            poly = "" if terms[0][0] == 1 else ("-" if terms[0][0] == -1 else str(terms[0][0]))
            return (poly + " " if poly else "") + self.vacuum_ket(style)
        if len(terms) > 1:
            poly = "(" + poly + ")"
        return poly + " " + self.vacuum_ket(style)
