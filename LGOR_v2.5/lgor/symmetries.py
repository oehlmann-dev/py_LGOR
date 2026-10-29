"""Target-space symmetries and the charges of every state.

Implements section 2.2 of Blaszczyk & Oehlmann, JHEP04(2016)068:

* Cartan charges q^i, one per Landau-Ginzburg field (eq. 2.17)::

      vacuum:  q^i = (alpha_i - 1)(nutilde_i + 1/2) - alpha_i (nu_i - 1/2)
      phi^i: +alpha_i   phibar^i: -alpha_i   psi^i: alpha_i - 1   psibar^i: 1 - alpha_i

  with sum_i q^i = q-.  A combination c.q is conserved by Qbar iff it is a symmetry
  of W (for Fermat W every q^i is).

* Extra (non-E6) U(1)s: the conserved combinations c.q under which all E6
  gauginos are neutral.  Their gauginos are the zero-weight states at q+ = -3/2;
  states at q+ = -3/2 with non-zero weight are roots of a non-Abelian enhancement,
  and the gauge group is read off from the root system.

* R-symmetry (eq. 2.15)::

      Q_R = 3 k0 - 2 q- + n.q   mod M,   M = 6 KOrder

  The integer vector n is fixed by requiring every gaugino (E6 and extra) to have
  Q_R = 3.  This leaves the freedom n -> n + (Abelian U(1) generator); the shortest
  such n is taken, i.e. the R-symmetry is not mixed with the Abelian U(1)s.

* Discrete symmetries (eq. 2.18)::

      Q_j = k_j + n_j.q   mod order_j,   with all gauginos neutral.

Generators can also be given by hand in the .lgo file (lines the old C++ code
ignores)::

    U1 "strA" 1 1 1 1 1 1 -2 -2 -2      # coefficients of q^1 ... q^n
    RCorrection 2 2 2 ...               # n in Q_R (skips the automatic fit)
    DiscreteCorrection 1 0 1 -1 ...     # n_1 for the first discrete symmetry

Ambiguities: n (and n_j) are only fixed up to adding symmetries under which every
gaugino is neutral (Abelian U(1)s, the E6 and SU(n) centres, ...).  The automatic
choice is the shortest n; give the generators by hand to match a convention.

R-charge of the chiral superfield: R(Phi) = R(psi) - 3, i.e. R(theta) = -3 and
R(W) = -6 mod M (the paper's normalisation).  Note: the tables of the paper
correspond to R(psi) + 3; for KOrder = 3 the two differ by a uniform shift of 6,
invisible in cubic couplings, but only R(psi) - 3 allows e.g. the quintic's
27^3 Yukawa couplings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg
from .monomial import B, BB, BF, F
from .sector import Sector, State
from .spectrum import ModelSpectrum

Vec = Tuple[Fraction, ...]


# ------------------------------------------------------------------ helpers
def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _den(xs) -> int:
    d = 1
    for x in xs:
        d = _lcm(d, Fraction(x).denominator)
    return d


def _mod(x: Fraction, m: int) -> Fraction:
    x = Fraction(x)
    return x - m * (x.numerator // (x.denominator * m))


def _dot(c, q) -> Fraction:
    return sum((Fraction(a) * b for a, b in zip(c, q)), Fraction(0))


def fmt(x) -> str:
    return str(Fraction(x))


def field_charges(sec: Sector, mono) -> Vec:
    """The Cartan charges (q^1..q^n) of oscillator monomial ``mono`` on the vacuum of ``sec``."""
    m = sec.model
    n, N, K = m.nfields, m.N, m.korder
    out = []
    for i, f in enumerate(m.fields):
        a = Fraction(f.kcharge, K)
        nu, nut = Fraction(sec.Bt[i], N), Fraction(sec.Ft[i], N)
        q = (a - 1) * (nut + Fraction(1, 2)) - a * (nu - Fraction(1, 2))
        q += a * (mono[B * n + i] - mono[BB * n + i]) + (a - 1) * (mono[F * n + i] - mono[BF * n + i])
        out.append(q)
    return tuple(out)


def vacuum_charges(sec: Sector) -> Vec:
    return field_charges(sec, (0,) * (4 * sec.model.nfields))


# ------------------------------------------------------------------ data
@dataclass
class StateInfo:
    state: State
    bunch_twist: Tuple[int, ...]
    q: Optional[Vec]                 # Cartan charges (None if not well defined)
    rep: str                         # E6 representation component, e.g. "27 (16_-1/2)"
    u1: List[Fraction] = field(default_factory=list)
    R: Optional[Fraction] = None     # R-charge of the fermion (mod M)
    R_superfield: Optional[Fraction] = None
    discrete: List[Optional[Fraction]] = field(default_factory=list)
    weight_zero: bool = True
    mixed: bool = False              # monomials with different Cartan charges q^i
    remnant: List[Optional[Fraction]] = field(default_factory=list)

    @property
    def sector(self) -> Sector:
        return self.state.sector


@dataclass
class GaugeFactor:
    name: str
    rank: int
    roots: List[Vec]
    label: str = ""


E6_COMPONENT = {
    Fraction(-2): "27 (1_-2)", Fraction(-1, 2): "27 (16_-1/2)", Fraction(1): "27 (10_1)",
    Fraction(2): "27bar (1_2)", Fraction(1, 2): "27bar (16bar_1/2)", Fraction(-1): "27bar (10_-1)",
    Fraction(-3, 2): "78 (16_-3/2)", Fraction(3, 2): "78 (16bar_3/2)", Fraction(0): "1 / 78 (45+1)_0",
}


def _name_simple(nroots: int, rank: int) -> str:
    if rank == 1 and nroots == 2:
        return "SU(2)"
    if nroots == rank * (rank + 1):
        return f"SU({rank + 1})"
    if rank >= 4 and nroots == 2 * rank * (rank - 1):
        return f"SO({2 * rank})"
    if (rank, nroots) == (6, 72):
        return "E6"
    if (rank, nroots) == (7, 126):
        return "E7"
    if (rank, nroots) == (8, 240):
        return "E8"
    return f"?(rank {rank}, {nroots} roots)"


def _rational_rank(vectors: List[Vec]) -> int:
    if not vectors:
        return 0
    d = _den(x for v in vectors for x in v)
    rows = [[int(x * d) for x in v] for v in vectors]
    return linalg.rank(rows, len(rows[0]))


# ------------------------------------------------------------------ analysis
class SymmetryAnalysis:
    def __init__(self, spec: ModelSpectrum, u1_generators: Optional[List[Tuple[str, List[Fraction]]]] = None,
                 r_correction: Optional[List[int]] = None):
        self.spec = spec
        self.model = spec.model
        self.n = self.model.nfields
        self.M = 6 * self.model.korder
        self.warnings: List[str] = []
        if u1_generators is None:
            u1_generators = getattr(self.model, "u1_generators", None) or None
        if r_correction is None:
            r_correction = getattr(self.model, "r_correction", None)
        self._user_u1 = u1_generators
        self._user_r = r_correction
        self.run()

    # -------------------------------------------------------------- steps
    def run(self):
        self._collect_states()
        self._conserved_directions()
        self._extra_u1s()
        self._gauge_group()
        if self.U1_source == "automatic":
            if self.factors:
                self._align_basis()
            self._normalize_u1()
            self._gauge_group()
        self._r_symmetry()
        self._discrete()
        self._remnants()
        self._assign_charges()
        self._checks()

    def _checks(self):
        g = self._gauginos()
        bad = [s for s in g if s.R is not None and s.R != 3]
        if bad:
            self.warnings.append(f"{len(bad)} gauginos do not have R-charge 3")
        for j in range(self.model.ndiscrete):
            if any(s.discrete and s.discrete[j] not in (None, 0) for s in g):
                self.warnings.append(f"some gauginos are charged under discrete symmetry {j + 1}")
        # all components of an E6 multiplet must carry the same charges
        for rep, comps in (("27", (Fraction(-2), Fraction(-1, 2), Fraction(1))),
                           ("27bar", (Fraction(2), Fraction(1, 2), Fraction(-1)))):
            for qp in (Fraction(-3, 2), Fraction(-1, 2)):
                sets = []
                for qm in comps:
                    c = {}
                    for s in self.states:
                        if s.state.q_plus == qp and s.state.q_minus == qm:
                            key = (tuple(s.u1), s.R, tuple(s.discrete), tuple(s.remnant))
                            c[key] = c.get(key, 0) + 1
                    sets.append(c)
                if not (sets[0] == sets[1] == sets[2]):
                    self.warnings.append(f"the components of the {rep}-plets (q+ = {qp}) carry different charges")

    def _collect_states(self):
        self.states: List[StateInfo] = []
        for b in self.spec.bunches:
            for sec in b.sectors:
                for st in sec.states():
                    qs = {field_charges(sec, mono) for _, mono in st.terms}
                    q = next(iter(qs))
                    info = StateInfo(st, b.twist, q, E6_COMPONENT.get(st.q_minus, "?"))
                    info._qs = qs
                    if sum(q) != st.q_minus:
                        self.warnings.append(f"sum of Cartan charges != q- for a state in {sec.label}")
                    self.states.append(info)
        untw = tuple([0] * self.model.ndiscrete)
        self.untwisted = None
        for b in self.spec.bunches:
            if b.twist == untw:
                self.untwisted = b
        if self.untwisted is None:
            self.warnings.append("untwisted bunch not computed: E6 gauginos unknown")

    def _conserved_directions(self):
        """C_W: combinations c.q conserved by Qbar (symmetries of W)."""
        K = self.model.korder
        rows = []
        for term in self.model.superpotential:
            for k in range(self.n):
                if term.powers[k] > 0:
                    rows.append([term.powers[i] * self.model.fields[i].kcharge - (K if i == k else 0)
                                 for i in range(self.n)])
        ker = linalg.kernel(rows, self.n) if rows else [[int(i == j) for i in range(self.n)] for j in range(self.n)]
        self.CW = linalg.lll(ker) if ker else []
        # states whose monomials carry different conserved charges cannot happen; check anyway
        for s in self.states:
            if len(s._qs) > 1:
                s.mixed = True        # fine as long as the conserved charges agree
                vals = {tuple(_dot(c, q) for c in self.CW) for q in s._qs}
                if len(vals) > 1:
                    self.warnings.append(f"state in {s.sector.label} mixes different conserved charges")

    def _gauginos(self) -> List[StateInfo]:
        return [s for s in self.states if s.state.q_plus == Fraction(-3, 2)]

    def _e6_gaugino_charges(self) -> List[Vec]:
        out = []
        if self.untwisted is None:
            return out
        for s in self._gauginos():
            if s.bunch_twist == self.untwisted.twist and abs(s.state.q_minus) == Fraction(3, 2):
                out.append(s.q)
        out.append(vacuum_charges(self.untwisted.sectors[1]))   # the 45 + 1 sit on the T(1;0) vacuum
        return out

    def _extra_u1s(self):
        """Basis c_a of the extra U(1)s."""
        e6 = self._e6_gaugino_charges()
        if self._user_u1:
            self.u1_names = [name for name, _ in self._user_u1]
            self.U1 = [list(map(Fraction, c)) for _, c in self._user_u1]
            for name, c in zip(self.u1_names, self.U1):
                if any(_dot(c, q) != 0 for q in e6):
                    self.warnings.append(f"user U(1) {name} does not commute with E6")
            self.U1_source = "user"
        else:
            # c in span(CW) with c.q(g) = 0 for the E6 gauginos
            if not self.CW:
                self.U1 = []
            else:
                d = _den(x for q in e6 for x in q)
                A = [[int(d * _dot(cw, q)) for cw in self.CW] for q in e6]
                # a field with W = x^2 is a trivial (k = 0) factor without a U(1) current: its
                # Cartan charge only depends on the sector, so it must not enter an extra U(1)
                for i in range(self.n):
                    if self.model.fermat_exponent(i) == 2:
                        A.append([int(cw[i]) for cw in self.CW])
                ys = linalg.kernel(A, len(self.CW)) if A else [[int(i == j) for i in range(len(self.CW))]
                                                                  for j in range(len(self.CW))]
                vecs = [[sum(y[b] * self.CW[b][i] for b in range(len(self.CW))) for i in range(self.n)] for y in ys]
                vecs = [linalg.primitive(v) for v in vecs]
                self.U1 = [list(map(Fraction, v)) for v in linalg.lll(vecs)] if vecs else []
            self.u1_names = [f"U1_{a + 1}" for a in range(len(self.U1))]
            self.U1_source = "automatic"

    def weight(self, q: Vec) -> Vec:
        return tuple(_dot(c, q) for c in self.U1)

    def _gauge_group(self):
        vectors = [s for s in self._gauginos() if s.state.q_minus == 0]
        self.extra_vectors = vectors
        self.charged_vectors = [s for s in self._gauginos()
                                if s.state.q_minus not in (0, Fraction(-3, 2), Fraction(3, 2))]
        zero, roots = [], []
        self._root_q = {}
        for s in vectors:
            w = self.weight(s.q)
            s.weight_zero = not any(w)
            (zero if s.weight_zero else roots).append(w)
            if not s.weight_zero:
                self._root_q[w] = s.q
        self.n_zero_weight_vectors = len(zero)
        self.roots = roots
        rootset = set(roots)
        # connected components: alpha ~ beta if alpha +- beta is a root
        parent = list(range(len(roots)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, a in enumerate(roots):
            for j in range(i):
                b = roots[j]
                if tuple(x + y for x, y in zip(a, b)) in rootset or tuple(x - y for x, y in zip(a, b)) in rootset:
                    parent[find(i)] = find(j)
        comps: Dict[int, List[Vec]] = {}
        for i, r in enumerate(roots):
            comps.setdefault(find(i), []).append(r)
        self.factors: List[GaugeFactor] = []
        for rs in comps.values():
            rk = _rational_rank(rs)
            self.factors.append(GaugeFactor(_name_simple(len(rs), rk), rk, rs))
        self.factors.sort(key=lambda f: (-f.rank, f.name, sorted(f.roots)))
        seen: Dict[str, int] = {}
        for f in self.factors:
            dup = sum(g.name == f.name for g in self.factors) > 1
            seen[f.name] = seen.get(f.name, 0) + 1
            f.label = f.name + ("abcdefghijklmnopqrstuvwxyz"[seen[f.name] - 1] if dup else "")
        nonab_rank = sum(f.rank for f in self.factors)
        self.n_abelian = len(self.U1) - nonab_rank
        # consistency: Cartan gauginos = extra rank (+1 for the E6 Cartan in T(1;0))
        expected = len(self.U1) + (1 if self.untwisted is not None else 0)
        if self.n_zero_weight_vectors != expected:
            self.warnings.append(
                f"{self.n_zero_weight_vectors} zero-weight vectors but {len(self.U1)} extra U(1) generators "
                f"(+1 E6 Cartan): some gauge symmetry is not generated by the Cartan charges q^i")

    def _candidate_u1s(self) -> List[List[Fraction]]:
        """All combinations c.q commuting with E6, ignoring W (the U(1)s of the
        Fermat point), with the same basis conventions as the automatic U(1)s."""
        e6 = self._e6_gaugino_charges()
        d = _den(x for q in e6 for x in q)
        A = [[int(d * x) for x in q] for q in e6]
        vecs = [linalg.primitive(v) for v in linalg.kernel(A, self.n)]
        basis = [list(map(Fraction, v)) for v in linalg.lll(vecs)] if vecs else []
        out = []
        allq = [q for s in self.states for q in s._qs]
        for c in basis:
            vals = [_dot(c, q) for q in allq]
            L = _den(vals)
            g = 0
            for v in vals:
                g = gcd(g, int(v * L))
            out.append([x * L / (g or 1) for x in c])
        return out

    def _remnants(self):
        """Discrete remnants of the U(1)s broken by W.

        Candidates: c = sum_a t_a b_a with b_a the Fermat-type U(1)s (integer charges
        on all states).  exp(2 pi i c.q) is a symmetry iff c.v is an integer for the
        charge v of every term of Qbar; t in Z^r acts trivially.  The remnant group
        {t : M t in Z^m} / Z^r is read off from the Smith normal form of M.
        """
        self.remnants: List[Tuple[str, List[Fraction], int]] = []
        self.broken_u1: List[List[Fraction]] = []
        if self._user_u1:
            return
        K = self.model.korder
        vs = set()
        for term in self.model.superpotential:
            for k in range(self.n):
                if term.powers[k] > 0:
                    v = tuple(Fraction(term.powers[i] * self.model.fields[i].kcharge, K) - (1 if i == k else 0)
                              for i in range(self.n))
                    if any(v):
                        vs.add(v)
        if not vs:
            return
        B = self._candidate_u1s()
        if not B:
            return
        r = len(B)
        rows = [[_dot(b, v) for b in B] for v in sorted(vs)]
        d = _den(x for row in rows for x in row)
        A = [[int(d * x) for x in row] for row in rows]
        diag, V = linalg.smith(A, r)
        gens = []
        for i in range(r):
            if diag[i] == 0:
                continue                        # unbroken direction (already a gauge U(1))
            g = Fraction(d, diag[i])
            if g.denominator == 1:
                continue                        # trivial
            t = [V[i][a] * g for a in range(r)]
            gens.append((t, g.denominator))
        broken = [i for i in range(r) if diag[i] != 0]
        self.broken_u1 = B
        if not gens:
            return
        order = 1
        for _, k in gens:
            order *= k
        # simple presentation "b_a mod k_a" if it generates the same group
        ks = []
        for a in range(r):
            col = [row[a] for row in rows]
            if all(x.denominator == 1 for x in col):
                g = 0
                for x in col:
                    g = gcd(g, int(x))
                ks.append(g)
            else:
                ks.append(0)
        prod = 1
        for k in ks:
            prod *= k if k > 1 else 1
        if all(k > 1 for k in ks) and prod == order:
            for a, (b, k) in enumerate(zip(B, ks)):
                self.remnants.append((f"Z{k}[{a + 1}]", list(b), k))
        else:
            for j, (t, k) in enumerate(gens):
                c = [sum(t[a] * B[a][i] for a in range(r)) * k for i in range(self.n)]
                self.remnants.append((f"Z{k}[{j + 1}]", c, k))
        # remove the subgroup that acts trivially on all states (e.g. the overall
        # phase rotation, which the GSO projection already gauges)
        self.remnant_relations: List[str] = []
        if order <= 20000:
            import itertools
            allq = {q for s in self.states for q in s._qs}
            triv = []
            for ms in itertools.product(*[range(k) for _, _, k in self.remnants]):
                if any(ms) and all(sum(Fraction(m_, k) * _dot(c, q)
                                       for m_, (_, c, k) in zip(ms, self.remnants)).denominator == 1 for q in allq):
                    triv.append(ms)
            ks = {k for _, _, k in self.remnants}
            if triv and len(ks) == 1:
                p = self.remnants[0][2]
                is_prime = p > 1 and all(p % q for q in range(2, int(p ** 0.5) + 1))
                if is_prime:
                    # F_p linear algebra: keep generators e_a not in span(trivial + kept)
                    def rank_mod(vectors):
                        rows = [list(v) for v in vectors]
                        rk, col = 0, 0
                        ncol = len(self.remnants)
                        while rk < len(rows) and col < ncol:
                            piv = next((i for i in range(rk, len(rows)) if rows[i][col] % p), None)
                            if piv is None:
                                col += 1
                                continue
                            rows[rk], rows[piv] = rows[piv], rows[rk]
                            inv = pow(rows[rk][col], -1, p)
                            rows[rk] = [(x * inv) % p for x in rows[rk]]
                            for i in range(len(rows)):
                                if i != rk and rows[i][col] % p:
                                    f = rows[i][col]
                                    rows[i] = [(a - f * b) % p for a, b in zip(rows[i], rows[rk])]
                            rk += 1
                            col += 1
                        return rk
                    r_ = len(self.remnants)
                    kept = []
                    for a in range(r_):
                        e = [int(i == a) for i in range(r_)]
                        if rank_mod(triv + [[int(i == b) for i in range(r_)] for b in kept] + [e]) > \
                                rank_mod(triv + [[int(i == b) for i in range(r_)] for b in kept]):
                            kept.append(a)
                    for ms in triv:
                        if rank_mod([ms]) and len([x for x in ms if x % p]) > 0:
                            terms = " + ".join(f"{m_}*{self.remnants[a][0]}" for a, m_ in enumerate(ms) if m_)
                            self.remnant_relations.append(f"{terms} = 0 mod {p} on all states")
                            break
                    self.remnants_all = list(self.remnants)
                    self.remnants = [self.remnants[a] for a in kept]
                    for a in range(r_):
                        if a not in kept:
                            self.remnant_relations.append(
                                f"{self.remnants_all[a][0]} is not independent (acts like a combination of the others)")
                else:
                    self.warnings.append("the remnant discrete group acts partly trivially on the states")
            elif triv:
                self.warnings.append("the remnant discrete group acts partly trivially on the states")

    def residual_group(self, vectors) -> List[int]:
        """Orders of the discrete group {c in C : c.v in Z for all v in vectors} modulo
        (elements acting trivially on all states  +  directions with c.v = 0).

        C is the space of all E6-commuting combinations c.q.  Basis independent: the
        reference lattice is T = {c : c.q(s) in Z for all states s}.
        """
        B = self._candidate_u1s()
        r = len(B)
        if r == 0 or not vectors:
            return []
        allq = sorted({q for st in self.states for q in st._qs})
        rows = [[_dot(b, q) for b in B] for q in allq]
        d = _den(x for row in rows for x in row)
        basis = linalg.lattice_basis([[int(x * d) for x in row] for row in rows], r)   # rows generate Q
        if len(basis) < r:
            return []
        # T = {y : R y in Z^r} with R = basis / d  ->  y = (R)^-1 z
        from .compare import _rref_solve
        Rm = [[Fraction(x, d) for x in b] for b in basis]
        cols = []
        for a in range(r):
            e = [Fraction(int(i == a)) for i in range(r)]
            cols.append(_rref_solve(Rm, e))          # column a of R^-1
        E = [[sum(cols[a][j] * B[j][i] for j in range(r)) for i in range(self.n)] for a in range(r)]
        X = [[_dot(E[a], v) for a in range(r)] for v in vectors]
        dX = _den(x for row in X for x in row)
        if dX != 1:
            self.warnings.append("residual group: non-integer charges (rounded)")
        Xi = [[int(x * dX) for x in row] for row in X]
        diag, _ = linalg.smith(Xi, r)
        return [x // dX if dX == 1 else x for x in diag if x > 1]

    def r_mixing_for(self, states) -> Optional[List[Fraction]]:
        """theta (in the U(1) basis) with R(s) + M theta.Q(s) = 0 mod M for the given states,
        i.e. an R-symmetry (R mixed with gauge rotations) under which they are neutral."""
        if self.n_R is None:
            return None
        M = self.M
        D = 6 * M
        r = len(self.U1)
        rows, rhs = [], []
        for st in states:
            rows.append([M * int(Fraction(x)) for x in st.u1])
            rhs.append(int(-D * _mod(st.R_superfield, M)))
        if not rows:
            return [Fraction(0)] * r
        G = len(rows)
        A = [row + [-D * M if g == i else 0 for g in range(G)] for i, row in enumerate(rows)]
        sol = linalg.solve_integer(A, rhs, r + G)
        if sol is None:
            return None
        return [Fraction(x, D) for x in sol[0][:r]]

    # ---------------------------------------------------------------- complete Z_p search
    def complete_discrete(self, p: int):
        """All non-R Z_p symmetries that are linear in the sector labels and Cartan charges,

            f = a k0 + sum_j b_j k_j + sum_i c_i (D q^i) + d (2 q-)   mod p,

        with every gaugino neutral and the three components of every 27 / 27bar equal,
        modulo gauge transformations (extra U(1) charges mod p and the E6 centre).
        Only terms that are well defined mod p are used (k0 if p | 2K, k_j if p | n_j);
        for non-Fermat W the q-part is restricted to the conserved combinations.
        Returns (number of independent Z_p factors, list of generator formulas).
        """
        from .report import build_multiplets
        from .mirror import _rank_fp, _kernel_fp
        st = self.states
        if not st:
            return 0, []
        Dq = _den(x for s in st for x in s.q)
        rows, names = [], []
        if (2 * self.model.korder) % p == 0:
            rows.append([s.sector.k0 % p for s in st]); names.append("k0")
        for j, o in enumerate(self.model.orders):
            if o % p == 0:
                rows.append([s.sector.twist[j] % p for s in st]); names.append(f"k{j + 1}")
        # q-part: a basis of the lattice of integer-valued functions sum_c y_c (c.q) on the
        # states (Smith form of the matrix Dq*(c.q)).  Using Dq*(c.q) itself would lose every
        # function whose values need the factor p of Dq, e.g. qhat_1 - qhat_2 for p = 3, Dq = 6.
        if self.CW:
            from .linalg import smith
            Vq = [[int(Dq * _dot(c, s.q)) for c in self.CW] for s in st]
            diag, V = smith(Vq, len(self.CW))
            for d, v in zip(diag, V):
                if not d:
                    continue
                rows.append([(sum(a * b for a, b in zip(v, r)) // d) % p for r in Vq])
                coef = [Fraction(Dq * sum(v[j] * self.CW[j][i] for j in range(len(self.CW))), d)
                        for i in range(len(self.model.fields))]
                names.append("(" + " + ".join(f"{x}*q^{self.model.fields[i].name}"
                                              for i, x in enumerate(coef) if x) + ")")
        # discrete remnants of U(1)s broken by a non-Fermat W (their Z_p parts)
        for j, (rname, _, k) in enumerate(self.remnants):
            if k % p == 0 and all(len(s.remnant) > j and s.remnant[j] is not None for s in st):
                rows.append([int(Fraction(s.remnant[j])) % p if Fraction(s.remnant[j]).denominator == 1
                             else None for s in st])
                if any(x is None for x in rows[-1]):
                    rows.pop()
                else:
                    names.append(rname if k == p else f"{rname}|Z{p}")
        rows.append([int(2 * s.state.q_minus) % p for s in st]); names.append("2q-")
        idx = {id(x): i for i, x in enumerate(st)}
        nb = len(rows)
        cons = [[rows[b][idx[id(s)]] for b in range(nb)] for s in st if s.state.q_plus == Fraction(-3, 2)]
        mults, _ = build_multiplets(self)
        for rep in ("27", "27bar"):
            for e in mults[rep]:
                comps = [c[1] for c in e]
                for s2 in comps[1:]:
                    cons.append([(rows[b][idx[id(comps[0])]] - rows[b][idx[id(s2)]]) % p for b in range(nb)])
        X = _kernel_fp(cons, nb, p) if cons else [[int(i == j) for i in range(nb)] for j in range(nb)]

        def tri(s):
            qm = s.state.q_minus
            return 1 if qm in (Fraction(-2), Fraction(-1, 2), Fraction(1)) else \
                2 if qm in (Fraction(2), Fraction(1, 2), Fraction(-1)) else 0
        G = [f for f in self._gauge_torus_rows(st)] + ([[tri(s) for s in st]] if p == 3 else [])
        G = [[int(x) % p for x in f] for f in G]
        rG = _rank_fp(G, p)
        gens = []
        cur = list(G)
        for x in X:
            f = [sum(x[b] * rows[b][i] for b in range(nb)) % p for i in range(len(st))]
            if _rank_fp(cur + [f], p) > _rank_fp(cur, p):
                cur.append(f)
                gens.append(" + ".join(f"{c}*{n}" if c != 1 else n for c, n in zip(x, names) if c) + f"  mod {p}")
        return len(gens), gens

    def _gauge_torus_rows(self, st):
        """Charge functions y.Q(s) of the gauge-torus elements exp(2 pi i y.Q / p) whose
        phases are p-th roots of unity on every state: y runs over a basis of the dual
        of the lattice spanned by the U(1) charge vectors Q(s) (so the rows are
        integers; reduced mod p they generate all gauge transformations of order p)."""
        r = len(self.U1)
        if r == 0:
            return []
        vecs = [[Fraction(x) for x in s.u1] for s in st]
        d = _den(x for v in vecs for x in v)
        basis = linalg.lattice_basis(sorted({tuple(int(x * d) for x in v) for v in vecs}), r)
        if len(basis) < r:
            return [[Fraction(s.u1[a]) for s in st] for a in range(r)]
        from .compare import _rref_solve
        Bm = [[Fraction(x, d) for x in b] for b in basis]          # rows b_i
        rows = []
        for j in range(r):                                         # y_j: b_i . y_j = delta_ij
            y = _rref_solve(Bm, [Fraction(int(i == j)) for i in range(r)])
            rows.append([sum(a * b for a, b in zip(y, v)) for v in vecs])
        return rows

    def discrete_group_summary(self) -> str:
        primes = set()
        for n in [2 * self.model.korder] + list(self.model.orders):
            m, q = n, 2
            while m > 1:
                while m % q == 0:
                    primes.add(q)
                    m //= q
                q += 1
        out = []
        for p in sorted(primes):
            k, gens = self.complete_discrete(p)
            out.append(f"Z{p}^{k}" if k else f"(no Z{p})")
            self.complete_discrete_gens = getattr(self, "complete_discrete_gens", {})
            self.complete_discrete_gens[p] = gens
        return " x ".join(x for x in out if not x.startswith("(")) or "none", out

    def _normalize_u1(self):
        """Rescale each automatic U(1) generator so that the charges of all states
        are coprime integers."""
        new = []
        for c in self.U1:
            vals = [_dot(c, s.q) for s in self.states if s.q is not None]
            L = _den(vals)
            g = 0
            for v in vals:
                g = gcd(g, int(v * L))
            g = g or 1
            new.append([x * L / g for x in c])
        self.U1 = new

    def _align_basis(self):
        """Re-choose the automatic U(1) basis: Cartans of each non-Abelian factor
        (spanned by the projections of its roots onto the U(1) space, Euclidean
        metric in q^i), then the Abelian U(1)s orthogonal to them."""
        C = [list(map(Fraction, c)) for c in self.U1]
        ortho = []
        for c in C:                                      # Gram-Schmidt
            v = list(c)
            for e in ortho:
                ee = _dot(e, e)
                k = _dot(v, e) / ee
                v = [a - k * b for a, b in zip(v, e)]
            if any(v):
                ortho.append(v)

        def proj(q):
            out = [Fraction(0)] * self.n
            for e in ortho:
                k = _dot(q, e) / _dot(e, e)
                out = [a + k * b for a, b in zip(out, e)]
            return out

        def int_basis(vectors):
            rows = []
            for v in vectors:
                d = _den(v)
                rows.append(linalg.primitive([int(x * d) for x in v]))
            # independent subset, then LLL
            basis = []
            for r in rows:
                if linalg.rank(basis + [r], self.n) > len(basis):
                    basis.append(r)
            return linalg.lll(basis)

        new, names, used = [], [], []
        counts: Dict[str, int] = {}
        for f in self.factors:
            vecs = int_basis([proj(self._root_q[w]) for w in f.roots])
            tag = f.label.replace("(", "").replace(")", "")
            for k, v in enumerate(vecs):
                new.append(list(map(Fraction, v)))
                names.append(f"{tag}_H{k + 1}")
                used.append(v)
        # Abelian part: c in span(C) orthogonal to all factor Cartans
        if used:
            Cint = [[int(x * _den(c)) for x in c] for c in C]
            A = [[sum(a * b for a, b in zip(cb, u)) for cb in Cint] for u in used]
            ys = linalg.kernel(A, len(Cint))
            ab = [[sum(y[b] * Cint[b][i] for b in range(len(Cint))) for i in range(self.n)] for y in ys]
            for k, v in enumerate(int_basis(ab) if ab else []):
                new.append(list(map(Fraction, v)))
                names.append(f"U1_{k + 1}")
        if len(new) == len(self.U1):
            self.U1, self.u1_names = new, names
        else:
            self.warnings.append("could not align the U(1) basis with the gauge factors")

    @property
    def gauge_group(self) -> str:
        parts = [f.name for f in self.factors]
        if self.n_abelian > 0:
            parts.append("U(1)" + (f"^{self.n_abelian}" if self.n_abelian > 1 else ""))
        return " x ".join(parts) if parts else "none"

    def _fit_correction(self, targets: List[Tuple[Vec, Fraction]], modulus: int) -> Optional[List[int]]:
        """Shortest integer n in span_Z(CW) with n.q(g) = t_g mod modulus for all (q(g), t_g)."""
        if not self.CW:
            return [0] * self.n if all(_mod(t, modulus) == 0 for _, t in targets) else None
        uniq = list({(q, _mod(t, modulus)) for q, t in targets})
        d = _den([x for q, _ in uniq for x in q] + [t for _, t in uniq])
        nb = len(self.CW)
        ncols = nb + len(uniq)
        A, bvec = [], []
        for g, (q, t) in enumerate(uniq):
            row = [int(d * _dot(cw, q)) for cw in self.CW] + [0] * len(uniq)
            row[nb + g] = -d * modulus
            A.append(row)
            bvec.append(int(d * t))
        sol = linalg.solve_integer(A, bvec, ncols)
        if sol is None:
            return None
        x0, ker = sol

        def to_n(y):
            return [sum(y[b] * self.CW[b][i] for b in range(nb)) for i in range(self.n)]
        n0 = to_n(x0)
        gens = [to_n(k) for k in ker]
        gens = [g for g in gens if any(g)]
        return linalg.shortest_in_coset(n0, gens)

    def _r_symmetry(self):
        if self._user_r is not None:
            self.n_R = list(self._user_r)
            self.R_source = "user"
        else:
            targets = []
            for s in self._gauginos():
                targets.append((s.q, 3 - 3 * s.sector.k0 + 2 * s.state.q_minus))
            if self.untwisted is not None:
                targets.append((vacuum_charges(self.untwisted.sectors[1]), Fraction(0)))
            self.n_R = self._fit_correction(targets, self.M)
            self.R_source = "fit"
            if self.n_R is None:
                self.warnings.append("no R-symmetry of the form 3k0 - 2q- + n.q gives all gauginos charge 3")

    def _discrete(self):
        self.n_disc: List[Optional[List[int]]] = []
        user = getattr(self.model, "discrete_corrections", {}) or {}
        for j, order in enumerate(self.model.orders):
            if j in user:
                self.n_disc.append(list(user[j]))
                continue
            targets = [(s.q, Fraction(-s.sector.twist[j])) for s in self._gauginos()]
            if self.untwisted is not None:
                targets.append((vacuum_charges(self.untwisted.sectors[1]), Fraction(0)))
            nj = self._fit_correction(targets, order)
            if nj is None:
                self.warnings.append(f"no discrete charge k_{j + 1} + n.q leaves all gauginos neutral")
            self.n_disc.append(nj)

    def _assign_charges(self):
        for s in self.states:
            if s.q is None:
                continue
            s.remnant = []
            for name, c, k in self.remnants:
                vals = {_mod(_dot(c, q), k) for q in s._qs}
                if len(vals) > 1:
                    self.warnings.append(f"{name} charge not defined for a state in {s.sector.label}")
                    s.remnant.append(None)
                else:
                    s.remnant.append(vals.pop())
            s.u1 = list(self.weight(s.q))
            k0 = s.sector.k0
            if self.n_R is not None:
                s.R = _mod(3 * k0 - 2 * s.state.q_minus + _dot(self.n_R, s.q), self.M)
                s.R_superfield = _mod(s.R + self.R_THETA, self.M)
            s.discrete = []
            for j, order in enumerate(self.model.orders):
                nj = self.n_disc[j]
                s.discrete.append(None if nj is None else _mod(s.sector.twist[j] + _dot(nj, s.q), order))

    # gaugino R = 3; chiral superfield R = fermion R + R_THETA (see paper: R(W_4D) = -6)
    R_THETA = -3

    # -------------------------------------------------------------- output
    def describe(self) -> str:
        names = [f.name for f in self.model.fields]
        out = []

        def lin(c):
            terms = []
            for x, nm in zip(c, names):
                x = Fraction(x)
                if x == 0:
                    continue
                coef = "" if abs(x) == 1 else f"{abs(x)} "
                terms.append(("- " if x < 0 else "+ ") + f"{coef}q{nm}")
            s = " ".join(terms)
            return s[2:] if s.startswith("+ ") else ("-" + s[2:] if s else "0")
        out.append(f"Cartan charges q^i: one per field ({', '.join('q' + n for n in names)}); sum_i q^i = q-")
        nvec = len(self.extra_vectors) - (1 if self.untwisted is not None else 0)
        out.append(f"Extra gauge group ({nvec} extra vector multiplets): {self.gauge_group}")
        for f in self.factors:
            out.append(f"  {f.label}: rank {f.rank}, {len(f.roots)} roots")
        if self.charged_vectors:
            out.append(f"  + {len(self.charged_vectors)} E6-charged vector states (E6 is enhanced)")
        out.append(f"Extra U(1) generators ({self.U1_source}):" + ("" if self.U1 else " none"))
        for nm, c in zip(self.u1_names, self.U1):
            out.append(f"  {nm} = {lin(c)}")
        if self.n_R is not None:
            corr = lin(self.n_R) if any(self.n_R) else ""
            out.append(f"R-symmetry: Q_R = 3 k0 - 2 q-{(' + (' + corr + ')') if corr else ''}  mod {self.M}"
                       f"   (gauginos: 3, superpotential W: {2 * self.R_THETA};"
                       f" chiral superfield R(Phi) = R(fermion) {self.R_THETA:+d})")

        if self.remnants:
            grp = " x ".join(f"Z{k}" for _, _, k in self.remnants)
            out.append(f"Discrete remnants of the U(1)s broken by W: {grp}")
            for name, c, k in self.remnants:
                out.append(f"  {name}: Q = {lin(c)}  mod {k}")
            for rel in getattr(self, "remnant_relations", []):
                out.append(f"  ({rel})")
        for j, (order, nj) in enumerate(zip(self.model.orders, self.n_disc)):
            if nj is not None:
                corr = lin(nj) if any(nj) else ""
                out.append(f"Discrete Z{order} #{j + 1}: Q = k{j + 1}{(' + (' + corr + ')') if corr else ''}  mod {order}")
        try:
            grp, _ = self.discrete_group_summary()
            out.append(f"Complete search, non-R Z_p symmetries modulo gauge (p | 2K or p | n_j): {grp}")
            for p, gens in sorted(getattr(self, "complete_discrete_gens", {}).items()):
                for g in gens:
                    out.append(f"  Z{p}: {g}")
        except Exception as err:
            out.append(f"(complete discrete search failed: {err})")
        for w in self.warnings:
            out.append("warning: " + w)
        return "\n".join(out)
