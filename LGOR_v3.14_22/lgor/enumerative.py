"""Lines (TDLs) of Fermat Gepner models versus enumerative data of the Calabi-Yau (LGOR 2.6, `enumerative`).

Three questions, for any Fermat model W = sum x_i^{n_i} / Z_d (d = KOrder) and the Verlinde lines of one factor:

1. Brane charges (linear data).  The RR ground states of the narrow sectors J^nu (every field twisted,
   nu w_i/d not an integer) carry the D-brane charges of the Kahler sector (H^even of X).  In sector nu
   factor i has label l_i = (nu mod n_i) - 1, and a line L_r of factor i acts by S_{r,l}/S_{0,l} [verified].
   The Gepner monodromy A (the quantum Z_d) acts by zeta_d^nu, and every narrow sector has exactly one
   ground state, so the characters of A separate all of them.  Hence every line acts on these charges as a
   polynomial p(A) (Lagrange interpolation) [verified arithmetic].  On RS branes W(B) = fusion
   [literature: Brunner-Douglas-Lawrence-Roemelsberger; Petkova-Zuber].  Consequence: on narrow-sector
   charges no line gives information beyond the invertible Z_d.  Genuine statements need non-linear data.

2. Kahler couplings at the Gepner point (non-linear data).  The genus-0 A-model correlators of X at the
   Gepner point are the B-model chiral-ring correlators of the Greene-Plesser mirror at its Fermat point
   [literature: mirror symmetry].  The polynomial Kahler moduli of X are the mirror's untwisted 27s; their
   3-point functions are kappa_abc(t_G) = -d^3 F, the 4-point functions d kappa_abc(t_G).  Computed exactly
   with lgor.tdl.FlatCouplings on the mirror; each zero is tested against the strict fusion rule of the
   chosen factor and against every abelian symmetry (integer lattice test).  A zero that the factor's
   fusion rule forbids and no abelian symmetry explains is a genuine line zero of enumerative data:
   the analytic continuation of the GV generating function to the Gepner point vanishes there.

3. Survival.  A line survives the Kahler deformations that are transparent to it (exponent 0 or k in its
   factor; Cordova-Rizi criterion).  Along that locus its zeros persist [inferred, first order checked].

For one-parameter hypersurfaces (h11 = 1, five fields) `OneParameter` also does the numerics of the
quintic check: GV invariants from the B-model, integral periods from the GV prepotential, Picard-Fuchs
continuation to the conifold and to the Gepner point, monodromies T, C, A, the integer matrix W = p(A),
the eigenvalue test at t_G and the deviation at large volume.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Model
from .tdl import (FlatCouplings, _hnf, channels, fermat_exponents, fuse, line_dim, marginal_monomials, strict_fusion_ok)


# ----------------------------------------------------------------------------- 1. brane charges
def weights(model: Model) -> Tuple[int, List[int]]:
    d = model.korder
    return d, [f.kcharge for f in model.fields]


def narrow_sectors(model: Model) -> List[Tuple[int, Tuple[int, ...]]]:
    """(nu, labels l_i) of every narrow sector J^nu, nu = 1..d-1."""
    n = fermat_exponents(model)
    d = model.korder
    out = []
    for nu in range(1, d):
        if all(nu % ni for ni in n):
            out.append((nu, tuple((nu % ni) - 1 for ni in n)))
    return out


def line_eigen(n: Sequence[int], factor: int, r: int, l: Sequence[int]) -> float:
    k = n[factor] - 2
    return line_dim(k, r, l[factor])


def _solve(M, b):
    """Gaussian elimination over complex numbers."""
    m = len(b)
    A = [list(row) + [b[i]] for i, row in enumerate(M)]
    for c in range(m):
        p = max(range(c, m), key=lambda i: abs(A[i][c]))
        A[c], A[p] = A[p], A[c]
        for i in range(m):
            if i != c and A[i][c] != 0:
                f = A[i][c] / A[c][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [A[i][m] / A[i][i] for i in range(m)]


def charge_polynomial(model: Model, factor: int, r: int) -> dict:
    """Coefficients c_j (j < m) with  L_r = sum_j c_j A^j  on the narrow-sector charges."""
    n = fermat_exponents(model)
    d = model.korder
    secs = narrow_sectors(model)
    zs = [complex(math.cos(2 * math.pi * nu / d), math.sin(2 * math.pi * nu / d)) for nu, _ in secs]
    lam = [line_eigen(n, factor, r, l) for _, l in secs]
    m = len(secs)
    c = _solve([[z ** j for j in range(m)] for z in zs], [complex(x) for x in lam])
    real = all(abs(x.imag) < 1e-9 for x in c)
    rat = [Fr(x.real).limit_denominator(1000) for x in c] if real else None
    exact = real and all(abs(float(q) - x.real) < 1e-9 for q, x in zip(rat, c))
    # the annihilating polynomial prod (x - zeta^nu) has integer coefficients (narrow set is Galois stable)
    return {"sectors": secs, "eigen": lam, "coeffs": rat if exact else c, "rational": exact,
            "integer": exact and all(q.denominator == 1 for q in rat), "m": m}


def forbidding_lines(k: int, ls: Sequence[int], special: bool):
    """Lines r (0 < r < k, up to r -> k - r) that alone forbid a coupling with labels ls in one factor:
    the product of the fields fuses to a single channel c and S_{r,c}/S_{0,c} differs from the value on the
    target label (0 in the special factor, k otherwise).  None if there are several channels."""
    ch = channels(k, ls)
    t = 0 if special else k
    if t in ch:
        return []
    if len(ch) != 1:
        return None
    c = next(iter(ch))
    return [r for r in range(1, k // 2 + 1) if abs(line_dim(k, r, c) - line_dim(k, r, t)) > 1e-9]


def u1_forced(q: Sequence[Tuple[int, ...]], n: Sequence[int], H: Sequence[int]) -> bool:
    """Whether the flat n-point coupling F_q (totally symmetric, F_q = d_E F_{q-E}) is forced to vanish by the
    U(1) current J_B of a block B of factors that stays decoupled along a subset E of the entries (every e in E
    lies inside B or avoids B).  Along E, J_B is exact, so F_{q-E}(t_E) vanishes identically unless its J_B charge
    mu_B = sum_{j in B} (s_j(q-E) - H_j)/n_j is an integer in [0, |q-E| - 3].  E empty is the ordinary U(1) rule.
    For 4-point couplings this is: some entry has exponent 0 in the special factor [verified on all survey models]."""
    N, r = len(q), len(q[0])
    s = [sum(x) for x in zip(*q)]
    m = [Fr(si - H[j], n[j]) for j, si in enumerate(s)]
    if any(mi.denominator != 1 or mi < 0 for mi in m) or sum(m) != N - 3:
        return True
    from itertools import combinations
    sup = [frozenset(j for j in range(r) if x[j]) for x in q]
    for size in range(0, N - 2):
        for E in combinations(range(N), size):
            for bits in range(1, 2 ** r):
                B = frozenset(j for j in range(r) if bits >> j & 1)
                if any(not (sup[e] <= B or not (sup[e] & B)) for e in E):
                    continue
                mu = sum(Fr(s[j] - sum(q[e][j] for e in E) - H[j], n[j]) for j in B)
                if mu.denominator != 1 or mu < 0 or mu > N - size - 3:
                    return True
    return False


# ----------------------------------------------------------------------------- 2. Kahler couplings
class KahlerCouplings:
    """Genus-0 A-model correlators of X at the Gepner point = mirror chiral-ring correlators."""

    def __init__(self, model: Model, mirror: Optional[Model] = None):
        from .mirror import mirror_model
        self.model = model
        self.mirror = mirror or mirror_model(model)
        self.n = fermat_exponents(self.mirror)
        self.K = [x - 2 for x in self.n]
        self.moduli = marginal_monomials(self.mirror, invariant=True)
        self.fc = FlatCouplings(self.n)

    def three_point(self):
        out = []
        ms = sorted(self.moduli)
        for i, a in enumerate(ms):
            for j in range(i, len(ms)):
                for k in range(j, len(ms)):
                    b, c = ms[j], ms[k]
                    if self.fc.add(a, b, c) == self.fc.H:
                        out.append((a, b, c))
        return out

    def four_point(self):
        """(quadruple, value set, special factor) for every abelian-allowed 4-point coupling."""
        from itertools import permutations
        out = []
        for q in sorted(self.fc.quadruples(self.moduli)):
            vals = {self.fc.four_point(*p) for p in set(permutations(q))}
            out.append((q, vals, self.fc.special_factor(q)))
        return out

    def lattice_explained(self, zeros, nonzeros) -> List[bool]:
        """For each zero coupling (list of monomials, with the marginal one marked by its special factor),
        whether an abelian symmetry can forbid it while allowing every non-zero coupling (3- and 4-point)."""
        idx = {}

        def col(key):
            if key not in idx:
                idx[key] = len(idx)
            return idx[key]

        def vec(c):
            q, sf = c
            keys = [("27", a) for a in q] if sf is None else \
                [("27", a) for a in q[1:]] + [("M", q[0], sf)]
            return keys
        allv = [vec(c) for c in nonzeros] + [vec(c) for c in zeros]
        for ks in allv:
            for kk in ks:
                col(kk)
        ncol = len(idx)

        def dense(ks):
            v = [0] * ncol
            for kk in ks:
                v[idx[kk]] += 1
            return v
        nz = [dense(vec(c)) for c in nonzeros]
        if not nz:
            return [True] * len(zeros)
        u0 = nz[0]
        basis = _hnf([[x - y for x, y in zip(u, u0)] for u in nz[1:]], ncol)

        def in_span(v):
            r = v[:]
            for p in sorted(basis):
                if r[p]:
                    b = basis[p]
                    if r[p] % b[p]:
                        return False
                    qq = r[p] // b[p]
                    r = [x - qq * y for x, y in zip(r, b)]
            return not any(r)
        return [not in_span([x - y for x, y in zip(dense(vec(c)), u0)]) for c in zeros]

    def analyse(self, factor: int) -> dict:
        k = self.K[factor]
        three = self.three_point()
        four = self.four_point()
        zeros, nonzeros, info = [], [], []
        for t in three:
            nonzeros.append((t, None))
        for q, vals, sf in four:
            # put a marginal-carrying entry first: the one with non-zero exponent in the special factor
            d = next(x for x in q if x[sf])
            rest = list(q)
            rest.remove(d)
            key = ((d,) + tuple(rest), sf)
            nz = any(v != 0 for v in vals)
            (nonzeros if nz else zeros).append(key)
            strict = strict_fusion_ok(k, [x[factor] for x in q], sf == factor)
            info.append((q, nz, strict, sf))
        expl = self.lattice_explained(zeros, nonzeros)
        zi = 0
        rows = []
        for q, nz, strict, sf in info:
            if nz:
                rows.append((q, True, strict, None, sf))
            else:
                rows.append((q, False, strict, expl[zi], sf))
                zi += 1
        H = [x - 2 for x in self.n]
        u1 = {r[0]: u1_forced(r[0], self.n, H) for r in rows if not r[1]}
        genuine = [r for r in rows if not r[1] and not r[2] and r[3] is False and not u1[r[0]]]
        beyond_phase = [r for r in rows if not r[1] and not r[2]]
        self.u1 = u1
        return {"three": three, "rows": rows, "genuine": genuine, "beyond_phase": beyond_phase, "u1": u1,
                "n_nonzero": len(nonzeros), "consistent": all(r[2] or not r[1] for r in rows)}

    def transparent(self, factor: int, r: int) -> List[Tuple[int, ...]]:
        """Moduli on which the line acts by +-d (a sign -d is compensated by a phase, see `survival`)."""
        k = self.K[factor]
        d = line_dim(k, r)
        return [a for a in self.moduli if abs(abs(line_dim(k, r, a[factor])) - d) < 1e-9]

    def survival(self, factor: int, r: int) -> dict:
        """Largest family kept by L_r or by L_r composed with a phase x_f -> w^j x_f of its own factor.
        L_r g_j acts on a modulus with exponent e by w^(j e) S_{r,e}/S_{0,e}; it is kept iff that equals d."""
        k, nf = self.K[factor], self.n[factor]
        d = line_dim(k, r)
        best = None
        for j in range(nf):
            kept = []
            for a in self.moduli:
                lam = line_dim(k, r, a[factor])
                ph = complex(math.cos(2 * math.pi * j * a[factor] / nf), math.sin(2 * math.pi * j * a[factor] / nf))
                if abs(ph * lam - d) < 1e-9:
                    kept.append(a)
            if best is None or len(kept) > len(best[1]):
                best = (j, kept)
        return {"phase": best[0], "kept": best[1], "n": nf}


# ----------------------------------------------------------------------------- 4. one-parameter numerics
class OneParameter:
    """Quintic-type numerics for X = P(w)[d] with five fields and h11 = 1 (needs mpmath)."""

    def __init__(self, model: Model, dps: int = 30, nfrob: int = 80, ngv: int = 10):
        import mpmath as mp
        self.mp = mp
        mp.mp.dps = dps
        d, w = weights(model)
        if len(w) != 5:
            raise ValueError("one-parameter numerics need five fields")
        self.d, self.w = d, w
        self.nus = [nu for nu, _ in narrow_sectors(model)]
        if len(self.nus) != 4:
            raise ValueError(f"{len(self.nus)} narrow sectors: not a one-parameter model")
        pw = 1
        for x in w:
            pw *= x
        e2 = sum(w[i] * w[j] for i in range(5) for j in range(i + 1, 5))
        e3 = sum(w[i] * w[j] * w[k] for i in range(5) for j in range(i + 1, 5) for k in range(j + 1, 5))
        self.kappa = Fr(d, pw)                 # H^3
        self.c2H = e2 * self.kappa             # c2.H
        self.chi = (e3 - d * e2) * self.kappa  # Euler number
        self.mu = mp.mpf(d) ** d / mp.fprod([mp.mpf(x) ** x for x in w])
        self.N = nfrob
        self.CF = [self._cn(n) for n in range(nfrob)]
        self.GV = self.gv(ngv)

    # Frobenius coefficients of c_n(rho) = G(d(n+rho)+1)/prod G(w_i(n+rho)+1), normalised at n = 0
    def _cn(self, n):
        mp = self.mp
        d, w = self.d, self.w
        a = [mp.loggamma(d * n + 1) - sum(mp.loggamma(x * n + 1) for x in w)]
        for j in range(1, 4):
            s = d ** j * (mp.polygamma(j - 1, d * n + 1) - mp.polygamma(j - 1, 1))
            s -= sum(x ** j * (mp.polygamma(j - 1, x * n + 1) - mp.polygamma(j - 1, 1)) for x in w)
            a.append(s / mp.factorial(j))
        e0, a1, a2, a3 = mp.e ** a[0], a[1], a[2], a[3]
        return [e0, e0 * a1, e0 * (a2 + a1 ** 2 / 2), e0 * (a3 + a1 * a2 + a1 ** 3 / 6)]

    def frob_u(self, u):
        """w_k(psi), u = log psi, z = 1/(mu psi^d)."""
        mp = self.mp
        lz = -mp.log(self.mu) - self.d * u
        z = mp.e ** lz
        w = [mp.mpc(0)] * 4
        for n in range(self.N):
            zn = z ** n
            c = self.CF[n]
            for k in range(4):
                w[k] += sum(c[k - j] * lz ** j / mp.factorial(j) for j in range(k + 1)) * zn
        return w

    # GV invariants from the B-model Yukawa C_{theta theta theta} = kappa/(1 - mu z)
    def gv(self, nmax):
        mp = self.mp
        with mp.workdps(mp.mp.dps + 4 * nmax):     # GV numbers grow like e^(c d): extra digits per degree
            return self._gv(nmax)

    def _gv(self, nmax):
        mp = self.mp
        M = nmax + 1
        CF = [self._cn(n) for n in range(M)]
        c0 = [CF[n][0] for n in range(M)]
        c1 = [CF[n][1] for n in range(M)]

        def mul(a, b):
            return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(M)]

        def inv(a):
            r = [mp.mpf(0)] * M
            r[0] = 1 / a[0]
            for k in range(1, M):
                r[k] = -sum(a[i] * r[k - i] for i in range(1, k + 1)) / a[0]
            return r

        def exp_(a):           # a[0] = 0
            r = [mp.mpf(0)] * M
            r[0] = mp.mpf(1)
            for k in range(1, M):
                r[k] = sum(i * a[i] * r[k - i] for i in range(1, k + 1)) / k
            return r
        g = mul(c1, inv(c0))                       # T = log z + g(z)
        qz = [mp.mpf(0)] + exp_(g)[:M - 1]          # q = z exp(g)
        # invert q(z) -> z(q)
        zq = [mp.mpf(0), 1 / qz[1]] + [mp.mpf(0)] * (M - 2)

        def compose(a, b):     # a(b(q)), b[0] = 0
            r = [mp.mpf(0)] * M
            p = [mp.mpf(1)] + [mp.mpf(0)] * (M - 1)
            for k in range(M):
                r = [x + a[k] * y for x, y in zip(r, p)]
                p = mul(p, b)
            return r
        for _ in range(M):
            e = compose(qz, zq)
            e[1] -= 1
            zq = [x - y / qz[1] for x, y in zip(zq, e)]
        # theta_z T = 1 + theta g ; Yukawa in t:  Y = C(z) / (w0^2 (theta T)^3)
        thT = [mp.mpf(1)] + [k * g[k] for k in range(1, M)]
        geo = [self.mu ** k for k in range(M)]
        C = [mp.mpf(self.kappa.numerator) / self.kappa.denominator * x for x in geo]
        Yz = mul(C, inv(mul(mul(c0, c0), mul(thT, mul(thT, thT)))))
        Yq = compose(Yz, zq)
        n = {}
        for D in range(1, M):
            s = Yq[D]
            for dd in range(1, D):
                if D % dd == 0 and dd in n:
                    s -= n[dd] * dd ** 3
            n[D] = s / D ** 3
        return [int(mp.nint(n[D])) for D in range(1, M)], +max(abs(n[D] - mp.nint(n[D])) for D in range(1, M))

    def Pi_LV(self, t, a2):
        mp = self.mp
        q = mp.e ** (2j * mp.pi * t)
        tpi = 2j * mp.pi
        kap = mp.mpf(self.kappa.numerator) / self.kappa.denominator
        c2 = mp.mpf(self.c2H.numerator) / self.c2H.denominator
        chi = mp.mpf(self.chi.numerator) / self.chi.denominator
        gv = self.GV[0]
        F = -kap / 6 * t ** 3 + a2 / 2 * t ** 2 + c2 / 24 * t + chi * mp.zeta(3) / (2 * tpi ** 3) \
            - sum(nd * mp.polylog(3, q ** dd) for dd, nd in enumerate(gv, 1)) / tpi ** 3
        F1 = -kap / 2 * t ** 2 + a2 * t + c2 / 24 - sum(nd * dd * mp.polylog(2, q ** dd) for dd, nd in enumerate(gv, 1)) / tpi ** 2
        return [2 * F - t * F1, F1, mp.mpf(1), t]

    def run(self, line_coeffs, psi0=2.0, fusion_terms=()):
        """fusion_terms: coefficient lists of the lines L_s in L_r x L_r = sum_s L_s (checked as W_r^2 = sum W_s)."""
        mp = self.mp
        d = self.d
        # integral basis: fit Pi = M w near psi0, scanning the quadratic coefficient a2 for integrality of C
        u0 = mp.log(mp.mpf(psi0))
        pts = [psi0, psi0 + 0.3, psi0 - 0.2 + 0.3j, psi0 + 0.1 - 0.4j, psi0 + 0.5 + 0.2j, psi0 - 0.1 - 0.1j, psi0 + 0.2 + 0.5j]
        Wv = [self.frob_u(mp.log(mp.mpmathify(p))) for p in pts]

        def rhs(u, y):
            w0, d1, d2, d3 = y
            # (e^{du} - 1) D^4 w = [prod(nu_k - D) - D^4] w
            co = _poly_prod([[nu, -1] for nu in self.nus])       # coefficients in D: co[j] D^j
            s = sum(co[j] * [w0, d1, d2, d3][j] for j in range(4))
            return [d1, d2, d3, s / (mp.e ** (d * u) - 1)]

        def seg(ua, ub, y):
            h = ub - ua
            return mp.odefun(lambda s_, Y: [h * v for v in rhs(ua + s_ * h, Y)], 0, y)(1)

        def transport(upath, ys):
            for a_, b_ in zip(upath[:-1], upath[1:]):
                ys = [seg(a_, b_, y) for y in ys]
            return ys
        Y0 = [[mp.diff(lambda u: self.frob_u(u)[k], u0, m) for m in range(4)] for k in range(4)]
        r = mp.mpf('0.3')
        arc = lambda th0, th1, nn: [mp.log(1 + r * mp.e ** (1j * (th0 + (th1 - th0) * kk / nn))) for kk in range(nn + 1)]
        loop = [u0, mp.log(1 + r)] + arc(0, 2 * mp.pi, 16) + [mp.log(1 + r), u0]
        Yloop = transport(loop, Y0)
        Mw = mp.matrix(Yloop) * mp.matrix(Y0) ** -1
        gpath = [u0, mp.log(1 + r)] + arc(0, mp.pi, 8) + [mp.log(mp.mpf('1e-7'))]
        YG = transport(gpath, Y0)
        best = None
        for m2 in range(-12, 13):
            a2 = mp.mpf(m2) / 2
            P_ = [[x * w[0] for x in self.Pi_LV(w[1] / w[0] / (2j * mp.pi), a2)] for w in Wv]
            A_ = mp.matrix(Wv)
            M = mp.matrix(4, 4)
            for i in range(4):
                sol = mp.qr_solve(A_, mp.matrix([P_[q][i] for q in range(len(pts))]))[0]
                for j in range(4):
                    M[i, j] = sol[j]
            C = M * Mw * M ** -1
            dev = max(abs(C[i, j] - mp.nint(mp.re(C[i, j]))) for i in range(4) for j in range(4))
            ts = [2j, 0.3 + 2.2j, -0.2 + 1.9j, 0.1 + 2.5j]
            P = mp.matrix([self.Pi_LV(mp.mpmathify(t), a2) for t in ts]).T
            P1 = mp.matrix([self.Pi_LV(mp.mpmathify(t) + 1, a2) for t in ts]).T
            T = P1 * P ** -1
            devT = max(abs(T[i, j] - mp.nint(mp.re(T[i, j]))) for i in range(4) for j in range(4))
            if best is None or dev + devT < best[0]:
                best = (dev + devT, a2, M, C, T, dev, devT)
        _, a2, M, C, T, devC, devT = best
        nint = lambda X: mp.matrix([[mp.nint(mp.re(X[i, j])) for j in range(4)] for i in range(4)])
        Ti, Ci, I4 = nint(T), nint(C), mp.eye(4)
        PiG = M * mp.matrix([y[0] for y in YG])
        PiG = PiG / PiG[2]
        out = {"a2": a2, "devC": devC, "devT": devT, "T": Ti, "C": Ci, "PiG": PiG, "GV": self.GV}
        for name, A in (("(CT)^-1", (Ci * Ti) ** -1), ("(TC)^-1", (Ti * Ci) ** -1)):
            if mp.norm(A ** d - I4) > 1e-15:
                continue
            v = A * PiG
            lam = v[2]
            res = mp.norm(v - lam * PiG) / mp.norm(PiG)
            if res > 1e-3:
                continue
            def poly(coeffs):
                P = mp.matrix(4, 4)
                for j, c in enumerate(coeffs):
                    cc = mp.mpf(c.numerator) / c.denominator if isinstance(c, Fr) else c
                    P += cc * A ** j
                return P
            W = poly(line_coeffs)
            fus = None
            if fusion_terms:
                S = mp.matrix(4, 4)
                for co in fusion_terms:
                    S += poly(co)
                fus = mp.norm(W * W - S)
            Wv_ = W * PiG
            eig = Wv_[2]
            lv = []
            for tt in (1.5, 2.0, 3.0):
                Pt = mp.matrix(self.Pi_LV(mp.mpc(0, tt), a2))
                lv.append((tt, mp.norm(W * Pt - eig * Pt) / mp.norm(Pt)))
            out.update({"A": A, "Aname": name, "eigA": lam, "resA": res, "W": W,
                        "W_int": all(abs(W[i, j] - mp.nint(mp.re(W[i, j]))) < 1e-12 for i in range(4) for j in range(4)),
                        "eigW": eig, "resW": mp.norm(Wv_ - eig * PiG) / mp.norm(PiG), "largevol": lv,
                        "fusion": fus})
            break
        return out


def _poly_prod(factors):
    """Coefficient list of prod (a + b D) for factors [a, b]."""
    co = [1]
    for a, b in factors:
        new = [0] * (len(co) + 1)
        for j, c in enumerate(co):
            new[j] += a * c
            new[j + 1] += b * c
        co = new
    return co


# ----------------------------------------------------------------------------- report
def _fmt_mon(a):
    return "x^(" + ",".join(map(str, a)) + ")"


def enumerative_text(model: Model, factor: int, r: int, numerics: bool = False, h11: Optional[int] = None) -> str:
    n = fermat_exponents(model)
    k = n[factor] - 2
    if not 0 < r < k:
        raise ValueError(f"factor {factor + 1} has k = {k}: choose 0 < r < k for a non-invertible line")
    dL = line_dim(k, r)
    L = [f"ENUMERATIVE TDL ANALYSIS of {model.name}   n = {n}, d = {model.korder}",
         f"line: factor {factor + 1} (k = {k}), r = {r}, d = {dL:.4f}" + ("   [odd r: needs the Cordova-Rizi dressing]" if r % 2 else ""),
         "Tags: [verified] computed here, [literature] quoted, [inferred] argued, [missing] not computed.", ""]
    # 1
    cp = charge_polynomial(model, factor, r)
    L.append("1. BRANE CHARGES (narrow sectors J^nu = RR charges of the Kahler sector)")
    L.append(f"   narrow sectors: {cp['m']}" + (f"   (2 + 2 h11 = {2 + 2 * h11})" if h11 is not None else ""))
    for (nu, l), e in zip(cp["sectors"], cp["eigen"]):
        L.append(f"     nu = {nu:3d}  labels {l}  A-character e^(2 pi i {nu}/{model.korder})  line eigenvalue {e:+.6f}")
    if cp["rational"]:
        terms = " + ".join(f"({c}) A^{j}" for j, c in enumerate(cp["coeffs"]) if c)
        L.append(f"   line on charges = {terms}   [verified; rational coefficients" + (", integer]" if cp["integer"] else "]"))
    else:
        L.append(f"   line on charges = polynomial in A with complex coefficients {[complex(round(c.real, 6), round(c.imag, 6)) for c in cp['coeffs']]}")
        L.append("   (not Galois equivariant: an odd line needs its dressing before it maps branes to branes)")
    L.append("   (unique representative of degree < %d; forms differing by prod_nu (A - e^(2 pi i nu/d)) act identically)" % cp["m"])
    L.append("   Every narrow sector has one ground state and a distinct A-character, so the line is a polynomial in the")
    L.append("   invertible Gepner monodromy A on these charges: no information beyond Z_d [verified].")
    if h11 is not None and cp["m"] < 2 + 2 * h11:
        L.append(f"   NOTE: {2 + 2 * h11 - cp['m']} Kahler-sector charges come from broad sectors; there a line can differ from p(A) [missing].")
    L.append("")
    # 2
    kc = KahlerCouplings(model)
    an = kc.analyse(factor)
    L.append("2. KAHLER COUPLINGS AT THE GEPNER POINT (genus-0 A-model = mirror chiral ring at its Fermat point)")
    L.append(f"   polynomial Kahler moduli (mirror untwisted 27s): {len(kc.moduli)}" + (f" of h11 = {h11}" if h11 is not None else ""))
    for a in kc.moduli:
        e = line_dim(k, r, a[factor])
        L.append(f"     {_fmt_mon(a)}   label in factor {factor + 1}: l = {a[factor]}, line eigenvalue {e:+.4f}" + ("  (+d)" if abs(e - dL) < 1e-9 else "  (-d)" if abs(e + dL) < 1e-9 else ""))
    L.append(f"   3-point kappa_abc(t_G): {len(an['three'])} allowed by the abelian symmetries, all non-zero (Fermat ring) [verified]")
    rows = an["rows"]
    nz = sum(1 for x in rows if x[1])
    z = [x for x in rows if not x[1]]
    L.append(f"   4-point d kappa_abc(t_G): {len(rows)} allowed by all phase/U(1) symmetries: {nz} non-zero, {len(z)} zero [verified]")
    L.append(f"     consistency: every non-zero one obeys the strict fusion rule of factor {factor + 1}: {an['consistent']}")
    for q, _, strict, expl, sf in z:
        tag = "forbidden by the fusion rule of factor %d" % (factor + 1) if not strict else "allowed by the fusion rule of factor %d" % (factor + 1)
        fl = forbidding_lines(k, [a[factor] for a in q], sf == factor)
        if fl:
            tag += f" (lines r = {fl}; chosen r = {r}: {'forbids it' if r in fl or k - r in fl else 'does not'})"
        L.append(f"     zero: {' '.join(_fmt_mon(a) for a in q)}  (special factor {sf + 1})  {tag};  "
                 + ("EFT lattice: an abelian charge assignment explains it" if expl else "EFT lattice: NO abelian charge assignment explains it"))
        if an["u1"][q]:
            L.append(f"        INVERTIBLE: an entry avoids factor {sf + 1}, so the U(1) current of that factor is exact along it and the")
            L.append("        3-point function of the other three (charged) vanishes identically; F_abcd = d_e kappa_rest = 0 [verified rule]")
    L.append(f"   zeros forbidden by the line and allowed by every phase symmetry of the Gepner point: {len(an['beyond_phase'])} [verified]")
    L.append(f"   ... of these, forced by a factor U(1) exact along one entry (symmetry of F_abcd): "
             f"{sum(1 for x in an['beyond_phase'] if an['u1'][x[0]])} [verified]")
    L.append(f"   ... of these, neither U(1)-forced nor explainable by ANY abelian charge assignment compatible with the")
    L.append(f"   {an['n_nonzero']} non-zero 3- and 4-point couplings (EFT lattice test): {len(an['genuine'])} [verified]")
    if an["beyond_phase"] and not an["genuine"]:
        L.append("   NOTE: the lattice test only sees the computed couplings; with few non-zero couplings it 'explains' any zero by an")
        L.append("   accidental charge assignment that need not be a symmetry of the CFT. Such zeros are line zeros not forbidden by")
        L.append("   the Gepner phase symmetries, but not proven beyond every invertible symmetry [inferred].")
    if r % 2:
        L.append("   NOTE: odd r. The dressing (needed for consistency with field identification) can turn the line's sign on")
        L.append("   even labels into a phase times a factor-wise fermion number; whether that is a symmetry of the model is [missing].")
    L.append(f"   GENUINE line zeros of genus-0 enumerative data at the Gepner point: {len(an['genuine'])}")
    if an["genuine"]:
        L.append("   Meaning: in a basis of divisor classes D_a, kappa_abc(t) = kappa^0_abc + sum_beta n_beta beta_a beta_b beta_c q^beta/(1-q^beta);")
        L.append("   its derivative, analytically continued to t_G and written in the label basis above, vanishes for these entries.")
        L.append("   The change from the label basis to the divisor basis needs the mirror map at t_G [missing].")
    L.append("   Higher-point (5-point and up) Kahler couplings not computed [missing].")
    L.append("")
    # 3
    sv = kc.survival(factor, r)
    L.append("3. SURVIVAL ALONG KAHLER DEFORMATIONS")
    gname = "L_%d" % r if sv["phase"] == 0 else "L_%d composed with x%d -> e^(2 pi i %d/%d) x%d" % (r, factor + 1, sv["phase"], sv["n"], factor + 1)
    L.append(f"   kept by {gname}: {len(sv['kept'])} of {len(kc.moduli)} polynomial Kahler moduli: {', '.join(_fmt_mon(a) for a in sv['kept']) or 'none'}")
    if sv["phase"]:
        L.append("   (the phase undoes the sign -d of the bare line on some moduli; the composite has the same zeros, since the")
        L.append("   phase alone allows every coupling counted in section 2)")
    L.append("   The line survives the family spanned by them (Cordova-Rizi criterion) [inferred: first order]; every zero it")
    L.append("   forbids stays zero along that family [inferred], so it constrains the instanton sums of that family, not of the")
    L.append("   full large-volume limit unless all Kahler moduli (twisted ones included) are kept.")
    L.append("   Twisted Kahler moduli are not covered here; `lgor lines` includes them.")
    L.append("")
    # 4
    if numerics:
        L.append("4. ONE-PARAMETER NUMERICS")
        try:
            op = OneParameter(model)
        except ValueError as e:
            L.append(f"   skipped: {e}")
            return "\n".join(L)
        gv, gerr = op.GV
        L.append(f"   H^3 = {op.kappa}, c2.H = {op.c2H}, chi = {op.chi}, mu = {op.mu}, narrow nu = {op.nus}")
        L.append(f"   GV invariants (B-model, rounded; max distance to integer {float(gerr):.1e}): {gv}")
        if not cp["rational"]:
            L.append("   skipped: the line is not a rational polynomial in A")
            return "\n".join(L)
        terms = sorted(fuse(k, r, r))
        fcs = [charge_polynomial(model, factor, s_)["coeffs"] if s_ else [Fr(1)] for s_ in terms]
        res = op.run(cp["coeffs"], fusion_terms=fcs if all(isinstance(c, Fr) for co in fcs for c in co) else ())
        mp = op.mp
        L.append(f"   integral basis Pi = w0 (F_0, F_1, 1, t), F_tt-coefficient a2 = {res['a2']}; T integer to {mp.nstr(res['devT'], 2)}, C integer to {mp.nstr(res['devC'], 2)}")
        L.append(f"   T = {[[int(res['T'][i, j]) for j in range(4)] for i in range(4)]}")
        L.append(f"   C = {[[int(res['C'][i, j]) for j in range(4)] for i in range(4)]}")
        L.append(f"   Gepner point: t_G = {mp.nstr(res['PiG'][3], 12)}")
        if "A" in res:
            A = res["A"]
            L.append(f"   A = {res['Aname']} = {[[int(mp.nint(mp.re(A[i, j]))) for j in range(4)] for i in range(4)]},  A^{op.d} = 1 [verified]")
            L.append(f"   Pi(t_G) is an A-eigenvector: residual {mp.nstr(res['resA'], 3)}, eigenvalue {mp.nstr(res['eigA'], 8)}")
            W = res["W"]
            L.append(f"   line on charges W = p(A) = {[[mp.nstr(mp.re(W[i, j]), 6) for j in range(4)] for i in range(4)]}  integer: {res['W_int']}")
            L.append(f"   W Pi(t_G) = {mp.nstr(res['eigW'], 10)} Pi(t_G): residual {mp.nstr(res['resW'], 3)}")
            if res["fusion"] is not None:
                L.append(f"   fusion on charges: W_{r} W_{r} = " + " + ".join(f"W_{s_}" for s_ in terms)
                         + f"  (W_0 = 1): |difference| = {mp.nstr(res['fusion'], 3)} [verified]")
            for tt, dv in res["largevol"]:
                L.append(f"   large volume t = {tt}i: |W Pi - eig Pi|/|Pi| = {mp.nstr(dv, 4)}")
        else:
            L.append("   no monodromy ordering gave A^d = 1 with Pi(t_G) an eigenvector [check failed]")
    return "\n".join(L)
