"""Graded matrix factorisations of W(x) - W(x') for a quasi-homogeneous LG block (LGOR 3.14, package mfcat).

Conventions
-----------
* An LG block has N fields x_1..x_N with charges q_i (W has charge 1).  Integer grading unit: D = lcm of the
  denominators of the q_i; then x_i has integer charge w_i = 2 q_i D, W has charge 2D and every matrix
  factorisation Q of W(x) - W(x') has charge D.
* Polynomials are dicts {exponent tuple of length 2N: complex}; positions 0..N-1 are x, N..2N-1 are x'.
* A defect (B-type line) is an MF Q of W(x) - W(x') with Z2 parity par[i] and integer charge deg[i] on each basis
  vector; entry Q[i][j] (map j -> i) is homogeneous of charge D + deg[i] - deg[j].
* Bulk action on RR ground states (= Jacobian ring by spectral flow):
      D(phi)(x) = Res_{x'} [ phi(x') str(dQ/dx'_1 ... dQ/dx'_N dQ/dx_1 ... dQ/dx_N) / prod_i dW/dx'_i ],
  normalised so that the identity defect acts as 1 (formula validated on Fermat blocks, gepner_survey/cm_points/
  mfdef.py; tags: [verified] on identity and phase defects of every block in the tests).
  On RR ground states an invertible phase defect g acts as det(g)^-1 g (spectral-flow factor), so the NS-normalised
  symmetry operator is det(g) * action.
"""
from fractions import Fraction as Fr
from math import gcd
from functools import reduce
from collections import defaultdict
import itertools
import cmath
import numpy as np

TOL = 1e-10


# ------------------------------------------------------------------------------------------------ polynomials
def padd(p, q, s=1):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + s * v
    return {k: v for k, v in r.items() if abs(v) > 1e-14}


def pscale(p, c):
    return {k: c * v for k, v in p.items()} if c else {}


def pmul(p, q):
    if not p or not q:
        return {}
    r = defaultdict(complex)
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            r[tuple(a + b for a, b in zip(k1, k2))] += v1 * v2
    return {k: v for k, v in r.items() if abs(v) > 1e-14}


def ppow(p, n, nv):
    r = {(0,) * nv: 1.0}
    for _ in range(n):
        r = pmul(r, p)
    return r


def pdiff(p, i):
    r = {}
    for k, v in p.items():
        if k[i]:
            kk = list(k)
            kk[i] -= 1
            r[tuple(kk)] = r.get(tuple(kk), 0) + v * k[i]
    return r


def mdiff(M, i):
    return [[pdiff(q, i) for q in row] for row in M]


def mmul(A, B):
    n, m, l = len(A), len(B), len(B[0])
    C = [[{} for _ in range(l)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            if not A[i][k]:
                continue
            for j in range(l):
                if B[k][j]:
                    C[i][j] = padd(C[i][j], pmul(A[i][k], B[k][j]))
    return C


# ------------------------------------------------------------------------------------------------ LG block
class LG:
    """Quasi-homogeneous W in N variables.  W: dict {N-tuple: coefficient}; q: list of Fractions."""

    def __init__(self, W, q, names=None):
        self.N = len(q)
        self.q = [Fr(x) for x in q]
        self.D = reduce(lambda a, b: a * b // gcd(a, b), [x.denominator for x in self.q], 1)
        self.w = [int(2 * x * self.D) for x in self.q]
        self.W = {tuple(k): complex(v) for k, v in W.items()}
        for k in self.W:
            assert sum(a * b for a, b in zip(k, self.w)) == 2 * self.D, ("W not quasi-homogeneous", k)
        self.names = names or ["x%d" % (i + 1) for i in range(self.N)]
        self.chat = sum(1 - 2 * x for x in self.q)
        N = self.N
        self.Wx = {k + (0,) * N: v for k, v in self.W.items()}
        self.Wxp = {(0,) * N + k: v for k, v in self.W.items()}
        self.WW = padd(self.Wx, self.Wxp, -1)
        self._jac = None

    @classmethod
    def fermat(cls, ns):
        N = len(ns)
        W = {}
        for i, n in enumerate(ns):
            e = [0] * N
            e[i] = n
            W[tuple(e)] = 1
        return cls(W, [Fr(1, n) for n in ns], ["x%d" % (i + 1) for i in range(N)])

    @classmethod
    def from_string(cls, s, names):
        import sympy as sp
        syms = sp.symbols(names)
        P = sp.Poly(sp.sympify(s), *syms)
        W = {m: complex(c) for m, c in P.terms()}
        # charges from homogeneity: solve sum_i m_i q_i = 1 over the monomials
        A = sp.Matrix([list(m) for m in W])
        sol = sp.linsolve((A, sp.ones(A.rows, 1)))
        (qs,) = list(sol)
        assert all(x.free_symbols == set() for x in qs), "charges not fixed by W; give q explicitly"
        return cls(W, [Fr(str(x)) for x in qs], list(names))

    def charge(self, m):
        """integer charge of an exponent tuple of length N or 2N"""
        w = self.w if len(m) == self.N else self.w + self.w
        return sum(a * b for a, b in zip(m, w))

    # ---- monomials of given integer charge in n = N or 2N variables
    def monos(self, c, two=True):
        w = self.w + self.w if two else self.w
        out = []

        def rec(i, rem, cur):
            if i == len(w):
                if rem == 0:
                    out.append(tuple(cur))
                return
            for a in range(rem // w[i] + 1):
                cur.append(a)
                rec(i + 1, rem - a * w[i], cur)
                cur.pop()
        if c >= 0:
            rec(0, c, [])
        return out

    # ---- Jacobian ring (one side)
    def jac(self):
        if self._jac is None:
            self._jac = Jacobian(self)
        return self._jac


class Jacobian:
    """Graded Jacobian ring C[x]/(dW) by numerical linear algebra per charge; residue functional."""

    def __init__(self, lg):
        self.lg = lg
        N, w = lg.N, lg.w
        self.dW = [{k[:N]: v for k, v in pdiff(lg.Wx, i).items()} for i in range(N)]
        self.top = int(round(lg.chat * 2 * lg.D))      # charge of the Hessian
        self.pieces = {}
        self.basis = []
        for c in range(self.top + 1):
            mons = lg.monos(c, two=False)
            if not mons:
                continue
            idx = {m: k for k, m in enumerate(mons)}
            gens = []
            for i in range(N):
                cd = c - (2 * lg.D - w[i])
                for m in lg.monos(cd, two=False):
                    v = np.zeros(len(mons), complex)
                    for k, val in self.dW[i].items():
                        v[idx[tuple(a + b for a, b in zip(k, m))]] += val
                    gens.append(v)
            J = np.array(gens).T if gens else np.zeros((len(mons), 0))
            # choose basis monomials greedily: monomials not in the span of J + previous choices
            chosen = []
            cur = J.copy()
            r0 = np.linalg.matrix_rank(cur, tol=1e-9) if cur.size else 0
            for k, m in enumerate(mons):
                e = np.zeros((len(mons), 1))
                e[k] = 1
                test = np.hstack([cur, e]) if cur.size else e
                r = np.linalg.matrix_rank(test, tol=1e-9)
                if r > r0:
                    chosen.append(m)
                    cur, r0 = test, r
            self.pieces[c] = (mons, idx, J, chosen)
            self.basis += chosen
        self.mu = len(self.basis)
        self.bindex = {m: k for k, m in enumerate(self.basis)}
        H = {(0,) * N: 1.0}
        Hm = [[pdiff(self.dW[i], j) for j in range(N)] for i in range(N)]
        self.hess = self._det(Hm)
        h = self.nf(self.hess)
        assert len(h) == 1 and self.top in self.pieces and len(self.pieces[self.top][3]) == 1, "socle not 1-dim"
        (self.socle, self.hcoef), = h.items()

    def _det(self, M):
        n = len(M)
        if n == 1:
            return M[0][0]
        r = {}
        for j in range(n):
            minor = [row[:j] + row[j + 1:] for row in M[1:]]
            r = padd(r, pmul(M[0][j], self._det(minor)), (-1) ** j)
        return r

    def nf(self, p):
        """normal form: dict {basis monomial: coef}"""
        bych = defaultdict(dict)
        for k, v in p.items():
            bych[self.lg.charge(k)][k] = v
        out = {}
        for c, part in bych.items():
            if c not in self.pieces:
                continue
            mons, idx, J, chosen = self.pieces[c]
            v = np.zeros(len(mons), complex)
            for k, val in part.items():
                v[idx[k]] += val
            B = np.zeros((len(mons), len(chosen)))
            for t, m in enumerate(chosen):
                B[idx[m], t] = 1
            A = np.hstack([B, J]) if J.size else B
            sol = np.linalg.lstsq(A, v, rcond=None)[0]
            for t, m in enumerate(chosen):
                if abs(sol[t]) > 1e-11:
                    out[m] = out.get(m, 0) + sol[t]
        return out

    def res(self, p):
        """Grothendieck residue Res[p dx / prod dW] (normalised Res[Hess] = mu)"""
        n = self.nf(p)
        return self.mu * n.get(self.socle, 0) / self.hcoef


# ------------------------------------------------------------------------------------------------ MFs
class MF:
    def __init__(self, lg, Q, par, deg, name=""):
        self.lg, self.Q, self.par, self.deg, self.name = lg, Q, list(par), list(deg), name
        self.n = len(self.par)

    def check(self, tol=1e-8):
        lg, n = self.lg, self.n
        for i in range(n):
            for j in range(n):
                q = self.Q[i][j]
                if q:
                    if self.par[i] == self.par[j]:
                        return False, "parity (%d,%d)" % (i, j)
                    for k in q:
                        if lg.charge(k) != lg.D + self.deg[i] - self.deg[j]:
                            return False, "charge (%d,%d)" % (i, j)
        Q2 = mmul(self.Q, self.Q)
        for i in range(n):
            for j in range(n):
                acc = padd(Q2[i][j], lg.WW, -1) if i == j else Q2[i][j]
                if any(abs(v) > tol for v in acc.values()):
                    return False, "Q^2 != W-W' at (%d,%d)" % (i, j)
        return True, "ok"


def koszul(lg, fs, gs, name=""):
    """Koszul MF on generators f_i (polys in 2N vars) with Q^2 = sum f_i g_i = W - W'."""
    n = len(fs)
    basis = [b for r in range(n + 1) for b in itertools.combinations(range(n), r)]
    idx = {b: i for i, b in enumerate(basis)}
    N2 = len(basis)
    Q = [[{} for _ in range(N2)] for _ in range(N2)]
    for j, b in enumerate(basis):
        for i in range(n):
            s = (-1) ** sum(1 for k in b if k < i)
            if i not in b:
                Q[idx[tuple(sorted(b + (i,)))]][j] = padd(Q[idx[tuple(sorted(b + (i,)))]][j], pscale(fs[i], s))
            else:
                Q[idx[tuple(k for k in b if k != i)]][j] = padd(Q[idx[tuple(k for k in b if k != i)]][j], pscale(gs[i], s))
    qf = []
    for f in fs:
        cs = {lg.charge(k) for k in f}
        assert len(cs) == 1, "generator not homogeneous"
        qf.append(cs.pop())
    deg = [sum(qf[i] - lg.D for i in b) for b in basis]
    return MF(lg, Q, [len(b) % 2 for b in basis], deg, name)


def solve_gs(lg, fs):
    """find homogeneous g_i with sum f_i g_i = W(x) - W(x') (linear algebra); None if impossible"""
    unk = []
    for i, f in enumerate(fs):
        cf = lg.charge(next(iter(f)))
        for m in lg.monos(2 * lg.D - cf):
            unk.append((i, m))
    tgt = lg.monos(2 * lg.D)
    ti = {m: k for k, m in enumerate(tgt)}
    A = np.zeros((len(tgt), len(unk)), complex)
    for col, (i, m) in enumerate(unk):
        for k, v in fs[i].items():
            A[ti[tuple(a + b for a, b in zip(k, m))], col] += v
    b = np.zeros(len(tgt), complex)
    for k, v in lg.WW.items():
        b[ti[k]] += v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.linalg.norm(A @ sol - b) > 1e-8:
        return None
    gs = [{} for _ in fs]
    for c, (i, m) in zip(sol, unk):
        if abs(c) > 1e-12:
            gs[i][m] = gs[i].get(m, 0) + c
    return gs


def ideal_defect(lg, fs, name=""):
    """Koszul defect of the ideal (f_1..f_N) in C[x,x']: needs W - W' in the ideal."""
    gs = solve_gs(lg, fs)
    if gs is None:
        raise ValueError("W(x) - W(x') is not in the ideal generated by %s" % name)
    return koszul(lg, fs, gs, name)


def _xv(lg, i, c=1.0, prime=False):
    e = [0] * (2 * lg.N)
    e[i + (lg.N if prime else 0)] = 1
    return {tuple(e): c}


def identity_defect(lg):
    return phase_defect(lg, [0] * lg.N, "1")


def phase_defect(lg, phases, name=None):
    """invertible defect of x_i -> e^{2 pi i phases_i} x_i (must preserve W): ideal (x_i - g_i x'_i)"""
    g = [cmath.exp(2j * cmath.pi * float(Fr(p))) for p in phases]
    for k, v in lg.W.items():
        ph = np.prod([g[i] ** k[i] for i in range(lg.N)])
        assert abs(ph - 1) < 1e-9, "phase does not preserve W"
    fs = [padd(_xv(lg, i), _xv(lg, i, g[i], True), -1) for i in range(lg.N)]
    return ideal_defect(lg, fs, name or "g" + str(tuple(str(Fr(p)) for p in phases)))


def perm_defect(lg, perm, phases=None, name=None):
    """invertible defect x_i -> phase_i * x_{perm(i)}: ideal (x_i - g_i x'_{perm i})"""
    phases = phases or [0] * lg.N
    g = [cmath.exp(2j * cmath.pi * float(Fr(p))) for p in phases]
    fs = [padd(_xv(lg, i), _xv(lg, perm[i], g[i], True), -1) for i in range(lg.N)]
    return ideal_defect(lg, fs, name or "perm%s" % (tuple(perm),))


def parse_ideal(lg, gens, params=None):
    """gens: list of strings in the block variable names and primed names (name + 'p'), e.g. 'x1^2 - a*x4p'."""
    import sympy as sp
    names = lg.names + [n + "p" for n in lg.names]
    syms = sp.symbols(names)
    loc = dict(zip(names, syms))
    loc.update(params or {})
    fs = []
    for s in gens:
        P = sp.Poly(sp.sympify(s.replace("^", "**"), locals=loc), *syms)
        fs.append({m: complex(c) for m, c in P.terms()})
    return fs


def direct_sum(*ms):
    lg = ms[0].lg
    n = sum(m.n for m in ms)
    Q = [[{} for _ in range(n)] for _ in range(n)]
    par, deg, o = [], [], 0
    for m in ms:
        for i in range(m.n):
            for j in range(m.n):
                Q[o + i][o + j] = m.Q[i][j]
        par += m.par
        deg += m.deg
        o += m.n
    return MF(lg, Q, par, deg, " + ".join(m.name for m in ms))


def shift(m, flip=True, s=0):
    """m[1] (parity flip, Q -> -Q) and/or grading shift by s"""
    Q = [[pscale(q, -1) if flip else q for q in row] for row in m.Q]
    return MF(m.lg, Q, [(p + flip) % 2 for p in m.par], [d + s for d in m.deg], m.name + ("[1]" if flip else ""))


def cone(A, B, phi, name=None):
    """cone of an even closed morphism phi: A -> B of charge D (degree 0): Q = [[-Q_A, 0], [phi, Q_B]] on A[1]+B.
    Its K-class is [B] - [A]; the bulk action is additive: D_cone = D_B - D_A."""
    n = A.n + B.n
    Q = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(A.n):
        for j in range(A.n):
            Q[i][j] = pscale(A.Q[i][j], -1)
    for i in range(B.n):
        for j in range(B.n):
            Q[A.n + i][A.n + j] = B.Q[i][j]
        for j in range(A.n):
            Q[A.n + i][j] = phi[i][j]
    return MF(A.lg, Q, [(p + 1) % 2 for p in A.par] + B.par, A.deg + B.deg, name or "Cone(%s -> %s)" % (A.name, B.name))
