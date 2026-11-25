"""General matrix-factorisation search and Fourier (rotation) defects (LGOR 3.14 mfcat).

mf_search  : numerical search for graded MFs Q of W(x) - W(x') of a given rank and grading, with arbitrary homogeneous
             polynomial entries (not only Koszul/ideal defects).  Q^2 = (W - W')*1 is a system of quadratic equations
             in the coefficients; it is solved by Levenberg-Marquardt from random complex starts.  Each converged
             solution is polished, checked exactly (MF.check) and classified by its bulk action (the bulk action is
             constant on connected families, so the distinct actions label the components that were hit).
gradings   : candidate gradings (even/odd basis charges) of rank-r MFs of a block.
rotation_defect : 'Fourier' defects for a quadric pair x_i^{2a} + x_j^{2b} = U^2 + V^2 (U = x_i^a, V = x_j^b of equal
             charge): the ideal (U - R11 U' - R12 V', V - R21 U' - R22 V') with R in O(2).  R = Hadamard
             (rotation by pi/4) is the T-duality/Fourier transform on the fixed points of the corresponding torus.
"""
import cmath
import itertools as it
import numpy as np
from fractions import Fraction as Fr
from .core import MF, ideal_defect, padd, mmul
from .hom import bulk_action


# ------------------------------------------------------------------------------------------------ Fourier defects
def quadric_pairs(lg):
    """pairs (i, j, a, b) with pure powers x_i^{2a}, x_j^{2b} (coefficient 1) in W"""
    pure = {}
    for m, c in lg.W.items():
        nz = [i for i, e in enumerate(m) if e]
        if len(nz) == 1 and abs(c - 1) < 1e-12:
            pure[nz[0]] = m[nz[0]]
    return [(i, j, pure[i] // 2, pure[j] // 2) for i in pure for j in pure
            if i < j and pure[i] % 2 == 0 and pure[j] % 2 == 0]


def rotation_defect(lg, i, j, a, b, theta, reflect=False, phases=None, name=None):
    """ideal (U - R11 U' - R12 V', V - R21 U' - R22 V', x_k - g_k x_k') with U = x_i^a, V = x_j^b and
    R = rotation by theta (or reflection if reflect); W - W' = U^2 + V^2 - U'^2 - V'^2 + ... lies in the ideal
    because R is orthogonal.  theta = pi/4 is the Hadamard ('Fourier') defect."""
    N = lg.N
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [s, -c]]) if reflect else np.array([[c, -s], [s, c]])
    g = [cmath.exp(2j * cmath.pi * float(Fr(t))) for t in (phases or [0] * N)]

    def mono(k, e, prime):
        t = [0] * (2 * N)
        t[k + (N if prime else 0)] = e
        return tuple(t)
    U, V, Up, Vp = mono(i, a, 0), mono(j, b, 0), mono(i, a, 1), mono(j, b, 1)
    fs = [{U: 1.0, Up: -R[0, 0], Vp: -R[0, 1]}, {V: 1.0, Up: -R[1, 0], Vp: -R[1, 1]}]
    fs = [{k: v for k, v in f.items() if abs(v) > 1e-14} for f in fs]
    for k in range(N):
        if k not in (i, j):
            fs.append({mono(k, 1, 0): 1.0, mono(k, 1, 1): -g[k]})
    tag = "refl" if reflect else "rot"
    return ideal_defect(lg, fs, name or "%s(x%d^%d,x%d^%d; %s pi)" % (tag, i + 1, a, j + 1, b,
                                                                     Fr(theta / np.pi).limit_denominator(48)))


def fourier_library(lg, angles=(np.pi / 4,), with_reflections=True):
    """Fourier-type defects of every quadric pair: rotations by the given angles (default Hadamard) and the
    corresponding reflections"""
    out = []
    for (i, j, a, b) in quadric_pairs(lg):
        for th in angles:
            out.append(("fourier", (i, j, a, b, th, False), rotation_defect(lg, i, j, a, b, th)))
            if with_reflections:
                out.append(("fourier", (i, j, a, b, th, True), rotation_defect(lg, i, j, a, b, th, True)))
    return out


# ------------------------------------------------------------------------------------------------ general MF search
class _System:
    """bilinear system Q(c)^2 - (W - W') 1 = 0 for a rank-(r_e + r_o) MF with fixed grading"""

    def __init__(self, lg, deg_even, deg_odd, sparsity=None):
        self.lg = lg
        D = lg.D
        self.par = [0] * len(deg_even) + [1] * len(deg_odd)
        self.deg = list(deg_even) + list(deg_odd)
        n = self.n = len(self.par)
        unk = []
        for i in range(n):
            for j in range(n):
                if self.par[i] == self.par[j]:
                    continue
                for m in lg.monos(D + self.deg[i] - self.deg[j]):
                    unk.append((i, j, m))
        self.unk = unk
        self.nu = len(unk)
        # equations: (i, k, monomial) for Q^2_{ik}, monomials of charge 2D + deg i - deg k
        eqs = {}
        trip = []
        byrow = {}
        for a, (i, j, m) in enumerate(unk):
            byrow.setdefault(i, []).append(a)
        for a, (i, j, m) in enumerate(unk):
            for b in byrow.get(j, []):
                _, k, mm = unk[b]
                key = (i, k, tuple(x + y for x, y in zip(m, mm)))
                if key not in eqs:
                    eqs[key] = len(eqs)
                trip.append((eqs[key], a, b))
        rhs = np.zeros(len(eqs) + 0, complex)
        for i in range(n):
            for k, v in lg.WW.items():
                key = (i, i, k)
                if key not in eqs:            # W - W' monomial not reachable on this diagonal: infeasible grading
                    eqs[key] = len(eqs)
                    rhs = np.append(rhs, 0)
                rhs[eqs[key]] += v
        self.ne = len(eqs)
        self.rhs = rhs
        t = np.array(trip, dtype=int).reshape(-1, 3)
        self.E, self.A, self.B = t[:, 0], t[:, 1], t[:, 2]
        self.eqkeys = eqs

    def feasible(self):
        """every basis vector must have a nonzero row and column possible, and every W - W' term reachable"""
        rows = {i for (i, j, m) in self.unk}
        cols = {j for (i, j, m) in self.unk}
        if len(rows) < self.n or len(cols) < self.n:
            return False
        reach = set(self.E.tolist())
        return all(self.eqkeys[(i, i, k)] in reach for i in range(self.n) for k in self.lg.WW)

    def res(self, c):
        r = np.zeros(self.ne, complex)
        np.add.at(r, self.E, c[self.A] * c[self.B])
        return r - self.rhs

    def jac(self, c):
        J = np.zeros((self.ne, self.nu), complex)
        np.add.at(J, (self.E, self.A), c[self.B])
        np.add.at(J, (self.E, self.B), c[self.A])
        return J

    def mf(self, c, name="Q"):
        Q = [[{} for _ in range(self.n)] for _ in range(self.n)]
        for coef, (i, j, m) in zip(c, self.unk):
            if abs(coef) > 1e-11:
                Q[i][j] = padd(Q[i][j], {m: complex(coef)})
        return MF(self.lg, Q, self.par, self.deg, name)


def _lm(sys, c0, iters=400, tol=1e-13):
    """complex Levenberg-Marquardt (holomorphic residual), then Newton polish"""
    c = c0.copy()
    lam = 1e-2
    r = sys.res(c)
    f = np.vdot(r, r).real
    for _ in range(iters):
        J = sys.jac(c)
        JH = J.conj().T
        g = JH @ r
        H = JH @ J
        while True:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H).real + 1e-12), -g)
            cn = c + step
            rn = sys.res(cn)
            fn = np.vdot(rn, rn).real
            if fn < f:
                c, r, f = cn, rn, fn
                lam = max(lam / 3, 1e-12)
                break
            lam *= 4
            if lam > 1e12:
                return c, f
        if f < tol:
            break
    # Newton polish (minimum-norm steps)
    for _ in range(20):
        J = sys.jac(c)
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        c = c + step
        r = sys.res(c)
        f = np.vdot(r, r).real
        if f < 1e-26:
            break
    return c, f


def mf_search(lg, deg_even, deg_odd, starts=20, seed=0, scale=1.0, known=None, verbose=False, max_unknowns=400):
    """random-start search for MFs of W(x) - W(x') with the given grading.  Returns a list of
    (MF, bulk action) with distinct bulk actions (also distinct from `known`, a list of matrices)."""
    sys = _System(lg, deg_even, deg_odd)
    if not sys.feasible() or sys.nu == 0 or sys.nu > max_unknowns:
        return []
    rng = np.random.default_rng(seed)
    found = []
    acts = list(known or [])
    for s in range(starts):
        c0 = scale * (rng.normal(size=sys.nu) + 1j * rng.normal(size=sys.nu)) / np.sqrt(2)
        c, f = _lm(sys, c0)
        if f > 1e-18:
            continue
        X = sys.mf(c, "MF%s|%s#%d" % (tuple(deg_even), tuple(deg_odd), s))
        ok, _ = X.check(tol=1e-7)
        if not ok:
            continue
        A = bulk_action(X)
        if np.max(np.abs(A)) < 1e-8:          # acts as zero: K-trivial (e.g. a contractible piece)
            continue
        if any(np.allclose(A, B, atol=1e-6) for B in acts):
            continue
        acts.append(A)
        found.append((X, A))
        if verbose:
            print("   new component from grading %s|%s: vacuum eigenvalue %s" % (deg_even, deg_odd,
                                                                                np.round(A[0, 0], 4)))
    return found


def gradings(lg, rank, dmax=None):
    """candidate gradings of rank-`rank` MFs (rank even + rank odd basis vectors), up to an overall shift
    (first even degree 0) and permutations within each parity: all degree vectors with entries in
    [-dmax, dmax] for which the system is feasible (every basis vector can carry a nonzero row and column and
    W - W' is reachable on every diagonal entry of Q^2)"""
    D = lg.D
    dmax = 2 * D if dmax is None else dmax
    vals = range(-dmax, dmax + 1)
    out = []
    for ev in it.combinations_with_replacement(vals, rank - 1):
        e = (0,) + ev
        if list(e) != sorted(e) and rank > 1 and min(ev) < 0:
            pass
        for o in it.combinations_with_replacement(vals, rank):
            s = _System(lg, e, o)
            if s.feasible():
                out.append((e, o))
    return out


def koszul_gradings(lg, nf=None):
    """gradings of Koszul MFs on nf generators of charges a_1..a_nf (deg e_S = sum_{i in S} (a_i - D)):
    all multisets of generator charges with 0 < a_i < 2D that are charges of monomials in x, x'"""
    D = lg.D
    nf = lg.N if nf is None else nf
    ch = sorted({c for c in range(1, 2 * D) if lg.monos(c)})
    out = []
    for qs in it.combinations_with_replacement(ch, nf):
        e, o = [], []
        for r in range(nf + 1):
            for S in it.combinations(range(nf), r):
                d = sum(qs[i] - D for i in S)
                (o if r % 2 else e).append(d)
        e0 = min(e) if False else 0
        out.append((tuple(sorted(e)), tuple(sorted(o)), qs))
    return out


def search_components(lg, nf=None, starts=8, seed=0, known=None, verbose=False, max_unknowns=300):
    """scan Koszul-type gradings (all generator charge multisets) with general (non-Koszul) entries and return the
    new components [(MF, action, generator charges)]"""
    acts = list(known or [])
    out = []
    for e, o, qs in koszul_gradings(lg, nf):
        res = mf_search(lg, e, o, starts=starts, seed=seed, known=acts, verbose=verbose, max_unknowns=max_unknowns)
        for X, A in res:
            acts.append(A)
            out.append((X, A, qs))
    return out


# ------------------------------------------------------------------------------------------------ phase twists
def twist(X, left=None, right=None, name=None):
    """compose a defect with phase symmetries on either side: Q(g x, h x') (g, h preserve W).  The result is again an
    MF of W(x) - W(x') with the same grading; its bulk action is the product of X's action with the phase actions."""
    lg = X.lg
    N = lg.N
    gl = [cmath.exp(2j * cmath.pi * float(Fr(t))) for t in (left or [0] * N)]
    gr = [cmath.exp(2j * cmath.pi * float(Fr(t))) for t in (right or [0] * N)]
    g = gl + gr
    Q = [[{m: v * np.prod([g[k] ** m[k] for k in range(2 * N)]) for m, v in q.items()} for q in row] for row in X.Q]
    return MF(lg, Q, X.par, X.deg, name or "%s|g%s,h%s" % (X.name, tuple(str(Fr(t)) for t in (left or [0] * N)),
                                                             tuple(str(Fr(t)) for t in (right or [0] * N))))


def twisted_closure(lg, comps, known=None, sides="right"):
    """all phase twists of the search components (right twists X(x, h x') by default; 'both' adds left twists),
    deduplicated by bulk action against `known` and each other.  Returns [(MF, action)]."""
    from .lines import phase_group
    G = phase_group(lg)
    acts = list(known or [])
    out = []
    for X in comps:
        pairs = [(None, h) for h in G] if sides == "right" else [(g, h) for g in G for h in G]
        for g, h in pairs:
            Y = twist(X, g, h)
            A = bulk_action(Y)
            if np.max(np.abs(A)) < 1e-8 or any(np.allclose(A, B, atol=1e-6) for B in acts):
                continue
            acts.append(A)
            out.append((Y, A))
    return out


def conjugate(X, name=None):
    """complex conjugate defect (W with real coefficients): Q -> Q-bar; bulk action -> conjugate"""
    Q = [[{m: np.conj(v) for m, v in q.items()} for q in row] for row in X.Q]
    return MF(X.lg, Q, X.par, X.deg, name or X.name + "*")


# ------------------------------------------------------------------------------------------------ rank-one (Verlinde) defects
def rank_one_defect(lg, Ss, name=None):
    """tensor product over the Fermat factors x_i^{n_i} of the rank-one factorisations
    P_S(x_i, x_i') = prod_{a in S} (x_i - eta^a x_i'), eta = e^{2 pi i/n_i}  (Brunner-Roggenkamp; S = {0} is the
    identity, consecutive S give the N=2 minimal-model Verlinde lines, S = {a} the phase eta^a).  As a Koszul MF with
    generators P_{S_i} and partners (x_i^n - x_i'^n)/P_{S_i}."""
    N = lg.N
    pure = {}
    for m, c in lg.W.items():
        nz = [i for i, e in enumerate(m) if e]
        assert len(nz) == 1, "rank_one_defect needs a Fermat block"
        pure[nz[0]] = (m[nz[0]], c)
    fs = []
    for i in range(N):
        n, c = pure[i]
        eta = cmath.exp(2j * cmath.pi / n)
        p = {tuple([0] * (2 * N)): 1.0}
        for a in Ss[i]:
            e1 = [0] * (2 * N); e1[i] = 1
            e2 = [0] * (2 * N); e2[N + i] = 1
            from .core import pmul as _pm
            p = _pm(p, {tuple(e1): 1.0, tuple(e2): -eta ** a})
        fs.append(p)
    return ideal_defect(lg, fs, name or "P%s" % ("x".join("{%s}" % ",".join(map(str, S)) for S in Ss)))


def rank_one_library(lg, consecutive_only=False):
    """all tensor products of rank-one defects P_S (S a non-empty proper subset of Z_n, or consecutive only)"""
    pure = {}
    for m, c in lg.W.items():
        nz = [i for i, e in enumerate(m) if e]
        if len(nz) != 1:
            return []
        pure[nz[0]] = m[nz[0]]
    per = []
    for i in range(lg.N):
        n = pure[i]
        if consecutive_only:
            subs = [tuple((a + k) % n for k in range(l)) for l in range(1, n) for a in range(n)]
        else:
            subs = [S for r in range(1, n) for S in it.combinations(range(n), r)]
        per.append(subs)
    out = []
    for Ss in it.product(*per):
        if all(len(S) == 1 for S in Ss):
            continue                                   # phases: already in library()
        out.append(("rank1", Ss, rank_one_defect(lg, Ss)))
    return out
