"""Bulk action and graded morphism spaces of defects (LGOR 3.14 mfcat)."""
import numpy as np
from .core import pmul, padd, mdiff, mmul, identity_defect, MF


# ------------------------------------------------------------------------------------------------ bulk action
def _str(M, par):
    acc = {}
    for i in range(len(par)):
        acc = padd(acc, M[i][i], -1 if par[i] else 1)
    return acc


def raw_bulk_action(mf):
    lg = mf.lg
    N = lg.N
    J = lg.jac()
    M = None
    for i in list(range(N, 2 * N)) + list(range(N)):
        dQ = mdiff(mf.Q, i)
        M = dQ if M is None else mmul(M, dQ)
    S = _str(M, mf.par)
    # split S = sum_{m'} c_{m'}(x) x'^{m'}
    byp = {}
    for k, v in S.items():
        byp.setdefault(k[N:], {})
        byp[k[N:]][k[:N]] = byp[k[N:]].get(k[:N], 0) + v
    A = np.zeros((J.mu, J.mu), complex)
    rescache = {}
    for col, b in enumerate(J.basis):
        tot = {}
        for mp, cx in byp.items():
            m = tuple(a + c for a, c in zip(b, mp))
            if lg.charge(m) != J.top:
                continue
            if m not in rescache:
                rescache[m] = J.res({m: 1.0})
            r = rescache[m]
            if abs(r) > 1e-14:
                tot = padd(tot, {k: r * v for k, v in cx.items()})
        for mm, v in J.nf(tot).items():
            A[J.bindex[mm], col] += v
    return A


_ID_NORM = {}


def bulk_action(mf):
    """matrix of the defect on RR ground states in the Jacobian basis lg.jac().basis (columns = images),
    normalised by the identity defect"""
    lg = mf.lg
    key = id(lg)
    if key not in _ID_NORM:
        A0 = raw_bulk_action(identity_defect(lg))
        s = A0[0, 0]
        assert np.allclose(A0, s * np.eye(len(A0)), atol=1e-8), "identity defect does not act as a scalar"
        _ID_NORM[key] = s
    return raw_bulk_action(mf) / _ID_NORM[key]


def eigen_report(A, basis, names):
    """eigenvalues and eigenvectors (as polynomials) of a bulk action, per charge block"""
    w, v = np.linalg.eig(A)
    out = []
    for k in range(len(w)):
        vec = v[:, k] / v[np.argmax(abs(v[:, k])), k]
        poly = " + ".join("(%s)%s" % (_c(c), _mono(m, names)) for c, m in zip(vec, basis) if abs(c) > 1e-8)
        out.append((w[k], poly))
    return out


def _c(c):
    c = complex(c)
    if abs(c.imag) < 1e-9:
        return "%.4g" % c.real
    return "%.4g%+.4gi" % (c.real, c.imag)


def _mono(m, names):
    s = "*".join("%s^%d" % (n, a) if a > 1 else n for n, a in zip(names, m) if a)
    return s or "1"


# ------------------------------------------------------------------------------------------------ Hom spaces
def _space(A, B, c, par):
    lg = A.lg
    unk = []
    for i in range(B.n):
        for j in range(A.n):
            if (B.par[i] - A.par[j]) % 2 != par:
                continue
            for m in lg.monos(c + B.deg[i] - A.deg[j]):
                unk.append((i, j, m))
    return unk


def dmatrix(A, B, c, par):
    """d(phi) = Q_B phi - (-1)^par phi Q_A : Hom^par_c -> Hom^{par+1}_{c+D}"""
    D = A.lg.D
    src = _space(A, B, c, par)
    tgt = _space(A, B, c + D, 1 - par)
    ti = {t: k for k, t in enumerate(tgt)}
    M = np.zeros((len(tgt), len(src)), complex)
    sgn = -((-1) ** par)
    for col, (i, j, m) in enumerate(src):
        for k in range(B.n):
            q = B.Q[k][i]
            for mm, v in q.items():
                M[ti[(k, j, tuple(a + b for a, b in zip(mm, m)))], col] += v
        for k in range(A.n):
            q = A.Q[j][k]
            for mm, v in q.items():
                M[ti[(i, k, tuple(a + b for a, b in zip(mm, m)))], col] += sgn * v
    return M, src


def _rank(M, tol=1e-8):
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())


def hom_dim(A, B, c, par):
    D = A.lg.D
    Dc, src = dmatrix(A, B, c, par)
    Dp, _ = dmatrix(A, B, c - D, 1 - par)
    return len(src) - _rank(Dc) - _rank(Dp)


def hom_dims(A, B, cmin=None, cmax=None):
    """{(charge, parity): dim} of H(Hom(A,B)); default charge window [-2D, top + 2D] (cohomology is finite)"""
    lg = A.lg
    top = lg.jac().top
    cmin = -2 * lg.D if cmin is None else cmin
    cmax = top + 2 * lg.D if cmax is None else cmax
    out = {}
    for c in range(cmin, cmax + 1):
        for p in (0, 1):
            d = hom_dim(A, B, c, p)
            if d:
                out[(c, p)] = d
    return out


def closed_reps(A, B, c, par):
    """orthonormal basis of a complement of exact in closed morphisms (numerical), and the unknown index list"""
    D = A.lg.D
    Dc, src = dmatrix(A, B, c, par)
    Dp, _ = dmatrix(A, B, c - D, 1 - par)
    if Dc.size:
        _, s, vh = np.linalg.svd(Dc)
        r = int((s > 1e-8 * max(1.0, s[0] if len(s) else 1)).sum())
        ker = vh[r:].conj().T
    else:
        ker = np.eye(len(src))
    if Dp.size and _rank(Dp):
        ue, _, _ = np.linalg.svd(Dp)
        E = ue[:, :_rank(Dp)]
        ker = ker - E @ (E.conj().T @ ker)
    u2, s2, _ = np.linalg.svd(ker) if ker.size else (np.zeros((len(src), 0)), [], None)
    r2 = int((np.asarray(s2) > 1e-8).sum())
    return u2[:, :r2], src


def vec_to_morphism(v, src, A, B):
    M = [[{} for _ in range(A.n)] for _ in range(B.n)]
    for coef, (i, j, m) in zip(v, src):
        if abs(coef) > 1e-12:
            M[i][j] = padd(M[i][j], {m: coef})
    return M


def generic_morphism(A, B, c=None, par=0, seed=0):
    """random combination of the cohomology classes in Hom^par_c(A,B) (default c = D: degree 0 for cones)"""
    c = A.lg.D if c is None else c
    reps, src = closed_reps(A, B, c, par)
    rng = np.random.default_rng(seed)
    coef = rng.normal(size=reps.shape[1]) + 1j * rng.normal(size=reps.shape[1])
    return vec_to_morphism(reps @ coef, src, A, B), reps.shape[1]


def is_simple(X):
    """graded-simple test: degree-0 even endomorphisms one-dimensional (End^0_0 = C id)"""
    return hom_dim(X, X, 0, 0) == 1
