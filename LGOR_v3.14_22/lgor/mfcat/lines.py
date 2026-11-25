"""Line-level tools on top of mfcat: bound states, comparison with target eigenvalues, fusion on bulk actions,
and the worked example of the Z6-II (x1,x4) block A4 lines (2026-11-24)."""
import cmath
from fractions import Fraction as Fr
import numpy as np
from .core import LG, MF, perm_defect, phase_defect, ideal_defect, parse_ideal, direct_sum, cone, identity_defect
from .hom import bulk_action, generic_morphism, is_simple, hom_dims, eigen_report


def det_phase(lg, phases):
    """det(g) for x_i -> e^{2 pi i phases_i} x_i; the NS-normalised symmetry operator is det(g) * bulk_action"""
    return cmath.exp(2j * cmath.pi * float(sum(Fr(p) for p in phases)))


def compare_to_target(A, T, tol=1e-7):
    return float(np.max(np.abs(A - T))) < tol


def fusion_on_actions(A, terms, tol=1e-7):
    """check A @ A == sum_k n_k * B_k  (terms: list of (n_k, B_k)); this is the fusion rule on the bulk action"""
    rhs = sum(n * B for n, B in terms)
    return float(np.max(np.abs(A @ A - rhs))) < tol


def bound_state(sources, targets, c=None, seed=0, name=None):
    """Cone(phi: (+)sources -> (+)targets) for a generic even closed phi of charge c (default D, i.e. degree 0).
    K-class [targets] - [sources]; returns (MF, dim of the morphism space used, (check ok, message))."""
    A = direct_sum(*sources)
    B = direct_sum(*targets)
    phi, dim = generic_morphism(A, B, c, 0, seed)
    X = cone(A, B, phi, name)
    return X, dim, X.check()


def _swap(lg, a):
    """swap defect of the (6,3) block: ideal (x1^2 - a x4', x4 - a^-1 x1'^2), a^3 = 1"""
    import sympy as sp
    fs = parse_ideal(lg, ["x1^2 - A*x4p", "x4 - B*x1p^2"], {"A": sp.nsimplify(a) if a == 1 else a, "B": 1 / a})
    return ideal_defect(lg, fs, "S_a")


def a4_demo(zeta_index=0, simple_check=True, verbose=True):
    """Z6-II block W1 = x1^6 + x4^3: A4 line V_zeta = Cone(J^3 + S_a -> J^2 + J^4), zeta = a^-1 = w^{-zeta_index}.
    Returns a dict of checks."""
    lg = LG.fermat([6, 3])
    lg.names = ["x1", "x4"]
    w = cmath.exp(2j * cmath.pi / 3)
    a = w ** zeta_index
    J = {m: phase_defect(lg, [Fr(m, 6), Fr(m, 3)], "J^%d" % m) for m in (2, 3, 4)}
    S = _swap(lg, a)
    out = {"S_check": S.check()[0]}
    acts = {m: bulk_action(J[m]) for m in J}
    aS = bulk_action(S)
    V, dim, chk = bound_state([J[3], S], [J[2], J[4]], name="V")
    out.update(morphism_space_dim=dim, V_check=chk[0])
    aV = bulk_action(V)
    out["V_action_additive"] = bool(np.allclose(aV, acts[2] + acts[4] - acts[3] - aS, atol=1e-7))
    # target: 3 on k=0, 0 on k=1,2,4,5, 3/-1 on x1(x4 + zeta x1^2) / x1(x4 - zeta x1^2)
    jac = lg.jac()
    zeta = 1 / a
    T = np.zeros((jac.mu, jac.mu), complex)
    bi = jac.bindex
    for m in [(0, 0), (4, 1)]:
        T[bi[m], bi[m]] = 3
    P = np.array([[1, 1], [zeta, -zeta]])           # columns: origin, triple in coords (x1 x4, x1^3)
    Bk = P @ np.diag([3, -1]) @ np.linalg.inv(P)
    I2 = [bi[(1, 1)], bi[(3, 0)]]
    for i in range(2):
        for j in range(2):
            T[I2[i], I2[j]] = Bk[i, j]
    out["V_equals_LGOR_A4_line"] = compare_to_target(aV, T)
    one = np.eye(jac.mu)
    out["fusion_V2_on_actions"] = fusion_on_actions(aV, [(1, one), (1, acts[2]), (1, acts[4]), (2, aV)])
    out["V_absorbs_1prime"] = bool(np.allclose(aV @ acts[2], aV, atol=1e-7))
    out["vacuum_eigenvalue"] = complex(aV[bi[(0, 0)], bi[(0, 0)]])
    if simple_check:
        out["V_simple"] = is_simple(V)
        out["S_simple"] = is_simple(S)
    if verbose:
        for k, v in out.items():
            print("  %-26s %s" % (k, v))
    return out, V, S


# ------------------------------------------------------------------------------------------------ library + realiser
def phase_group(lg):
    """all diagonal symmetries x_i -> e^{2 pi i k_i/n_i} of a Fermat-type W (exponent of x_i in its pure power);
    general W: the subgroup of prod Z_{2D/w_i} preserving every monomial"""
    import itertools as it
    orders = [int(round(2 * lg.D / w)) for w in lg.w]
    out = []
    for ks in it.product(*[range(n) for n in orders]):
        ph = [Fr(k, n) for k, n in zip(ks, orders)]
        if all(sum(Fr(a) * p for a, p in zip(m, ph)) % 1 == 0 for m in lg.W):
            out.append(ph)
    return out


def power_swaps(lg):
    """pairs (i, j, a, b, p) with x_i^{a p}, x_j^{b p} pure powers in W, p >= 2: the cubic-type identity
    x_i^{ap} + x_j^{bp} = U^p + V^p with U = x_i^a, V = x_j^b of equal charge"""
    pure = {}
    for m, c in lg.W.items():
        nz = [i for i, e in enumerate(m) if e]
        if len(nz) == 1 and abs(c - 1) < 1e-12:
            pure[nz[0]] = m[nz[0]]
    out = []
    for i in pure:
        for j in pure:
            if i >= j:
                continue
            for p in range(2, min(pure[i], pure[j]) + 1):
                if pure[i] % p == 0 and pure[j] % p == 0:
                    a, b = pure[i] // p, pure[j] // p
                    if (a, b) != (1, 1):
                        out.append((i, j, a, b, p))
    return out


def swap_defect(lg, i, j, a, b, p, c=1.0, phases=None):
    """ideal (x_i^a - c x_j'^b, x_j^b - c^-1 x_i'^a, x_k - g_k x_k' for k != i,j), c^p = 1: the involution
    U <-> V of U^p + V^p (composed with a phase g on the spectator fields)"""
    N = lg.N
    g = [cmath.exp(2j * cmath.pi * float(Fr(t))) for t in (phases or [0] * N)]

    def mono(k, e, prime):
        t = [0] * (2 * N)
        t[k + (N if prime else 0)] = e
        return tuple(t)
    fs = [{mono(i, a, False): 1.0, mono(j, b, True): -c}, {mono(j, b, False): 1.0, mono(i, a, True): -1 / c}]
    for k in range(N):
        if k not in (i, j):
            fs.append({mono(k, 1, False): 1.0, mono(k, 1, True): -g[k]})
    return ideal_defect(lg, fs, "swap(x%d^%d<->x%d^%d, c=%s)" % (i + 1, a, j + 1, b, _cs(c)))


def _cs(c):
    c = complex(c)
    ang = cmath.phase(c) / (2 * cmath.pi)
    return str(Fr(ang).limit_denominator(60) % 1) if abs(abs(c) - 1) < 1e-9 else "%.3g" % abs(c)


def library(lg, max_swaps=None):
    """standard defect library of a block: all diagonal phase defects + all power-swap defects composed with the
    phase group (deduplicated by bulk action)"""
    lib = []
    for ph in phase_group(lg):
        lib.append(("phase", ph, phase_defect(lg, ph, "g" + str(tuple(str(x) for x in ph)))))
    for (i, j, a, b, p) in power_swaps(lg):
        for r in range(p):
            c = cmath.exp(2j * cmath.pi * r / p)
            lib.append(("swap", (i, j, a, b, p, r), swap_defect(lg, i, j, a, b, p, c)))
    pure = {}
    for m, cf in lg.W.items():
        nz = [i for i, e in enumerate(m) if e]
        if len(nz) == 1:
            pure[nz[0]] = m[nz[0]]
    for i in pure:
        for j in pure:
            if i < j and pure[i] == pure[j]:
                perm = list(range(lg.N))
                perm[i], perm[j] = j, i
                for ph in phase_group(lg):
                    lib.append(("perm", (tuple(perm), tuple(ph)),
                                perm_defect(lg, perm, ph, name="perm(x%d<->x%d)g%s" % (i + 1, j + 1,
                                                                                    tuple(str(x) for x in ph)))))
    acts, keep = [], []
    for kind, data, X in lib:
        A = bulk_action(X)
        if any(np.allclose(A, B, atol=1e-8) for B in acts):
            continue
        acts.append(A)
        keep.append((kind, data, X, A))
    return keep


def _key(v):
    w = np.concatenate([v.real, v.imag])
    return tuple(np.round(w * 1e5).astype(np.int64) + 0)


def realise_fast(target, lib, max_terms=3, max_coef=3):
    """hash-based version of realise for large libraries: all solutions with <= 3 terms (integer coefficients
    |n| <= max_coef); for 4 terms meet-in-the-middle over pairs.  Same output format as realise."""
    import itertools as it
    V = [x[3].reshape(-1) for x in lib]
    T = target.reshape(-1)
    n = len(V)
    coefs = [c for c in range(-max_coef, max_coef + 1) if c]
    table = {}
    for k in range(n):
        for c in coefs:
            table.setdefault(_key(c * V[k]), []).append((c, k))
    sols = []
    for (c, k) in table.get(_key(T), []):
        sols.append([(c, k)])
    if sols or max_terms < 2:
        return sols
    for k in range(n):
        for c in coefs:
            R = T - c * V[k]
            for (c2, k2) in table.get(_key(R), []):
                if k2 > k:
                    sols.append([(c, k), (c2, k2)])
    if sols or max_terms < 3:
        return sols
    for k, l in it.combinations(range(n), 2):
        for c in coefs:
            for d in coefs:
                R = T - c * V[k] - d * V[l]
                for (c3, k3) in table.get(_key(R), []):
                    if k3 > l:
                        sols.append([(c, k), (d, l), (c3, k3)])
        if len(sols) > 50:
            break
    if sols or max_terms < 4:
        return sols
    pairs = {}
    for k, l in it.combinations(range(n), 2):
        for c in coefs:
            for d in coefs:
                pairs.setdefault(_key(c * V[k] + d * V[l]), []).append(((c, k), (d, l)))
    for k, l in it.combinations(range(n), 2):
        for c in coefs:
            for d in coefs:
                for (a, b) in pairs.get(_key(T - c * V[k] - d * V[l]), []):
                    if a[1] > l:
                        sols.append([(c, k), (d, l), a, b])
                        if len(sols) > 20:
                            return sols
    return sols


def realise(target, lib, max_terms=5, max_coef=3, tol=1e-7):
    """integer combinations sum n_X [X] of library defects whose bulk action equals target (smallest first).
    Returns list of [(n, index)] solutions (at most a few)."""
    import itertools as it
    n = len(lib)
    T = target.reshape(-1)
    sols = []
    for size in range(1, max_terms + 1):
        for sub in it.combinations(range(n), size):
            M = np.array([lib[k][3].reshape(-1) for k in sub]).T
            x, *_ = np.linalg.lstsq(M, T, rcond=None)
            if np.linalg.norm(M @ x - T) > tol * max(1, np.linalg.norm(T)):
                continue
            xr = np.round(x.real)
            if np.max(np.abs(x - xr)) > 1e-6 or np.any(xr == 0) or np.max(np.abs(xr)) > max_coef:
                continue
            sols.append([(int(c), k) for c, k in zip(xr, sub)])
        if sols:
            return sols
    return sols


def realise_bound_state(sol, lib, seed=0, name="L"):
    """build the cone (+)_{n<0} X^{|n|} -> (+)_{n>0} X^n of a realisation; returns (MF, report dict)"""
    src, tgt = [], []
    for c, k in sol:
        (tgt if c > 0 else src).extend([lib[k][2]] * abs(c))
    rep = {"terms": ["%+d %s" % (c, lib[k][2].name) for c, k in sol]}
    if not tgt:                                   # all coefficients negative: K-class -[X] = [X[1]]
        from .core import shift as _shift
        tgt, src = [_shift(X) for X in src], []
    if not src:
        X = direct_sum(*tgt)
        if len(tgt) == 1 or is_simple(X):
            rep.update(kind="single defect" if len(tgt) == 1 else "direct sum", check=True, simple=is_simple(X))
            return X, rep
        Y, info = odd_bound_state(tgt, seed=seed, name=name)
        if Y is not None:
            rep.update(kind="iterated odd cone", check=Y.check()[0], simple=is_simple(Y), steps=info,
                       vacuum_eigenvalue=complex(bulk_action(Y)[0, 0]))
            return Y, rep
        rep.update(kind="direct sum (no odd morphisms found, not simple)", check=True, simple=False)
        return X, rep
    X, dim, chk = bound_state(src, tgt, seed=seed, name=name)
    rep.update(kind="cone", morphism_space_dim=dim, check=chk[0])
    if dim == 0:
        rep["note"] = "no degree-0 morphisms: the cone is a direct sum (not a bound state)"
    rep["simple"] = is_simple(X)
    rep["vacuum_eigenvalue"] = complex(bulk_action(X)[0, 0])
    return X, rep


def gtlines_targets(ns, nmax=5):
    """bulk-action matrices (Jacobian monomial basis of LG.fermat(ns)) of the certified gtlines lines of a
    Fermat block, one per (candidate group, solution, irrep of dim > 1)"""
    import sympy as sp
    from ..gtlines.pipeline import analyse_block
    lg = LG.fermat(list(ns))
    jac = lg.jac()
    bi = jac.bindex
    rep, _ = analyse_block(tuple(ns), nmax=nmax, a_picture=False)
    out = []
    for c in rep["candidates"]:
        if not c.get("solB") or c.get("status") == "duplicate":
            continue
        irr = c["SG"].irreps()
        for si, s in enumerate(c["solB"]):
            for ri, (d, ch) in enumerate(irr):
                if d == 1:
                    continue
                T = np.zeros((jac.mu, jac.mu), complex)
                for st, lab in c["labels"].items():
                    if not isinstance(lab, tuple):
                        T[bi[tuple(st)], bi[tuple(st)]] = complex(ch[lab])
                for key, sts in c["tsec"].items():
                    M = np.array(sp.Matrix(s["mats"][key]).evalf().tolist(), dtype=complex)   # rows = eigenvectors
                    Dg = np.diag([complex(ch[o]) for o in c["tcls"][key]])
                    A = M.T @ Dg @ np.linalg.inv(M.T)
                    idx = [bi[tuple(t)] for t in sts]
                    for a in range(len(idx)):
                        for b in range(len(idx)):
                            T[idx[a], idx[b]] = A[a, b]
                out.append({"group": c["group"], "solution": si, "irrep": ri, "dim": d, "target": T})
    return lg, out


def extended_library(lg, starts=8, seed=1, fourier=True, sides="right", verbose=False, rank1=True):
    """library(lg) + Fourier (rotation) defects + components found by the general MF search (Koszul-type gradings,
    arbitrary entries), closed under complex conjugation and phase twists X(x, h x'); deduplicated by bulk action"""
    from .search import fourier_library, search_components, twisted_closure, conjugate
    lib = library(lg)
    acts = [x[3] for x in lib]
    if fourier:
        for kind, data, X in fourier_library(lg):
            A = bulk_action(X)
            if not any(np.allclose(A, B, atol=1e-8) for B in acts):
                acts.append(A)
                lib.append((kind, data, X, A))
    if rank1:
        from .search import rank_one_library
        for kind, data, X in rank_one_library(lg, consecutive_only=(rank1 == "consecutive")):
            A = bulk_action(X)
            if np.max(np.abs(A)) > 1e-8 and not any(np.allclose(A, B, atol=1e-8) for B in acts):
                acts.append(A)
                lib.append((kind, data, X, A))
    comps = search_components(lg, starts=starts, seed=seed, known=acts, verbose=verbose)
    Xs = [X for X, A, qs in comps]
    Xs += [conjugate(X) for X in Xs]
    for Y, A in twisted_closure(lg, Xs, known=acts, sides=sides):
        lib.append(("mf", None, Y, A))
    return lib


def realise_block(ns, nmax=5, max_terms=5, verbose=True, search=False, starts=8, seed=1, lib=None):
    """try to realise every gtlines line of a Fermat block as a (bound state of) library defects
    (search=True: use extended_library, i.e. also Fourier defects and general MF-search components)"""
    lg, targets = gtlines_targets(ns, nmax)
    if lib is None:
        lib = extended_library(lg, starts=starts, seed=seed, verbose=verbose) if search else library(lg)
    lines = []
    if verbose:
        print("block %s: %d library defects (%d phases, %d swaps, %d Fourier, %d MF-search), %d gtlines lines" % (
            tuple(ns), len(lib), sum(1 for x in lib if x[0] == "phase"), sum(1 for x in lib if x[0] == "swap"),
            sum(1 for x in lib if x[0] == "fourier"), sum(1 for x in lib if x[0] == "mf"), len(targets)))
        print("   (+ %d rank-one/Verlinde tensor defects)" % sum(1 for x in lib if x[0] == "rank1"))
    for t in targets:
        sols = (realise_fast(t["target"], lib, max_terms=min(max_terms, 4)) if len(lib) > 40
                else realise(t["target"], lib, max_terms=max_terms))
        sols.sort(key=lambda so: (len(so), all(c > 0 for c, _ in so), sum(abs(c) for c, _ in so)))
        entry = dict(t)
        entry.pop("target")
        if not sols:
            entry["realised"] = False
        else:
            X, rep = realise_bound_state(sols[0], lib)
            entry.update(realised=True, n_solutions=len(sols), **rep)
        lines.append(entry)
        if verbose:
            print("  %-22s sol %d irrep %d (d=%d): %s" % (t["group"], t["solution"], t["irrep"], t["dim"],
                  ("; ".join(entry["terms"]) + " | %s, simple=%s" % (entry["kind"], entry.get("simple")))
                  if entry["realised"] else "NOT realised by <= %d library terms" % max_terms))
    return lines


def odd_cone(A, B, psi, name=None):
    """cone of an odd closed morphism psi: A -> B (charge D): Q = [[Q_A, 0], [psi, Q_B]], K-class [A] + [B]"""
    n = A.n + B.n
    Q = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(A.n):
        for j in range(A.n):
            Q[i][j] = A.Q[i][j]
    for i in range(B.n):
        for j in range(B.n):
            Q[A.n + i][A.n + j] = B.Q[i][j]
        for j in range(A.n):
            Q[A.n + i][j] = psi[i][j]
    return MF(A.lg, Q, A.par + B.par, A.deg + B.deg, name or "Cone[odd](%s -> %s)" % (A.name, B.name))


def odd_bound_state(parts, seed=0, name=None):
    """glue defects with positive K-coefficients by odd morphisms: X_1 = parts[0], X_{k+1} = Cone(parts[k] -> X_k)
    using a generic odd closed morphism of charge D after a grading shift of parts[k] (searched in [-2D, top+2D]).
    Returns (MF or None, list of (shift, Hom dim))."""
    from .core import shift as _shift
    X = parts[0]
    info = []
    lg = X.lg
    top = lg.jac().top
    for P in parts[1:]:
        found = None
        for c in range(-2 * lg.D, top + 2 * lg.D + 1):
            Ps = _shift(P, flip=False, s=lg.D - c)          # charge-c morphisms of P become charge D
            psi, dim = generic_morphism(Ps, X, lg.D, 1, seed)
            if dim:
                found = (odd_cone(Ps, X, psi, name), c, dim)
                break
        if found is None:
            return None, info
        X, c, dim = found
        info.append((c, dim))
    return X, info
