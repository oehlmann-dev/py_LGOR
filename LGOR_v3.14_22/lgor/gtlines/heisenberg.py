"""Heisenberg orbit lines (LGOR 3.14): lines from the non-abelian group of fixed-point labels and translations.

gtlines builds G = A x| <sigma> with A abelian (fixed points of theta^k*, sigma = theta acting on them) and takes
Rep(G).  On the theta^k-twisted states with sigma^k = 1 the fixed points carry two kinds of operators:

  * labels       Z_v : f(a) -> chi_v(a) f(a)        (characters of A; the quantum symmetry of the twisted sector)
  * translations X_u : f(a) -> f(a - u)             (shifts by the lattice vectors that relate fixed points)

with Z_v X_u = chi_v(u) X_u Z_v: together they generate a Heisenberg group H (a central extension of A x A^).
theta acts on H by conjugation (X_u -> X_{sigma u}, Z_v -> Z_{sigma* v}).  Every theta-orbit O of Weyl operators
W(u, v) ~ X_u Z_v gives an operator L_O = sum_{w in O} W(w) commuting with theta, hence a candidate line of
quantum dimension |O| acting on the theta-invariant fixed-point functions = the chiral states of the sector.

  * orbits of pairwise commuting operators: diagonalisable together; these are the Rep(G)-type lines that
    gtlines already finds (each certified orbit basis is the eigenbasis of one of them);
  * orbits of non-commuting operators: NOT of Rep(G) type for any abelian A; their eigenvalues are typically
    complex/irrational (Z6-II theta^3: +-sqrt3 or +-i sqrt3).  These are the new candidate lines.

Lifting to the chiral ring needs the normalisation of the orbit vectors relative to the indicator functions of
the sigma-orbits, which the diagonal (Rep(G)) certification leaves free.  It is fixed here by requiring every
commuting orbit line to coincide with a certified line [verified where it is used: all commuting lines match].
Action outside the fixed-point sectors: d on class-e states, 0 on rotation classes (as for the d > 1 irreps of
G whose characters vanish there) [inferred].  The phase of W(u,v) (Hermitian W^2 = 1 or plain X_u Z_v) is not
fixed by the construction; both variants are produced and the fusion closure decides.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from fractions import Fraction as Fr

import numpy as np


# ------------------------------------------------------------------------------------------------ fixed-point algebra
def _shift(A, u):
    P = np.zeros((A.n, A.n))
    for a in A.els:
        P[A.idx[A.add(a, u)], A.idx[a]] = 1
    return P


def _char(A, v):
    return np.diag([np.exp(2j * np.pi * float(sum(Fr(x * y, m) for x, y, m in zip(v, a, A.mods)))) for a in A.els])


def _theta(A, sigma):
    T = np.zeros((A.n, A.n))
    for i in range(A.n):
        T[sigma[i], i] = 1
    return T


def weyl(A, u, v, variant="hermitian"):
    """X_u Z_v, normalised so that W^2 = 1 ('hermitian') or left as it is ('plain')"""
    M = _shift(A, u) @ _char(A, v)
    if variant == "plain":
        return M
    M2 = M @ M
    s = np.trace(M2) / A.n
    if np.allclose(M2, s * np.eye(A.n), atol=1e-9) and abs(s) > 1e-9:
        return M / np.sqrt(s)
    return M


def orbit_operators(A, sigma, variant="hermitian"):
    """theta-orbits of the non-trivial Weyl operators.  Returns list of dicts
    {members: [(u, v)], ops: [matrix], L: sum, commuting: bool, size}"""
    T = _theta(A, sigma)
    Ti = T.T
    pairs = [(u, v) for u in A.els for v in A.els if not (all(x == 0 for x in u) and all(x == 0 for x in v))]
    ops = {p: weyl(A, *p, variant=variant) for p in pairs}

    def ident(M):
        for p, W in ops.items():
            for ph in (1, -1, 1j, -1j):
                if np.allclose(M, ph * W, atol=1e-9):
                    return p
        return None
    seen, out = set(), []
    for p in pairs:
        if p in seen:
            continue
        mem, Ms = [p], [ops[p]]
        M = ops[p]
        while True:
            M = T @ M @ Ti
            if np.allclose(M, Ms[0], atol=1e-9):
                break
            Ms.append(M)
            q = ident(M)
            if q is not None:
                mem.append(q)
        seen |= set(mem)
        comm = all(np.allclose(a @ b, b @ a, atol=1e-9) for a in Ms for b in Ms)
        out.append({"members": mem, "ops": Ms, "L": sum(Ms), "commuting": comm, "size": len(Ms)})
    return out


def invariant_matrix(A, sigma, L, reps):
    """matrix (columns = images) of L on the theta-invariant functions, basis = indicator functions of the
    sigma-orbits of the representatives `reps` (element tuples), in that order"""
    orbit_of = {}
    for a in A.els:
        orb, x = [], A.idx[a]
        while x not in orb:
            orb.append(x)
            x = sigma[x]
        orbit_of[a] = frozenset(orb)
    E = np.zeros((A.n, len(reps)))
    for j, r in enumerate(reps):
        for x in orbit_of[r]:
            E[x, j] = 1
    R = np.linalg.pinv(E) @ L @ E
    assert np.allclose(L @ E, E @ R, atol=1e-8), "operator does not preserve the invariant subspace"
    return R


# ------------------------------------------------------------------------------------------------ lift to the chiral ring
def _sector_data(cand, key):
    SG, tsec, tcls = cand["SG"], cand["tsec"], cand["tcls"]
    k = key[0]
    reps = []
    for c in tcls[key]:
        a = next(a for a in SG.A.els if SG.cls(a, k) == c)
        reps.append(a)
    return reps


def _orbit_basis_matrix(Op, M):
    """closure convention Op = M^-1 R^T M  ->  R (columns = images in the orbit basis of M)"""
    return (M @ Op @ np.linalg.inv(M)).T


def fit_scales(cand, certified, by_q, ns, orbs):
    """orbit-vector scales of the reference solution in every fixed-point sector, fixed by matching the commuting
    orbit lines to certified lines.  Returns (lam, ok, messages); lam[key] = (reps, t, M, idx, q)."""
    from .closure import _charge
    SG = cand["SG"]
    fixkeys = [key for key in cand["tsec"] if SG.sig[key[0] % SG.m] == SG.e]
    report = []
    sol0 = cand["solB"][0]
    lam = {}
    ok_all = True
    for key in fixkeys:
        reps = _sector_data(cand, key)
        ks = cand["tsec"][key]
        q = _charge(ks[0], ns)
        M = np.array([[complex(x) for x in row] for row in
                      (sol0["mats"][key].tolist() if hasattr(sol0["mats"][key], "tolist") else sol0["mats"][key])])
        sts = by_q[q]
        idx = [sts.index(s) for s in ks]
        targets = [_orbit_basis_matrix(L["ops"][q][1][np.ix_(idx, idx)], M) for L in certified]
        # fit the orbit-vector scales t_j (t_0 = 1) from the commuting orbit lines
        n = len(reps)
        t = [1.0] + [None] * (n - 1)
        for o in orbs:
            if not o["commuting"]:
                continue
            R = invariant_matrix(SG.A, SG.sigma, o["L"], reps)
            for sgn in (1, -1):
                Rs = sgn * R
                for T in targets:
                    if not np.allclose(np.diag(Rs), np.diag(T), atol=1e-7):
                        continue
                    trial = list(t)
                    good = True
                    for i in range(n):
                        for j in range(n):
                            if i == j or abs(Rs[i, j]) < 1e-9:
                                if i != j and abs(T[i, j]) > 1e-7:
                                    good = False
                                continue
                            r = T[i, j] / Rs[i, j]           # = t_j / t_i
                            if trial[i] is not None and trial[j] is None:
                                trial[j] = r * trial[i]
                            elif trial[j] is not None and trial[i] is None:
                                trial[i] = trial[j] / r
                            elif trial[i] is not None and abs(trial[j] / trial[i] - r) > 1e-6:
                                good = False
                    if good:
                        t = trial
                        break
        if any(x is None for x in t):
            ok_all = False
            report.append("   sector %s: orbit-vector scales not fixed by the commuting lines" % (key,))
            continue
        lam[key] = (reps, np.array(t, complex), M, idx, q)
    return lam, ok_all, report


def lift_operator(cand, by_q, lam, Rfun, d, name, untwisted=None):
    """closure entry for a fixed-point operator: Rfun(reps) = matrix on the invariant functions (columns = images,
    indicator basis); d on class-e labelled states (or `untwisted`), 0 on all other labelled states"""
    SG = cand["SG"]
    e_cls = SG.cindex[SG.e]
    ops = {}
    for qq, sts in by_q.items():
        A_ = np.zeros((len(sts), len(sts)), complex)
        for a, s in enumerate(sts):
            lab = cand["labels"].get(s)
            if lab is not None and not isinstance(lab, tuple) and lab == e_cls:
                A_[a, a] = d if untwisted is None else untwisted(s)
        ops[qq] = [sts, A_]
    for key, (reps, t, M, idx, q) in lam.items():
        Rt = np.diag(1 / t) @ Rfun(reps) @ np.diag(t)
        ops[q][1][np.ix_(idx, idx)] = np.linalg.inv(M) @ Rt.T @ M
    return {"name": name, "d": d, "group": SG.name(), "ops": {qq: (v[0], v[1]) for qq, v in ops.items()}}


def heisenberg_lines(rep, ns, variants=("hermitian", "plain"), verbose=False):
    """New (non-commuting) Heisenberg orbit lines of every certified candidate of a block, as closure entries
    {name, d, ops: {charge: (states, matrix)}} in the convention of closure.block_operators.  Also returns a
    report (list of strings)."""
    from .closure import block_operators, _charge
    from .tft import block_states
    lines, inv = block_operators(rep, ns)
    allst = block_states(ns)
    by_q = defaultdict(list)
    for s in allst:
        by_q[_charge(s, ns)].append(s)
    out, report = [], []
    for ci, cand in enumerate(rep["candidates"]):
        if not str(cand.get("status", "")).startswith("certified") or not cand.get("solB"):
            continue
        SG = cand["SG"]
        fixkeys = [key for key in cand["tsec"] if SG.sig[key[0] % SG.m] == SG.e]
        if not fixkeys:
            continue
        certified = [L for L in lines if L.get("group") == SG.name() and not L.get("dressed")]
        for variant in variants:
            orbs = orbit_operators(SG.A, SG.sigma, variant)
            ncomm = [o for o in orbs if not o["commuting"]]
            report.append("%s (%s Weyl operators): %d theta-orbits, %d commuting, %d non-commuting (sizes %s)"
                          % (SG.name(), variant, len(orbs), len(orbs) - len(ncomm), len(ncomm),
                             sorted(o["size"] for o in ncomm)))
            if not ncomm:
                continue
            lam, ok_all, msgs = fit_scales(cand, certified, by_q, ns, orbs)
            report += msgs
            if not ok_all:
                continue
            for oi, o in enumerate(ncomm):
                for sgn in (1, -1, 1j, -1j):
                    ops = {}
                    for qq, sts in by_q.items():
                        A_ = np.zeros((len(sts), len(sts)), complex)
                        for a, s in enumerate(sts):
                            lab = cand["labels"].get(s)
                            if lab is not None and not isinstance(lab, tuple):
                                A_[a, a] = o["size"] if SG.classes[lab] == [SG.e] or lab == SG.cindex[SG.e] else 0
                        ops[qq] = [sts, A_]
                    for key, (reps, t, M, idx, q) in lam.items():
                        R = sgn * invariant_matrix(SG.A, SG.sigma, o["L"], reps)
                        Rt = np.diag(1 / t) @ R @ np.diag(t)
                        Op = np.linalg.inv(M) @ Rt.T @ M
                        ops[q][1][np.ix_(idx, idx)] = Op
                    ent = {"name": "Heis[%s] %s orbit%d %s" % (SG.name().split("= ")[-1], variant, oi,
                                                              {1: "+", -1: "-", 1j: "+i", -1j: "-i"}[sgn]),
                           "d": o["size"], "group": SG.name(), "heis": True,
                           "ops": {qq: (v[0], v[1]) for qq, v in ops.items()}}
                    out.append(ent)
    # dedup
    uniq = []
    for e in out:
        if not any(all(np.allclose(e["ops"][q][1], f["ops"][q][1], atol=1e-7) for q in e["ops"]) for f in uniq):
            uniq.append(e)
    return uniq, report


# ------------------------------------------------------------------------------------------------ completed closure
def _phase_group(ns):
    out = []
    for ks in itertools.product(*[range(n) for n in ns]):
        out.append(tuple(Fr(k, n) for k, n in zip(ks, ns)))
    return out


def full_invertibles(ns, by_q):
    """all phase symmetries x_i -> e^{2 pi i k_i/n_i} x_i and permutations of equal exponents, on the NS chiral
    ring (closure convention, charge blocks)"""
    inv = []
    perms = [p for p in itertools.permutations(range(len(ns))) if all(ns[i] == ns[p[i]] for i in range(len(ns)))]
    for p in perms:
        for g in _phase_group(ns):
            ops = {}
            for q, sts in by_q.items():
                idx = {s: i for i, s in enumerate(sts)}
                A = np.zeros((len(sts), len(sts)), complex)
                for s in sts:
                    t = tuple(s[p[i]] for i in range(len(ns)))
                    A[idx[t], idx[s]] = np.exp(2j * np.pi * float(sum(Fr(a) * x for a, x in zip(s, g))))
                ops[q] = (sts, A)
            inv.append({"name": "g%s%s" % (tuple(str(x) for x in g), "" if p == tuple(range(len(ns))) else " perm%s" % (p,)),
                        "d": 1, "ops": ops})
    return inv


def verlinde_lines(ns, by_q):
    """tensor products of su(2)_{n-2} Verlinde lines (N=2 minimal-model lines with m = s = 0) of the factors"""
    out = []
    for ls in itertools.product(*[range(n - 1) for n in ns]):
        if all(l == 0 for l in ls):
            continue
        d = 1.0
        for l, n in zip(ls, ns):
            d *= np.sin(np.pi * (l + 1) / n) / np.sin(np.pi / n)
        ops = {}
        for q, sts in by_q.items():
            A = np.diag([np.prod([np.sin(np.pi * (l + 1) * (a + 1) / n) / np.sin(np.pi * (a + 1) / n)
                                  for l, a, n in zip(ls, s, ns)]) for s in sts]).astype(complex)
            ops[q] = (sts, A)
        out.append({"name": "Verlinde%s" % (ls,), "d": d, "ops": ops, "group": "Verlinde"})
    return out


def decompose_milp(P, D, items, tol=1e-6):
    """non-negative integer combination sum n_i items_i = P with sum n_i d_i = D (scipy MILP).  Returns list of
    names, 'outside-span', or None (in the span, no non-negative integer solution)."""
    keys = sorted(P)
    A = np.array([np.concatenate([it["ops"][q][1].ravel() for q in keys]) for it in items]).T
    b = np.concatenate([P[q][1].ravel() for q in keys])
    Ar = np.vstack([A.real, A.imag]); br = np.concatenate([b.real, b.imag])
    x, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    if np.linalg.norm(Ar @ x - br) > 1e-6 * max(1.0, np.linalg.norm(br)):
        return "outside-span"
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None
    keep = np.abs(Ar).sum(axis=1) > 1e-12
    Ar2, br2 = Ar[keep], br[keep]
    d = np.array([float(it["d"]) for it in items])
    cons = [LinearConstraint(Ar2, br2 - tol, br2 + tol), LinearConstraint(d[None, :], D - tol, D + tol)]
    res = milp(c=np.ones(len(items)), constraints=cons, integrality=np.ones(len(items)),
               bounds=Bounds(0, np.floor(D / d.min() + 1e-9)))
    if not res.success:
        return None
    n = np.round(res.x).astype(int)
    if np.linalg.norm(Ar @ n - br) > 1e-5:
        return None
    out = []
    for i, c in enumerate(n):
        out += [items[i]["name"]] * int(c)
    return out


def completed_closure(ns, nmax=5, heis=True, budget=400000, verbose=False):
    """Fusion closure of the certified gtlines lines with the full invertible symmetry group, the Verlinde lines of
    the factors and (heis=True) the Heisenberg orbit lines.  Products tested: L x (g L' g^-1) for certified
    non-invertible L, L' of the same candidate group and every invertible g.  Returns (text, open products)."""
    from .pipeline import analyse_block
    from .closure import block_operators, _charge, _prod, decompose
    from .tft import block_states
    rep, _ = analyse_block(tuple(ns), nmax=nmax, a_picture=False)
    lines, inv0 = block_operators(rep, ns)
    by_q = defaultdict(list)
    for s in block_states(ns):
        by_q[_charge(s, ns)].append(s)
    inv = full_invertibles(ns, by_q)
    base = [L for L in lines if not L.get("dressed")]
    known = list(base) + verlinde_lines(ns, by_q)
    hl, hrep = heisenberg_lines(rep, ns) if heis else ([], [])
    known += hl

    def conj(L, g):
        return {"name": "%s^%s" % (L["name"], g["name"]), "d": L["d"], "group": L.get("group"),
                "ops": {q: (L["ops"][q][0], g["ops"][q][1] @ L["ops"][q][1] @ np.linalg.inv(g["ops"][q][1]))
                        for q in L["ops"]}}
    # close the known non-invertible lines under conjugation by invertibles, then dress
    allk = []
    for L in known:
        for g in inv:
            C = conj(L, g)
            if not any(all(np.allclose(C["ops"][q][1], X["ops"][q][1], atol=1e-7) for q in C["ops"]) for X in allk):
                allk.append(C)
    dressed = []
    for L in allk:
        for g in inv:
            E = {"name": "%s*%s" % (g["name"], L["name"]), "d": L["d"], "group": L.get("group"), "heis": L["name"].startswith("Heis"),
                 "ops": {q: (L["ops"][q][0], g["ops"][q][1] @ L["ops"][q][1]) for q in L["ops"]}}
            if not any(all(np.allclose(E["ops"][q][1], X["ops"][q][1], atol=1e-7) for q in E["ops"]) for X in dressed):
                dressed.append(E)
    txt = ["COMPLETED FUSION CLOSURE of block %s: %d certified non-invertible lines, %d invertibles (full phase x perm "
           "group), %d Verlinde, %d Heisenberg candidates; %d dressed/conjugated known lines"
           % (tuple(ns), len(base), len(inv), len(known) - len(base) - len(hl), len(hl), len(dressed))]
    txt += ["  " + r for r in hrep]
    opened = []
    for a, b in itertools.combinations_with_replacement(range(len(base)), 2):
        if base[a].get("group") != base[b].get("group"):
            continue
        for g in inv:
            Lb = conj(base[b], g)
            P = _prod(base[a], Lb)
            D = base[a]["d"] * base[b]["d"]
            dec = decompose_milp(P, D, [x for x in dressed if not x.get("heis")] + inv)
            if isinstance(dec, list):
                continue
            dech = decompose_milp(P, D, dressed + inv) if heis else dec
            key = (base[a]["name"], base[b]["name"])
            if isinstance(dech, list):
                opened.append((key, g["name"], "closed by Heisenberg lines", dech))
            else:
                opened.append((key, g["name"], "OPEN (%s)" % dech, None))
    seen = set()
    for (key, gname, status, dec) in opened:
        tag = (key, status.split()[0])
        txt.append("  %s x %s^%s: %s%s" % (key[0], key[1], gname, status, (" = " + " + ".join(dec)) if dec else ""))
    n_open = sum(1 for o in opened if o[2].startswith("OPEN"))
    n_heis = sum(1 for o in opened if o[2].startswith("closed"))
    txt.append("  RESULT: %d products need lines beyond invertibles + Verlinde + certified; %d of them decompose once the "
               "Heisenberg lines are added; %d remain open" % (len(opened), n_heis, n_open))
    return "\n".join(txt), opened


# ------------------------------------------------------------------------------------------------ Clifford / Fourier
def clifford_generators(A):
    """Fourier transform on A (F[a,b] = chi_b(a)/sqrt|A|, using the standard self-pairing of A) and the quadratic
    phase gates diag(exp(2 pi i a_i^2 / (2 m_i))) (even m_i) or diag(exp(2 pi i a_i^2/m_i)) (odd m_i), one per
    cyclic factor; these normalise the Heisenberg group (Weil representation of the modular group on fixed points)"""
    n = A.n
    F = np.array([[np.exp(2j * np.pi * float(sum(Fr(x * y, m) for x, y, m in zip(a, b, A.mods)))) for b in A.els]
                  for a in A.els]) / np.sqrt(n)
    gens = [("F", F)]
    for i, m in enumerate(A.mods):
        den = 2 * m if m % 2 == 0 else m
        P = np.diag([np.exp(2j * np.pi * a[i] ** 2 / den) for a in A.els])
        gens.append(("P%d" % i, P))
    return gens


def clifford_elements(A, sigma, max_elems=3000):
    """elements of the group generated by the Clifford generators and theta, modulo phases (BFS)"""
    T = _theta(A, sigma)
    gens = clifford_generators(A) + [("T", T)]

    def key(M):
        k = np.argmax(np.abs(M.ravel()) > 1e-9)
        ph = M.ravel()[k] / abs(M.ravel()[k])
        Mn = M / ph
        return tuple(np.round(Mn.ravel() * 1e6).astype(np.int64).tolist()) + tuple(np.round(Mn.ravel().imag * 1e6).astype(np.int64).tolist())
    start = np.eye(A.n, dtype=complex)
    seen = {key(start): ("1", start)}
    frontier = [("1", start)]
    while frontier and len(seen) < max_elems:
        nxt = []
        for nm, M in frontier:
            for gn, G in gens:
                X = G @ M
                k = key(X)
                if k not in seen:
                    seen[k] = (gn + "." + nm, X)
                    nxt.append((gn + "." + nm, X))
        frontier = nxt
    return list(seen.values())


def clifford_lines(rep, ns, max_orbit=6, max_elems=3000):
    """theta-orbit sums of Clifford elements (modulo phase), lifted like the Heisenberg lines; only orbits whose
    members are not Weyl operators (those are the Heisenberg lines) and not theta powers.  Each orbit sum is
    produced with the overall phases 1, -1, i, -i.  Returns (entries, report)."""
    from .closure import block_operators, _charge
    from .tft import block_states
    lines, inv = block_operators(rep, ns)
    by_q = defaultdict(list)
    for s in block_states(ns):
        by_q[_charge(s, ns)].append(s)
    out, report = [], []
    for cand in rep["candidates"]:
        if not str(cand.get("status", "")).startswith("certified") or not cand.get("solB"):
            continue
        SG = cand["SG"]
        if not any(SG.sig[key[0] % SG.m] == SG.e for key in cand["tsec"]):
            continue
        certified = [L for L in lines if L.get("group") == SG.name() and not L.get("dressed")]
        orbs = orbit_operators(SG.A, SG.sigma, "hermitian")
        lam, ok, msgs = fit_scales(cand, certified, by_q, ns, orbs)
        if not ok:
            report += msgs
            continue
        T = _theta(SG.A, SG.sigma)
        weyls = [weyl(SG.A, u, v) for u in SG.A.els for v in SG.A.els]
        els = clifford_elements(SG.A, SG.sigma, max_elems)
        Rs = []
        for nm, C in els:
            if any(np.allclose(np.abs(C), np.abs(W), atol=1e-9) and abs(abs(np.trace(C.conj().T @ W)) - SG.A.n) < 1e-6
                   for W in weyls):
                continue                                        # Weyl operator (up to phase): Heisenberg line
            orb = [C]
            M = C
            while len(orb) <= max_orbit:
                M = T @ M @ T.T
                if np.allclose(M, orb[0], atol=1e-9):
                    break
                orb.append(M)
            if len(orb) > max_orbit:
                continue
            L = sum(orb)
            try:
                R0 = invariant_matrix(SG.A, SG.sigma, L, [r for r in lam[next(iter(lam))][0]])
            except AssertionError:
                continue
            if any(np.allclose(R0, X, atol=1e-7) for X, _ in Rs):
                continue
            Rs.append((R0, (nm, L, len(orb))))
        report.append("%s: %d Clifford elements (mod phase), %d distinct theta-orbit sums of size <= %d on the "
                      "invariant functions" % (SG.name(), len(els), len(Rs), max_orbit))
        for i, (R0, (nm, L, d)) in enumerate(Rs):
            for ph in (1, -1, 1j, -1j):
                out.append(lift_operator(cand, by_q, lam, lambda reps, L=L, ph=ph: ph * invariant_matrix(SG.A, SG.sigma, L, reps),
                                         d, "Cliff[%s] %s %s" % (SG.name().split("= ")[-1], nm, {1: "+", -1: "-", 1j: "+i", -1j: "-i"}[ph])))
    return out, report


def test_candidates(ns, cands, nmax=5, tol=1e-6, verbose=False, full=True):
    """consistency test of candidate lines against the closed set (certified + full invertibles + Verlinde +
    Heisenberg): a candidate L passes if L x L-bar contains the identity and L x X decomposes (non-negative integers)
    for every known X, allowing L, L-bar and their dressings/conjugates as new summands.
    Returns list of (name, d, passed, reason)."""
    from .pipeline import analyse_block
    from .closure import block_operators, _charge, _prod
    from .tft import block_states
    rep, _ = analyse_block(tuple(ns), nmax=nmax, a_picture=False)
    lines, _ = block_operators(rep, ns)
    by_q = defaultdict(list)
    for s in block_states(ns):
        by_q[_charge(s, ns)].append(s)
    inv = full_invertibles(ns, by_q)
    hl, _ = heisenberg_lines(rep, ns)
    base = [L for L in lines if not L.get("dressed")] + verlinde_lines(ns, by_q) + hl

    def same(a, b):
        return all(np.allclose(a["ops"][q][1], b["ops"][q][1], atol=1e-7) for q in a["ops"])

    def orbit(Ls):
        conj = []
        for L in Ls:
            for h in inv:
                E = {"name": "%s^%s" % (L["name"], h["name"]), "d": L["d"],
                     "ops": {q: (L["ops"][q][0], h["ops"][q][1] @ L["ops"][q][1] @ np.linalg.inv(h["ops"][q][1]))
                             for q in L["ops"]}}
                if not any(same(E, X) for X in conj):
                    conj.append(E)
        out = []
        for L in conj:
            for g in inv:
                E = {"name": "%s*%s" % (g["name"], L["name"]), "d": L["d"],
                     "ops": {q: (L["ops"][q][0], g["ops"][q][1] @ L["ops"][q][1]) for q in L["ops"]}}
                if not any(same(E, X) for X in out):
                    out.append(E)
        return out
    known = orbit(base)
    res = []
    for C in cands:
        if any(same(C, X) for X in known):
            res.append((C["name"], C["d"], None, "already known"))
            continue
        Cb = {"name": C["name"] + "-bar", "d": C["d"], "ops": {q: (C["ops"][q][0], C["ops"][q][1].conj()) for q in C["ops"]}}
        items = known + orbit([C, Cb]) + inv
        P = _prod(C, Cb)
        dec = decompose_milp(P, C["d"] ** 2, items)
        if not isinstance(dec, list) or not any(n.startswith("g('0'") and n.count("'0'") == len(ns) for n in dec):
            res.append((C["name"], C["d"], False, "L x Lbar: %s" % (dec if not isinstance(dec, list) else "no identity")))
            continue
        bad = None
        for X in (base if full else []):
            dec = decompose_milp(_prod(C, X), C["d"] * X["d"], items)
            if not isinstance(dec, list):
                bad = (X["name"], dec)
                break
        res.append((C["name"], C["d"], bad is None, "ok" if bad is None else "L x %s: %s" % bad))
        if verbose:
            print(res[-1], flush=True)
    return res
