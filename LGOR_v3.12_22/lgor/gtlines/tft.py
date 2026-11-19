"""State labels, TFT certification (B picture: Saito flat coordinates; A picture: FJRW), Verlinde test."""
import cmath
import itertools
import math
from collections import Counter
from fractions import Fraction as Fr

from .fixedpoints import orbit_counts

B2 = lambda x: x * x - x + Fr(1, 6)


def block_states(ns):
    return [s for s in itertools.product(*[range(n - 1) for n in ns])]


def charge(s, ns):
    return sum(Fr(a, n) for a, n in zip(s, ns))


def sector(s, ns, K):
    return int(charge(s, ns) * K) % K


def label_states(ns, K, tab, SG):
    """Returns (labels, tsec, tcls) or (None, reason, None).
    labels[s] = class index, or ('T', key) for a state of an unknown-basis sector key = (k, K q);
    tsec[key] = chiral states of sector k with charge q, tcls[key] = the G-classes its basis vectors carry.
    Sectors are split by R-charge because a line preserving the block N=2 algebra cannot mix charges; for
    c_hat = 1 every twisted sector has a single charge, so this is the old rule [for c_hat > 1: inferred]."""
    A, labels, tsec, tcls = SG.A, {}, {}, {}
    own = orbit_counts([sum(1 for i, j in enumerate(SG.sig[l % SG.m]) if i == j) for l in range(K)], K)
    for s in block_states(ns):
        k = sector(s, ns, K)
        if k == 0:
            labels[s] = SG.cls(A.els[0], 0)
            continue
        if SG.sig[k % SG.m] == SG.e and (tab[k][0] != A.n or orbit_counts(tab[k], K) != own):
            return None, "sector %d: sigma^k = 1 but its fixed set is not A as a Z_K-set" % k, None
        cl = []
        for a in A.els:
            c = SG.cls(a, k)
            if c not in cl:
                cl.append(c)
        if len(cl) == 1:
            labels[s] = cl[0]
            continue
        key = (k, int(charge(s, ns) * K))
        labels[s] = ('T', key)
        tsec.setdefault(key, []).append(s)
        tcls[key] = cl
    for key, sts in tsec.items():
        if len(sts) != len(tcls[key]):
            return None, "sector %d (charge %s): %d chiral states vs %d classes" % (
                key[0], Fr(key[1], K), len(sts), len(tcls[key])), None
    return labels, tsec, tcls


def _msets(ch, n, target, start=0):
    """multisets of indices into ch (sorted ascending, non-negative ints) of size n with sum target"""
    if n == 0:
        if target == 0:
            yield ()
        return
    hi = ch[-1]
    for i in range(start, len(ch)):
        c = ch[i]
        if c * n > target:
            break
        if target - c > (n - 1) * hi:
            continue
        for rest in _msets(ch, n - 1, target - c, i):
            yield (i,) + rest


def _keystr(key):
    return "%d_%d" % key if isinstance(key, tuple) else str(key)


def certify(ns, SG, labels, tsec, tcls, nmax=5, corr_fn=None, max_tuples=3_000_000):
    """Impose all class-forbidden genus-0 couplings (n = 3..nmax) on the unknown orbit bases.
    corr_fn(ns, states) -> Fraction (default: B-model product_corr on monomials; mirrorlines-style FJRW for the A
    picture).  Only tuples with total charge c_hat + n - 3 are enumerated (the dimension axiom on both pictures).
    Returns dict with conditions, solutions and statistics."""
    import sympy as sp
    from ..frobenius import product_corr
    if corr_fn is None and nmax > 5 and len(ns) > 1:
        # 3.12.1: n >= 6 correlators of a product block by WDVV reconstruction (lgor/gtlines/wdvv.py)
        from .wdvv import get_tft
        corr_fn = get_tft(ns, nmax).corr_fn()
    K = math.lcm(*ns)
    chat_units = int(sum(1 - Fr(2, n) for n in ns) * K)
    C = {}
    for key, sts in tsec.items():
        C[key] = sp.Matrix(len(sts), len(sts), lambda i, j: sp.Symbol("c%s_%d_%d" % (_keystr(key), i, j)))
    pool = [('T', key, i) for key in sorted(tsec) for i in range(len(tsec[key]))]
    pool += [s for s in block_states(ns) if not isinstance(labels[s], tuple)]

    def units(entry):
        if isinstance(entry[0], str):
            return entry[1][1]
        return int(charge(entry, ns) * K)

    pool.sort(key=units)
    pch = [units(x) for x in pool]

    def cl(entry):
        return tcls[entry[1]][entry[2]] if isinstance(entry[0], str) else labels[entry]

    def expand(entry):
        if isinstance(entry[0], str):
            key, i = entry[1], entry[2]
            return [(tsec[key][j], C[key][i, j]) for j in range(len(tsec[key]))]
        return [(entry, 1)]

    cache = {}

    def corr(sts):
        key = tuple(sorted(sts))
        if key not in cache:
            try:
                cache[key] = (corr_fn or product_corr)(list(ns), list(key))
            except NotImplementedError:
                cache[key] = None
        return cache[key]

    conds, refuted, skipped, nforb, ncharge, truncated = set(), [], 0, Counter(), Counter(), []
    nontriv, lineonly = Counter(), Counter()
    conds_n = {}
    for n in range(3, nmax + 1):
        for cnt, idx in enumerate(_msets(pch, n, chat_units + (n - 3) * K)):
            if cnt >= max_tuples:
                truncated.append(n)
                break
            ncharge[n] += 1
            tup = [pool[i] for i in idx]
            if SG.allowed([cl(x) for x in tup]):
                continue
            nforb[n] += 1
            abel_ok = SG.allowed_abelian([cl(x) for x in tup])
            tot = 0
            for combo in itertools.product(*[expand(x) for x in tup]):
                v = corr([st for st, _ in combo])
                if v is None:
                    skipped += 1
                    continue
                if v:
                    cf = sp.Rational(v.numerator, v.denominator)
                    for _, c in combo:
                        cf *= c
                    tot += cf
            tot = sp.expand(tot)
            if tot != 0:
                nontriv[n] += 1
                if abel_ok:
                    lineonly[n] += 1
                if not tot.free_symbols:
                    refuted.append(tuple(tup))
                else:
                    conds.add(tot)
                    conds_n.setdefault(n, set()).add(tot)
    res = {"forbidden": dict(nforb), "charge_allowed": dict(ncharge), "nontrivial_forbidden": dict(nontriv),
           "line_only": dict(lineonly),
           "conditions": len(conds), "refuted_by": refuted[:5], "skipped_terms": skipped, "truncated": truncated,
           "solutions": [], "C": C, "conds": conds_n}
    if refuted:
        return res
    # charts: each orbit row gets a pivot column with coefficient 1 (rows <-> orbits; pivots a permutation)
    sols = []
    ks = sorted(tsec)
    for perms in itertools.product(*[list(itertools.permutations(range(len(tsec[k])))) for k in ks]):
        sub = {}
        for k, p in zip(ks, perms):
            for i, j in enumerate(p):
                sub[C[k][i, j]] = 1
        unknowns = sorted({s for k in ks for s in C[k].free_symbols} - set(sub), key=str)
        allq = [e for e in (sp.expand(c.subs(sub)) for c in conds) if e != 0]
        if any(not e.free_symbols for e in allq):
            continue
        family = False
        if len(unknowns) <= 4:
            found = None
            for ncut in sorted(conds_n):
                eqs = [e for e in (sp.expand(c.subs(sub)) for m in conds_n if m <= ncut for c in conds_n[m]) if e != 0]
                if any(not e.free_symbols for e in eqs):
                    found = []
                    break
                f0 = sp.solve(eqs, unknowns, dict=True) if eqs else [{}]
                if all(len(f) == len(unknowns) for f in f0):
                    found = f0
                    break
            if found is None:
                found = sp.solve(allq, unknowns, dict=True) if allq else [{}]
            found = [f for f in found if len(f) < len(unknowns) or
                     all(abs(complex(sp.N(e.subs(f)))) < 1e-9 for e in allq)]
        else:
            # 3.12.2: staged numeric solve.  Solve the conditions up to the smallest n for which the solution set
            # is finite (no positive-dimensional component seen), then keep only those solutions that satisfy all
            # conditions (after Newton polishing on the full system).  A direct random-start solve of the full,
            # larger system missed solutions at n = 7, 8 for the (4,4) block.
            found, family = None, False
            for ncut in sorted(conds_n):
                eqs = [e for e in (sp.expand(c.subs(sub)) for m in conds_n if m <= ncut for c in conds_n[m]) if e != 0]
                if not eqs or any(not e.free_symbols for e in eqs):
                    continue
                cand, fam = _numeric_solve(eqs, unknowns, starts=max(80, 12 * len(unknowns)))
                if fam and ncut != max(conds_n):
                    continue
                family = fam
                found = _filter_full(cand, allq, unknowns) if ncut != max(conds_n) else cand
                break
            if found is None:
                found, family = _numeric_solve(allq, unknowns, starts=max(80, 12 * len(unknowns)))
            res["numeric"] = True
            ex = []
            for f in found:
                g = {u: _exactify(v) for u, v in f.items()}
                if all(sp.simplify(e.subs(g)) == 0 for e in allq):
                    ex.append(g)
                else:
                    ex.append(f)
            found = ex
            if family:
                sols.append({"family": True, "free": "positive-dimensional component (numerical rank test)", "mats": {}})
        for f in found:
            full = dict(sub)
            full.update(f)
            mats = {k: C[k].subs(full) for k in ks}
            if any(m.free_symbols for m in mats.values()):
                free = set().union(*[m.free_symbols for m in mats.values()])
                sols.append({"family": True, "free": free, "mats": mats})
                continue
            if any(abs(complex(sp.N(m.det()))) < 1e-9 for m in mats.values()):
                continue
            mats = {k: _normalise(m) for k, m in mats.items()}
            if not any(_same(mats, s["mats"]) for s in sols if not s.get("family")):
                sols.append({"family": False, "mats": mats})
    res["solutions"] = sols
    return res


# ----------------------------------------------------------------- A picture: FJRW invariants of narrow sectors
def _frac(x):
    return x - math.floor(x)


def chiodo4(r, ths):
    """genus-0 4-point r-spin invariant with narrow twists ths (one coordinate, degree -2 bundle): -ch_1(R pi_* L)."""
    q = Fr(1, r)
    s = B2(q) - sum(B2(t) for t in ths)
    for (i, j) in [(0, 1), (0, 2), (0, 3)]:
        s += B2(_frac(q - ths[i] - ths[j]))
    return -s / 2


class _RSpinAdapter:
    """exposes .r and .corr(exponent labels) for 4 labels via Chiodo, so frobenius._deg1_class can split with it"""
    def __init__(self, r):
        self.r = r

    def corr(self, lab):
        if any(a < 0 or a > self.r - 2 for a in lab):
            return Fr(0)
        if len(lab) != 4:
            raise ValueError
        ths = [Fr(a + 1, self.r) for a in lab]
        if Fr(2, self.r) - sum(ths) != -2:
            return Fr(0)
        return chiodo4(self.r, ths)


LIT_COUNT = {"n5_single": 0}


def fjrw_corr(ns, states):
    """FJRW genus-0 invariant of narrow block states given by Krawitz labels a (theta_i = (a_i+1)/n_i).
    n = 3: concavity; n = 4: Chiodo GRR [verified = Saito]; n = 5 two coordinates of degree 1: splitting of
    Chiodo classes; n = 5 one coordinate of degree 2: r-spin = A_{r-1} theorem [literature, counted in LIT_COUNT]."""
    import sympy as sp
    from .. import frobenius as fb
    n = len(states)
    degs = []
    for f, r in enumerate(ns):
        ths = [Fr(s[f] + 1, r) for s in states]
        d = Fr(n - 2, r) - sum(ths)
        if d.denominator != 1 or d > -1:
            return Fr(0)
        degs.append(int(-d - 1))
    if sum(degs) != n - 3:
        return Fr(0)
    big = [f for f in range(len(ns)) if degs[f] > 0]
    if n == 3:
        return Fr(1)
    if n == 4:
        f = big[0]
        return chiodo4(ns[f], [Fr(s[f] + 1, ns[f]) for s in states])
    if n == 5 and len(big) == 2:
        cols = [tuple(s[f] for s in states) for f in big]
        cf = fb._deg1_class(_RSpinAdapter(ns[big[0]]), cols[0])
        cg = fb._deg1_class(_RSpinAdapter(ns[big[1]]), cols[1])
        val = (cf.T * fb._G * cg)[0]
        return Fr(int(sp.fraction(val)[0]), int(sp.fraction(val)[1]))
    if n == 5 and len(big) == 1:
        LIT_COUNT["n5_single"] += 1
        f = big[0]
        return fb.frob(ns[f]).corr(tuple(s[f] for s in states))
    raise NotImplementedError


def _numeric_solve(eqs, unknowns, starts=40, seed=0):
    """Damped complex Gauss-Newton from random starts.  Returns (isolated solutions, family_found)."""
    import numpy as np
    import sympy as sp
    n = len(unknowns)
    F = sp.lambdify(unknowns, eqs, "numpy")
    J = sp.lambdify(unknowns, [[sp.diff(e, u) for u in unknowns] for e in eqs], "numpy")
    f_ = lambda x: np.array(F(*x), dtype=complex).ravel()
    j_ = lambda x: np.array(J(*x), dtype=complex).reshape(len(eqs), n)
    rng = np.random.default_rng(seed)
    sols, family = [], False
    for _ in range(starts):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        for _it in range(300):
            f = f_(x)
            nf = np.linalg.norm(f)
            if nf < 1e-14:
                break
            dx = np.linalg.lstsq(j_(x), -f, rcond=None)[0]
            lam = 1.0
            while lam > 1e-6:
                xn = x + lam * dx
                if np.linalg.norm(f_(xn)) < nf:
                    break
                lam /= 2
            x = xn
            if np.max(np.abs(x)) > 1e8:
                break
        if np.linalg.norm(f_(x)) > 1e-10 or np.max(np.abs(x)) > 1e6:
            continue
        if np.linalg.matrix_rank(j_(x), tol=1e-7) < n:
            family = True
            continue
        if not any(np.max(np.abs(x - y)) < 1e-7 for y in sols):
            sols.append(x)
    return [dict(zip(unknowns, [complex(v) for v in x])) for x in sols], family


def _filter_full(cand, eqs, unknowns, tol=1e-8):
    """Newton-polish candidate solutions on the full system and keep those that solve it."""
    import numpy as np
    import sympy as sp
    if not cand:
        return []
    n = len(unknowns)
    F = sp.lambdify(unknowns, eqs, "numpy")
    J = sp.lambdify(unknowns, [[sp.diff(e, u) for u in unknowns] for e in eqs], "numpy")
    f_ = lambda x: np.array(F(*x), dtype=complex).ravel()
    j_ = lambda x: np.array(J(*x), dtype=complex).reshape(len(eqs), n)
    out = []
    for c in cand:
        x = np.array([complex(c[u]) for u in unknowns])
        for _ in range(20):
            f = f_(x)
            if np.linalg.norm(f) < 1e-14:
                break
            x = x + np.linalg.lstsq(j_(x), -f, rcond=None)[0]
        scale = max(1.0, float(np.max(np.abs(x))))
        if np.linalg.norm(f_(x)) < tol * scale ** 4:
            out.append(dict(zip(unknowns, [complex(v) for v in x])))
    return out


def _exactify(val):
    import sympy as sp
    re, im = round(val.real, 12), round(val.imag, 12)
    g = sp.nsimplify(complex(re, im), [sp.sqrt(3), sp.sqrt(2), sp.Rational(1, 3)], tolerance=1e-9)
    return g


def _normalise(M):
    """scale each row so that its first non-zero entry is 1"""
    import sympy as sp
    M = sp.Matrix(M)
    for i in range(M.rows):
        piv = next(M[i, j] for j in range(M.cols) if abs(complex(sp.N(M[i, j]))) > 1e-9)
        M[i, :] = (M[i, :] / piv).applyfunc(lambda x: sp.nsimplify(x) if not x.has(sp.Float) else sp.N(x))
    return M


def _same(m1, m2):
    import sympy as sp
    if set(m1) != set(m2):
        return False
    return all(abs(complex(sp.N(a)) - complex(sp.N(b))) < 1e-7 for k in m1 for a, b in zip(m1[k], m2[k]))


# ----------------------------------------------------------------- 6. Verlinde comparison
def _verlinde_eig(ns, r, s):
    v = 1.0
    for n, ri, ai in zip(ns, r, s):
        v *= math.sin(math.pi * (ri + 1) * (ai + 1) / n) / math.sin(math.pi * (ai + 1) / n)
    return v


def line_on_monomials(ns, SG, labels, tsec, tcls, mats, rho):
    """Matrix of the line rho on each sector in the monomial basis.  Returns (diagonal?, {state: eig})."""
    import sympy as sp
    eig, diag = {}, True
    for s, lab in labels.items():
        if not isinstance(lab, tuple):
            eig[s] = rho[lab]
    for k, sts in tsec.items():
        M = sp.Matrix(mats[k]).evalf()
        D = sp.diag(*[complex(rho[o]) for o in tcls[k]])
        Op = M.inv() * D * M          # acts on monomial coordinates (rows of M = orbit vectors)
        for i in range(len(sts)):
            for j in range(len(sts)):
                if i != j and abs(complex(Op[i, j])) > 1e-9:
                    diag = False
            eig[sts[i]] = complex(Op[i, i])
    return diag, eig


def verlinde_match(ns, eig):
    for r in itertools.product(*[range(n - 1) for n in ns]):
        if all(abs(_verlinde_eig(ns, r, s) - complex(v)) < 1e-9 for s, v in eig.items()):
            return r
    return None


# ----------------------------------------------------------------- 7. action of phase symmetries
def act_on_solution(ns, tsec, mats, phases):
    """phases[f]: Fraction, field f -> e^{2 pi i phases[f]} x_f.  Returns transformed (normalised) mats."""
    import sympy as sp
    out = {}
    for k, sts in tsec.items():
        M = sp.Matrix(mats[k])
        for j, s in enumerate(sts):
            ph = sum(Fr(a) * p for a, p in zip(s, phases)) % 1
            M[:, j] = M[:, j] * sp.exp(2 * sp.pi * sp.I * sp.Rational(ph.numerator, ph.denominator))
        out[k] = _normalise(M.applyfunc(lambda x: sp.expand_complex(x)))
    return out




def _basis_str_safe(sts, M):
    import sympy as sp
    rows = []
    for i in range(M.rows):
        terms = []
        for j, s in enumerate(sts):
            c = M[i, j]
            c = sp.nsimplify(c) if not getattr(c, "has", lambda *_: False)(sp.Float) else sp.N(c, 6)
            if c != 0:
                mono = "*".join("y%d^%d" % (f + 1, a) if a > 1 else "y%d" % (f + 1) for f, a in enumerate(s) if a) or "1"
                terms.append(mono if c == 1 else "(%s)%s" % (c, mono))
        rows.append(" + ".join(terms))
    return "{" + ", ".join(rows) + "}"
