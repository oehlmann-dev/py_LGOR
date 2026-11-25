"""Rank-one matrix-factorisation (B-type) defects of Fermat LG orbifolds and their survival under
deformations of W (``lgor mfdefects``).

[literature: Brunner-Roggenkamp 2007 (B-type defects in LG models); Carqueville-Runkel 2012]
For W = sum_f x_f^{n_f}, W(x) - W(y) = sum_f prod_{a in Z_n} (x_f - eta_f^a y_f), eta_f = exp(2 pi i/n_f).
A rank-one defect of factor f is  P_S = prod_{a in S}(x_f - eta^a y_f),  Q_S = (x_f^n - y_f^n)/P_S,  S a subset
of Z_{n_f}.  S = {0} is the identity defect, S = {m, m+1, .., m+l} (consecutive) are the Verlinde-type
lines L_{l, m} of the N=2 minimal model [literature]; non-consecutive S are further rank-one B-type defects
(their decomposition into simple CFT lines is not computed here [missing]).  Diagonal phase symmetries act
on x and y alike, so every P_S is equivariant under the orbifold group [verified: P_S(wx, wy) = w^|S| P_S].

First-order survival under W -> W + eps*M:  the tensor-product defect D = (x) D_f deforms iff
    M(x) - M(y)  in  I_D = sum_f (P_{S_f}, Q_{S_f})      (identity defect (x_f - y_f, Q_0) on untouched factors),
because H^0(End D) = C[x, y] / I_D for a tensor product of rank-one factorisations.  Ideal membership
is decided by a Groebner basis over GF(p) with p = 1 mod lcm(n_f) (eta in GF(p)), with two primes as a
check [verified mod p].  Control: the quintic Fibonacci line (S = {-1, 0, 1} in one factor) survives a
monomial iff its exponent in that factor is 0 or 3 (the 50-dim locus) [verified in tests].
"""
from __future__ import annotations

import itertools
from functools import reduce
from math import gcd
from typing import Dict, List, Sequence, Tuple

V, LIT, INF, MISS = "[verified]", "[literature]", "[inferred]", "[missing]"


def _lcm(a, b):
    return a * b // gcd(a, b)


def _is_prime(p):
    if p < 2:
        return False
    i = 2
    while i * i <= p:
        if p % i == 0:
            return False
        i += 1
    return True


def primes_for(n_list: Sequence[int], count: int = 2, start: int = 1000) -> List[int]:
    L = reduce(_lcm, n_list, 1)
    out, p = [], (start // L + 1) * L + 1
    while len(out) < count:
        if _is_prime(p):
            out.append(p)
        p += L
    return out


def root_of_unity(n: int, p: int) -> int:
    """A primitive n-th root of unity in GF(p) (n | p - 1)."""
    for g in range(2, p):
        e = pow(g, (p - 1) // n, p)
        if all(pow(e, n // q, p) != 1 for q in range(2, n + 1) if n % q == 0 and _is_prime(q)):
            return e
    raise ValueError("no root")


class MFSurvival:
    def __init__(self, exponents: Sequence[int], p: int):
        import sympy as sp
        self.sp = sp
        self.n = list(exponents)
        self.p = p
        self.r = len(self.n)
        self.x = sp.symbols(f"x1:{self.r + 1}")
        self.y = sp.symbols(f"y1:{self.r + 1}")
        self.eta = [root_of_unity(nf, p) for nf in self.n]

    def factor_ideal(self, f: int, S: Sequence[int]):
        sp = self.sp
        x, y, n, e = self.x[f], self.y[f], self.n[f], self.eta[f]
        P = sp.Integer(1)
        Q = sp.Integer(1)
        for a in range(n):
            lin = x - (pow(e, a, self.p)) * y
            if a % n in {s % n for s in S}:
                P *= lin
            else:
                Q *= lin
        return [sp.expand(P), sp.expand(Q)]

    def ideal(self, defect: Dict[int, Sequence[int]]):
        gens = []
        for f in range(self.r):
            gens += self.factor_ideal(f, defect.get(f, [0]))
        return gens

    def survives(self, defect: Dict[int, Sequence[int]], mono: Sequence[int], G=None) -> bool:
        sp = self.sp
        if G is None:
            G = sp.groebner(self.ideal(defect), *self.x, *self.y, modulus=self.p, order="grevlex")
        mx = sp.Integer(1)
        my = sp.Integer(1)
        for f, a in enumerate(mono):
            mx *= self.x[f] ** a
            my *= self.y[f] ** a
        return G.contains(sp.expand(mx - my))

    def groebner(self, defect):
        return self.sp.groebner(self.ideal(defect), *self.x, *self.y, modulus=self.p, order="grevlex")


def single_factor_table(exponents: Sequence[int], f: int, sets: Sequence[Sequence[int]], monos, primes=None):
    """{S: [survives(mono) for mono in monos]} for one-factor defects, checked with two primes."""
    primes = primes or primes_for(exponents)
    res = {}
    for S in sets:
        vals = None
        for p in primes:
            M = MFSurvival(exponents, p)
            G = M.groebner({f: S})
            v = [M.survives({f: S}, mono, G) for mono in monos]
            if vals is None:
                vals = v
            elif vals != v:
                raise RuntimeError(f"prime dependence for S={S}")
        res[tuple(S)] = vals
    return res


def defect_sets(n: int, full_upto: int = 7, extra_size: int = 2):
    """Subsets S of Z_n (no identification by rotation: a rotation composes the defect with the phase
    symmetry x -> eta^c x, which the deformation may break).  All proper non-empty subsets for n <= full_upto;
    otherwise all consecutive sets plus non-consecutive sets of size <= extra_size.  Returns (S, consecutive)."""
    out = []
    def consec(S):
        k = len(S)
        return any(set(S) == {(m + j) % n for j in range(k)} for m in range(n))
    if n <= full_upto:
        for size in range(1, n):
            for S in itertools.combinations(range(n), size):
                out.append((S, consec(S)))
        return out
    for size in range(1, n):
        for m in range(n):
            S = tuple(sorted({(m + j) % n for j in range(size)}))
            out.append((S, True))
    seen = {S for S, _ in out}
    for size in range(2, extra_size + 1):
        for S in itertools.combinations(range(n), size):
            if S not in seen and not consec(S):
                out.append((S, False))
    return out


def report(model, monos=None, max_exponent: int = 12, max_monos: int = 400) -> str:
    from ..tdl import fermat_exponents, marginal_monomials
    n = fermat_exponents(model)
    if monos is None:
        monos = marginal_monomials(model)
    monos = [tuple(m) for m in monos][:max_monos]
    L = ["MATRIX-FACTORISATION (B-TYPE) DEFECTS: first-order survival under the marginal deformations", "-" * 100]
    L.append(f"model {model.name}: exponents {n}; {len(monos)} invariant marginal monomials tested; Groebner over GF(p), "
             f"p in {primes_for(n)} (results agree) " + V)
    L.append("  S = roots in P_S = prod_(a in S)(x - eta^a y); |S| = 1: invertible phase line x -> eta^a x; consecutive |S| > 1:")
    L.append("  Verlinde-type L_(l,m); non-consecutive: further rank-one B-type defects (simplicity/decomposition not computed " + MISS + ");")
    L.append("  S and its complement give the same defect up to a shift, only one is listed")
    for f, nf in enumerate(n):
        if nf <= 2 or nf > max_exponent:
            if nf > max_exponent:
                L.append(f"  factor x{f + 1} (n = {nf}): skipped (n > {max_exponent}) " + MISS)
            continue
        sets, seenS = [], set()
        for S, c in defect_sets(nf):
            comp = tuple(sorted(set(range(nf)) - set(S)))
            if S == (0,) or comp == (0,) or comp in seenS:      # P_S and P_(complement) = same defect up to shift
                continue
            seenS.add(S)
            sets.append((S, c))
        tab = single_factor_table(n, f, [s for s, _ in sets], monos)
        id_pattern = tuple([True] * len(monos))
        verl = {}
        for S, c in sets:
            if c:
                verl.setdefault(tuple(tab[S]), []).append(S)
        L.append(f"  factor x{f + 1} (n = {nf}): {len(sets)} defects tested")
        for pat, Ss in sorted(verl.items(), key=lambda t: -sum(t[0])):
            exps = sorted({m[f] for m, ok in zip(monos, pat) if ok})
            kind = "invertible/Verlinde" if any(len(S) == 1 for S in Ss) and any(len(S) > 1 for S in Ss) else \
                ("invertible phase" if all(len(S) == 1 for S in Ss) else "Verlinde-type")
            L.append(f"    {kind} S in {[set(S) for S in Ss][:6]}{' ...' if len(Ss) > 6 else ''}: kept by {sum(pat)}/{len(monos)};"
                     f" kept x{f + 1}-exponents {exps}")
        def coset(S):
            m = len(S)
            if nf % m:
                return False
            step = nf // m
            return set(S) == {(S[0] + j * step) % nf for j in range(m)}
        cos = [S for S, c in sets if not c and coset(S)]
        if cos:
            L.append(f"    cosets of subgroups Z_m (P_S = x^m - eta^(cm) y^m: Z_m condensation x phase, composites of invertible lines "
                     + INF + f"): {[set(S) for S in cos][:6]}{' ...' if len(cos) > 6 else ''}")
        new = [(S, tab[S]) for S, c in sets if not c and not coset(S) and tuple(tab[S]) not in verl and tuple(tab[S]) != id_pattern]
        newfull = [(S, tab[S]) for S, c in sets if not c and not coset(S) and tuple(tab[S]) == id_pattern]
        L.append(f"    other non-consecutive defects with a survival pattern of no Verlinde-type defect (candidates): {len(new)} " + V)
        for S, v in new[:8]:
            exps = sorted({m[f] for m, ok in zip(monos, v) if ok})
            L.append(f"      S = {set(S)}: kept by {sum(v)}/{len(monos)}; kept x{f + 1}-exponents {exps}")
        if newfull:
            L.append(f"    non-consecutive defects kept by EVERY tested monomial: {[set(S) for S, _ in newfull][:8]} " + V)
    # tensor products of factor defects: maximal preserved families
    r = len(n)
    if r <= 6 and max(n) <= 8 and monos:
        prod1 = tuple([1] * r)
        mx, fam, hit = maximal_families(n, monos, must_contain=[prod1] if prod1 in monos else [])
        L.append(f"  tensor products of factor defects (any S per factor, >= 1 non-invertible factor): {len(fam)} survival patterns, "
                 f"{len(mx)} maximal families " + V)
        for k in mx[:8]:
            L.append(f"    family of {len(k)}/{len(monos)} monomials, e.g. " +
                     " x ".join(f"x{f + 1}:{set(S)}" for f, S in enumerate(fam[k]) if S != (0,)))
        if prod1 in monos:
            L.append(f"    maximal families containing x1...x{r} (the mirror's Kaehler / large-volume direction): {len(hit)} " + V)
    L.append("  A defect kept by every marginal monomial of a family survives the whole first-order family " + V + ";")
    L.append("  higher orders and Kaehler (twisted-sector) deformations are not treated " + MISS + ".")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- tensor products
def factor_status(n: int, S: Sequence[int], a: int, p: int):
    """Status of (x^a, y^a) in A_S = GF(p)[x, y] / (P_S, Q_S):
    'zero' (both vanish), ('prop', c) (x^a = c y^a, c != 0), 'indep' otherwise.  Exact over GF(p)."""
    import sympy as sp
    x, y = sp.symbols("x y")
    e = root_of_unity(n, p)
    P, Q = sp.Integer(1), sp.Integer(1)
    Sset = {s % n for s in S}
    for b in range(n):
        lin = x - pow(e, b, p) * y
        if b in Sset:
            P *= lin
        else:
            Q *= lin
    G = sp.groebner([sp.expand(P), sp.expand(Q)], x, y, modulus=p, order="grevlex")
    u = G.reduce(x ** a)[1]
    v = G.reduce(y ** a)[1]
    u = sp.Poly(u, x, y, modulus=p)
    v = sp.Poly(v, x, y, modulus=p)
    if u.is_zero and v.is_zero:
        return "zero"
    if u.is_zero or v.is_zero:
        return "indep"
    du, dv = u.as_dict(), v.as_dict()
    if set(du) != set(dv):
        return "indep"
    k0 = next(iter(du))
    c = (int(du[k0]) * pow(int(dv[k0]), p - 2, p)) % p
    if all((int(du[k]) - c * int(dv[k])) % p == 0 for k in du):
        return ("prop", c)
    return "indep"


def tensor_survives(statuses, p: int) -> bool:
    """statuses: per factor status for one monomial.  The tensor product of rank-one defects survives iff
    (some factor has both x^a, y^a zero) or (every factor proportional and prod c_f = 1)."""
    if any(s == "zero" for s in statuses):
        return True
    if any(s == "indep" for s in statuses):
        return False
    c = 1
    for s in statuses:
        c = (c * s[1]) % p
    return c == 1


def direction_report(exponents: Sequence[int], directions: Dict[str, Sequence[Sequence[int]]],
                     label: str = "", primes=None, max_factor: int = 12) -> str:
    """For each named direction (list of monomials, generic coefficients) find every tensor product of
    rank-one factor defects (any S per factor, identity allowed) that survives all its monomials to first
    order, and classify: invertible (all |S_f| = 1, i.e. a phase symmetry) or containing a non-invertible factor."""
    primes = primes or primes_for(exponents)
    n = list(exponents)
    L = [f"TENSOR-PRODUCT DEFECTS ALONG DEFORMATION DIRECTIONS {label}", "-" * 100]
    L.append("  criterion " + V + ": in A = (x)_f A_(S_f), x^M = y^M iff some factor has x^a = y^a = 0, or every factor has")
    L.append("  x^a = c_f y^a with prod c_f = 1 (rank-one tensors); computed exactly over GF(p) for two primes")
    for name, monos in directions.items():
        monos = [tuple(m) for m in monos]
        results = []
        for p in primes:
            # per factor: list of (S, status vector over monomials)
            per = []
            for f, nf in enumerate(n):
                if nf > max_factor:
                    per.append([((0,), None)])
                    continue
                sets = [(0,)] + [S for S, _ in defect_sets(nf) if S != (0,)]
                seenS, lst = set(), []
                for S in sets:
                    comp = tuple(sorted(set(range(nf)) - set(S)))
                    if comp in seenS or (comp == (0,) and S != (0,)):
                        continue
                    seenS.add(S)
                    st = [factor_status(nf, S, m[f] % nf if m[f] < nf else m[f], p) for m in monos]
                    lst.append((S, st))
                per.append(lst)
            # group the choices of each factor by their status vector
            classes = []
            for lst in per:
                d = {}
                for S, st in lst:
                    d.setdefault(None if st is None else tuple(st), []).append(S)
                classes.append(list(d.items()))
            surv = []
            for combo in itertools.product(*classes):
                ok = True
                for j in range(len(monos)):
                    sts = [c[0][j] if c[0] is not None else ("prop", 1) for c in combo]
                    if not tensor_survives(sts, p):
                        ok = False
                        break
                if ok:
                    for choice in itertools.product(*[c[1] for c in combo]):
                        surv.append(tuple(choice))
            results.append(set(surv))
        if results[0] != results[-1]:
            L.append(f"  {name}: PRIME DEPENDENCE, results differ " + MISS)
            continue
        surv = results[0]
        inv = [s for s in surv if all(len(S) == 1 for S in s)]
        non = [s for s in surv if not all(len(S) == 1 for S in s)]
        L.append(f"  {name} ({len(monos)} monomials): {len(surv)} surviving tensor defects; invertible (phases) {len(inv)}, "
                 f"with a non-invertible factor {len(non)} " + V)
        for s in non[:10]:
            L.append("      " + " x ".join(f"x{f + 1}:{set(S)}" for f, S in enumerate(s) if S != (0,)))
    return "\n".join(L) + "\n"


def maximal_families(exponents: Sequence[int], monos: Sequence[Sequence[int]], p: int = None, max_factor: int = 12,
                     must_contain: Sequence[Sequence[int]] = ()):
    """For every tensor product of rank-one factor defects with at least one non-invertible factor, the set of
    monomials it survives; returns the maximal such sets (as sorted index tuples) with example defects, and
    whether any of them contains all monomials in `must_contain`."""
    p = p or primes_for(exponents)[0]
    n = list(exponents)
    monos = [tuple(m) for m in monos]
    per = []
    for f, nf in enumerate(n):
        sets = [(0,)] + [S for S, _ in defect_sets(nf) if S != (0,)]
        seenS, d = set(), {}
        for S in sets:
            comp = tuple(sorted(set(range(nf)) - set(S)))
            if comp in seenS or (comp == (0,) and S != (0,)):
                continue
            seenS.add(S)
            st = tuple(factor_status(nf, S, m[f], p) for m in monos)
            d.setdefault(st, []).append(S)
        per.append(list(d.items()))
    fam = {}
    for combo in itertools.product(*per):
        noninv = any(any(len(S) > 1 for S in c[1]) for c in combo)
        if not noninv:
            continue
        # surviving monomials; phases can be re-chosen inside a class only if all members are |S|=1 -> keep simple:
        idx = tuple(j for j in range(len(monos)) if tensor_survives([c[0][j] for c in combo], p))
        ex = tuple(next((S for S in c[1] if len(S) > 1), c[1][0]) for c in combo)
        if idx not in fam:
            fam[idx] = ex
    keys = list(fam)
    maximal = [k for k in keys if not any(set(k) < set(o) for o in keys)]
    need = {monos.index(tuple(m)) for m in must_contain if tuple(m) in monos}
    hit = [k for k in maximal if need <= set(k)]
    return sorted(maximal, key=len, reverse=True), fam, hit
