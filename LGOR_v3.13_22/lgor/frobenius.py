"""A_{r-1} Frobenius manifolds (Saito flat coordinates of x^r) and genus-0 correlators up to 5 points of a
Thom-Sebastiani product of such factors (tensor-product CohFT [literature]).  Validated against
examples02/rcft/tft5 (A4: 4-point -1/5, -1/5, -2/5; 5-point 0, 2/25) [verified]."""
import itertools
from fractions import Fraction as Fr
import sympy as sp

_FROB = {}


class Ar:
    def __init__(self, r):
        self.r = r
        x = sp.symbols('x')
        n = r - 1                     # ring basis x^0..x^{r-2}
        a = sp.symbols('a0:%d' % n)
        T = sp.symbols('t0:%d' % n)
        W = x**r + sum(a[i] * x**i for i in range(n))
        Nser = r + 3
        L = x * sp.series((W / x**r)**sp.Rational(1, r), x, sp.oo, Nser).removeO()
        t = {}
        for k in range(n):
            e = sp.expand(sp.series(L**(k + 1), x, sp.oo, Nser).removeO())
            t[n - 1 - k] = sp.simplify(sp.Rational(r, k + 1) * e.coeff(x, -1))
        sol = sp.solve([sp.Eq(T[j], t[j]) for j in range(n)], a, dict=True)[0]
        Wt = sp.expand(W.subs(sol))
        self.x, self.T, self.n = x, T, n
        self.Wp = sp.diff(Wt, x)
        self.dW = [sp.diff(Wt, T[j]) for j in range(n)]
        self.base = self._c3(0, 0, r - 2).subs({T[j]: 0 for j in range(n)})
        self.cache = {}

    def _c3(self, i, j, k):
        x = self.x
        q, rem = sp.div(sp.Poly(sp.expand(self.dW[i] * self.dW[j] * self.dW[k]), x), sp.Poly(self.Wp, x))
        return rem.as_expr().coeff(x, self.r - 2) / self.r

    def corr(self, lab):
        """genus-0 correlator <x^a1 ... x^an> (n = 3,4,5,...), flat coordinates, normalised <1 1 x^{r-2}> = 1"""
        lab = tuple(sorted(lab))
        if any(l < 0 or l > self.r - 2 for l in lab):
            return Fr(0)
        if lab not in self.cache:
            f = self._c3(*lab[:3]) / self.base
            for l in lab[3:]:
                f = sp.diff(f, self.T[l])
            self.cache[lab] = Fr(str(sp.nsimplify(f.subs({self.T[j]: 0 for j in range(self.n)}))))
        return self.cache[lab]

    def degree(self, lab):
        s = sum(lab) - (self.r - 2)
        if s < 0 or s % self.r:
            return None
        return s // self.r


def frob(r):
    if r not in _FROB:
        _FROB[r] = Ar(r)
    return _FROB[r]


# Mbar_{0,5}: boundary divisors D_kl, intersection form (as in tft5.py)
PAIRS = list(itertools.combinations(range(5), 2))
def _inter(p, q):
    if p == q:
        return -1
    return 1 if not set(p) & set(q) else 0
_I = [[_inter(p, q) for q in PAIRS] for p in PAIRS]
_BASIS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
_bi = [PAIRS.index(b) for b in _BASIS]
_G = sp.Matrix([[_I[i][j] for j in _bi] for i in _bi])
_Ginv = _G.inv()


def _deg1_class(F, lab):
    v = []
    for (k, l) in PAIRS:
        s = lab[k] + lab[l]
        rest = [lab[m] for m in range(5) if m not in (k, l)]
        v.append(F.corr([s] + rest) if s <= F.r - 2 else Fr(0))
    c = _Ginv * sp.Matrix([sp.Rational(v[i].numerator, v[i].denominator) for i in _bi])
    ok = all(sum(c[j] * _I[PAIRS.index(p)][_bi[j]] for j in range(5)) == sp.Rational(v[n].numerator, v[n].denominator)
             for n, p in enumerate(PAIRS))
    assert ok, "splitting inconsistency"
    return c


def product_corr(rs, states):
    """states: list of n tuples (exponent per factor); rs: list of r per factor.  n = 3, 4 or 5."""
    n = len(states)
    nf = len(rs)
    cols = [tuple(s[f] for s in states) for f in range(nf)]
    degs = []
    for f in range(nf):
        d = frob(rs[f]).degree(cols[f])
        if d is None:
            return Fr(0)
        degs.append(d)
    if sum(degs) != n - 3:
        return Fr(0)
    big = [f for f in range(nf) if degs[f] > 0]
    if n == 3:
        return Fr(1)
    if len(big) == 1:
        return frob(rs[big[0]]).corr(cols[big[0]])
    if n == 5 and len(big) == 2:
        cf = _deg1_class(frob(rs[big[0]]), cols[big[0]])
        cg = _deg1_class(frob(rs[big[1]]), cols[big[1]])
        val = (cf.T * _G * cg)[0]
        return Fr(int(sp.fraction(val)[0]), int(sp.fraction(val)[1]))
    raise NotImplementedError("n=%d with degree pattern %s" % (n, degs))
