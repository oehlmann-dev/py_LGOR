"""2D fixed-point data of an LG block from J-twisted, J^l-twined Witten indices (see package docstring)."""
import cmath
import itertools
import math
from collections import Counter
from fractions import Fraction as Fr

TAU0 = 1.1j + 0.13


# ----------------------------------------------------------------- 2. twisted-twined indices
def _theta1_series(tau, u, N=30):
    s = 0
    for n in range(-N, N):
        s += (-1) ** n * cmath.exp(1j * math.pi * tau * (n + 0.5) ** 2 + 2j * math.pi * (n + 0.5) * u)
    return -1j * s


def _logtheta1(tau, x, y, z):
    """log theta1(tau, x tau + y + z), x, y rational, z small; argument reduced by quasi-periodicity."""
    m, n = math.floor(x), math.floor(y)
    u0 = float(x - m) * tau + float(y - n) + z
    return (cmath.log(_theta1_series(tau, u0)) + 1j * math.pi * (m + n)
            - 1j * math.pi * m * m * tau - 2j * math.pi * m * u0)


def twined_index(qs, a, b, tau=TAU0, z=1e-9):
    c = float(sum(1 - 2 * q for q in qs))
    lg = 1j * math.pi * c * (a * a * tau + 2 * a * z) + 1j * math.pi * c * (a + b)
    for q in qs:
        lg += _logtheta1(tau, (1 - q) * a, (1 - q) * b, float(1 - q) * z)
        lg -= _logtheta1(tau, q * a, q * b, float(q) * z)
    return cmath.exp(lg)


def _as_charges(block):
    """block = Fermat exponents (ints) or R-charges q_i (Fractions)."""
    return [Fr(1, x) if isinstance(x, int) else Fr(x) for x in block]


def block_order(block):
    return math.lcm(*[q.denominator for q in _as_charges(block)])


def chat(block):
    return sum(1 - 2 * q for q in _as_charges(block))


def fixed_point_table(block, tau=TAU0):
    """chi(k,l) for an LG block (Fermat exponents or charges; any quasi-homogeneous W: the index only needs the
    charges).  Returns (K, table, all_integer).  Rows that are non-positive are flipped (uniform fermion number of
    the theta^k ground states, [inferred]); flipped rows are listed in table_signs()."""
    import numpy as np
    qs = _as_charges(block)
    K = block_order(qs)
    Z = np.array([[twined_index(qs, a, b, tau) for b in range(K)] for a in range(K)])
    kk = np.arange(K)
    E = np.exp(2j * np.pi * np.outer(kk, kk) / K)
    chi = E @ Z.T @ np.conj(E).T / K          # chi[k,l] = (1/K) sum_{a,b} e^{2 pi i (k b - l a)/K} Z[a,b]
    R = np.round(chi.real).astype(int)
    ok = bool(np.max(np.abs(chi - R)) < 1e-4)
    tab = [list(map(int, row)) for row in R]
    _SIGNS[(tuple(qs), K)] = []
    for k in range(1, K):
        if all(x <= 0 for x in tab[k]) and any(x < 0 for x in tab[k]):
            tab[k] = [-x for x in tab[k]]
            _SIGNS[(tuple(qs), K)].append(k)
    return K, tab, ok


_SIGNS = {}


def table_signs(block):
    qs = _as_charges(block)
    return _SIGNS.get((tuple(qs), block_order(qs)), [])


def euler_total(block, tau=TAU0):
    """(1/K) sum_{a,b} Z_{a,b}: Witten index of LG/J (sign convention check)."""
    qs = _as_charges(block)
    K = block_order(qs)
    return sum(twined_index(qs, a, b, tau) for a in range(K) for b in range(K)) / K


def _mobius(n):
    r, p = 1, 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0:
                return 0
            r = -r
        p += 1
    return -r if n > 1 else r


def orbit_counts(row, K):
    """row[l] = #fixed points of theta^l.  Returns {orbit size: number} or None if not a permutation character."""
    divs = [d for d in range(1, K + 1) if K % d == 0]
    out = {}
    for s in divs:
        tot = sum(_mobius(s // d) * row[d % K] for d in divs if s % d == 0)
        if tot < 0 or tot % s:
            return None
        if tot:
            out[s] = tot // s
    return out


