"""Modular data of Fermat Gepner models (N=2 minimal models, simple currents, Verlinde lines).

Every Fermat field x_i^{n_i} is an N=2 minimal model at level k_i = n_i - 2.  Its primaries are
(l, m, s) with 0 <= l <= k, m mod 2n, s mod 4, l + m + s even, and the field identification
(l, m, s) ~ (k - l, m + n, s + 2).  Standard data (e.g. Gepner 1988; Greene-Plesser 1990):

    h(l,m,s) = l(l+2)/4n - m^2/4n + s^2/8          (mod 1)
    S_{(l,m,s),(l',m',s')} ~ sin(pi (l+1)(l'+1)/n) exp(i pi m m'/n) exp(-i pi s s'/2)
    fusion: su(2)_k in l, addition in m and s.

A simple current with l = 0, J = (0, m_J, s_J), has monodromy charge with a label a
    Q_J(a) = h_J + h_a - h_{J a} = sum_f [ m_J m_a / 2n_f - s_J s_a / 4 ]   (mod 1).
For an orbifold current J_g = (0, 2 a_i, 0) of the phase symmetry x_i -> exp(2 pi i a_i/n_i) x_i,
Q_{J_g}(a) is the phase by which g acts on the chiral primary, e.g. a l/n on x^l.

Currents used for a Fermat LGOR model with n_i = KOrder / q_i:
  beta_0 = spacetime spinor x (0,1,1) in every factor (spacetime supercharge, left: E6 currents)
  beta_i = spacetime vector x (0,0,2) in factor i     (worldsheet supercurrent of factor i)
  j      = (0,2,0) in every factor                     (the LG orbifold by J = exp(2 pi i J_0))
  g      = (0,2 a_i,0) for every orbifold generator of the .lgo file
  w_ij   = (0,0,2)_i (0,0,2)_j                         (alignment of NS/R between factors)
Only labels with l-independent currents occur, so Q_J depends on (m, s) only.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Model

Label = Tuple[int, int, int]          # (l, m, s) of one factor
PLabel = Tuple[Label, ...]            # product label


class Factor:
    def __init__(self, k: int):
        if k < 0:
            raise ValueError("level must be >= 0")
        self.k = k
        self.n = k + 2

    def __repr__(self):
        return f"Factor(k={self.k})"

    def canon(self, l: int, m: int, s: int) -> Label:
        k, n = self.k, self.n
        m %= 2 * n
        s %= 4
        if (l + m + s) % 2:
            raise ValueError(f"(l,m,s)=({l},{m},{s}) violates l+m+s even")
        a = (l, m, s)
        b = (k - l, (m + n) % (2 * n), (s + 2) % 4)
        return min(a, b)

    def labels(self) -> List[Label]:
        out = set()
        for l in range(self.k + 1):
            for m in range(2 * self.n):
                for s in range(4):
                    if (l + m + s) % 2 == 0:
                        out.add(self.canon(l, m, s))
        return sorted(out)

    def h(self, a: Label) -> Fr:
        l, m, s = a
        n = self.n
        return (Fr(l * (l + 2), 4 * n) - Fr(m * m, 4 * n) + Fr(s * s, 8)) % 1

    def dim(self, l: int) -> float:
        return math.sin(math.pi * (l + 1) / self.n) / math.sin(math.pi / self.n)

    def eig(self, a: Label, b: Label) -> complex:
        """S_{ab} / S_{0b}: eigenvalue of the Verlinde line a on the primary b."""
        (la, ma, sa), (lb, mb, sb) = a, b
        n = self.n
        su2 = math.sin(math.pi * (la + 1) * (lb + 1) / n) / math.sin(math.pi * (lb + 1) / n)
        return su2 * cmath.exp(1j * math.pi * ma * mb / n) * cmath.exp(-1j * math.pi * sa * sb / 2)

    def su2_fuse(self, a: int, b: int) -> List[int]:
        return list(range(abs(a - b), min(a + b, 2 * self.k - a - b) + 1, 2))

    def fuse(self, a: Label, b: Label) -> List[Label]:
        return [self.canon(c, a[1] + b[1], a[2] + b[2]) for c in self.su2_fuse(a[0], b[0])]

    def lclass(self, l: int) -> int:
        """Representative of l under l ~ k - l (the su(2) label up to the identification)."""
        return min(l, self.k - l)


def fermat_levels(model: Model) -> List[int]:
    ns = []
    for i in range(model.nfields):
        n = model.fermat_exponent(i)
        if n is None:
            raise ValueError("defect analysis needs a Fermat superpotential (x_i^n_i terms only)")
        ns.append(n)
    if any(len([p for p in t.powers if p]) != 1 for t in model.superpotential):
        raise ValueError("defect analysis needs a Fermat superpotential (x_i^n_i terms only)")
    return [n - 2 for n in ns]


# --------------------------------------------------------------------------- abelian currents
class Current:
    """A simple current with l = 0 in every factor: (m_f, s_f) per factor, plus a spacetime
    SO(2) label s0 (0 internal, 1 spinor, 2 vector).  Orbifold currents have s0 = 0."""

    def __init__(self, name: str, ms: Sequence[Tuple[int, int]], s0: int = 0, kind: str = "orbifold",
                 phase: Optional[Sequence[int]] = None):
        self.name = name
        self.ms = [tuple(x) for x in ms]
        self.s0 = s0
        self.kind = kind          # 'extension' (beta, w, j) or 'orbifold' (symmetry orbifold) or 'gauged'
        self.phase = list(phase) if phase is not None else None     # a_i (phase exp(2 pi i a_i / n_i))

    def __repr__(self):
        return f"Current({self.name})"


class GepnerData:
    """Product of N=2 minimal models of a Fermat LGOR model, with its currents."""

    def __init__(self, model: Model):
        self.model = model
        self.levels = fermat_levels(model)
        self.factors = [Factor(k) for k in self.levels]
        self.r = len(self.factors)
        self.ns = [f.n for f in self.factors]
        r = self.r
        self.beta0 = Current("beta_0 (spacetime supercharge / E6)", [(1, 1)] * r, s0=1, kind="extension")
        self.betas = [Current(f"beta_{i + 1} (supercurrent of factor {i + 1})",
                              [(0, 2) if j == i else (0, 0) for j in range(r)], s0=2, kind="extension")
                      for i in range(r)]
        self.jcur = Current("j (LG orbifold by J)", [(2, 0)] * r, kind="orbifold", phase=[1] * r)
        self.orbifold = []
        for j, N in enumerate(model.orders):
            a = []
            for i, f in enumerate(model.fields):
                c = f.charges[j]
                num = c * self.ns[i]
                if num % N:
                    raise ValueError(f"orbifold generator {j + 1} is not a phase symmetry of x_{i + 1}^{self.ns[i]}")
                a.append((num // N) % self.ns[i])
            self.orbifold.append(Current(f"g_{j + 1} (orbifold Z{N})", [(2 * x, 0) for x in a],
                                         kind="orbifold", phase=a))
        self.align = [Current(f"w_1{j + 1}", [(0, 2) if t in (0, j) else (0, 0) for t in range(r)], kind="extension")
                      for j in range(1, r)]

    # ------------------------------------------------------------- monodromies
    def Q(self, J: Current, line: PLabel) -> Fr:
        """Monodromy charge of the product label ``line`` (internal, s0 = 0) with the current J."""
        q = Fr(0)
        for (mJ, sJ), (l, m, s), n in zip(J.ms, line, self.ns):
            q += Fr(mJ * m, 2 * n) - Fr(sJ * s, 4)
        return q % 1

    def Qab(self, J: Current, ms: Sequence[Tuple[int, int]]) -> Fr:
        """Same, for an abelian part given as (m, s) per factor."""
        q = Fr(0)
        for (mJ, sJ), (m, s), n in zip(J.ms, ms, self.ns):
            q += Fr(mJ * m, 2 * n) - Fr(sJ * s, 4)
        return q % 1

    def h_current(self, J: Current) -> Fr:
        h = Fr(J.s0 * J.s0, 8)
        for (m, s), n in zip(J.ms, self.ns):
            h += -Fr(m * m, 4 * n) + Fr(s * s, 8)
        return h % 1

    def canon(self, line) -> PLabel:
        return tuple(f.canon(*a) for f, a in zip(self.factors, line))

    def dim(self, line) -> float:
        d = 1.0
        for f, a in zip(self.factors, line):
            d *= f.dim(a[0])
        return d

    def fuse(self, a: PLabel, b: PLabel) -> Dict[PLabel, int]:
        out: Dict[PLabel, int] = {(): 1}
        for i, f in enumerate(self.factors):
            new: Dict[PLabel, int] = {}
            for c in f.fuse(a[i], b[i]):
                for pre, mult in out.items():
                    key = pre + (c,)
                    new[key] = new.get(key, 0) + mult
            out = new
        return out

    def conj(self, a: PLabel) -> PLabel:
        return self.canon(tuple((l, -m, -s) for l, m, s in a))

    def identical_blocks(self) -> List[List[int]]:
        """Groups of factor indices with the same level and the same orbifold phases pattern-wise."""
        blocks: Dict[int, List[int]] = {}
        for i, k in enumerate(self.levels):
            blocks.setdefault(k, []).append(i)
        return [b for b in blocks.values() if len(b) > 1]
