"""Model definition: fields, charges, superpotential, and the .lgo file format.

An .lgo file looks like::

    Name = "Q1Fermat2"
    KOrder = 5
    Order = 5 5          # orders of the discrete Z_n symmetries (may be empty)
    Field "1" 1 1 0      # name, R-charge numerator q_i (alpha_i = q_i / KOrder), discrete charges
    ...
    SPotTerm 1 5 0 0 0 0 # coefficient, then the power of every field

Lines with any other leading word are ignored (as in the original C++).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List, Optional


@dataclass
class Field:
    name: str
    kcharge: int                 # R-charge numerator: alpha = kcharge / KOrder
    charges: List[int]           # charges under the discrete symmetries


@dataclass
class SuperpotentialTerm:
    coeff: int
    powers: List[int]


@dataclass
class Model:
    name: str
    korder: int
    orders: List[int]
    fields: List[Field]
    superpotential: List[SuperpotentialTerm]
    filename: Optional[str] = None
    # optional symmetry generators given by hand (see lgor.symmetries)
    u1_generators: list = field(default_factory=list)          # [(name, [coeff per field]), ...]
    r_correction: Optional[List[int]] = None
    discrete_corrections: dict = field(default_factory=dict)   # {j: [coeff per field]}

    # derived integers (computed in __post_init__)
    N: int = field(init=False)          # "OrderHCF": common denominator of all twists
    korder_f: int = field(init=False)   # N / KOrder
    order_f: List[int] = field(init=False)  # N / order_j

    def __post_init__(self):
        n = 2 * self.korder
        for o in self.orders:
            n = n * o // math.gcd(n, o)
        self.N = n
        self.korder_f = n // self.korder
        self.order_f = [n // o for o in self.orders]

    # ------------------------------------------------------------------ basics
    @property
    def nfields(self) -> int:
        return len(self.fields)

    @property
    def ndiscrete(self) -> int:
        return len(self.orders)

    # ------------------------------------------------------------------ checks
    def check(self) -> List[str]:
        """Return a list of consistency problems (empty if the model is fine).

        Same checks as ModelData::CheckConsistency in the C++ code.
        """
        problems = []
        for j in range(self.ndiscrete):
            s = sum(f.charges[j] for f in self.fields)
            if s != 0:
                problems.append(f"charges of discrete symmetry {j} do not sum to zero ({s})")
        c = sum(self.korder - 2 * f.kcharge for f in self.fields)
        if c != 3 * self.korder:
            problems.append("central charge is not c = 9 (sum of (1 - 2 alpha_i) != 3)")
        for t, term in enumerate(self.superpotential):
            if any(p < 0 for p in term.powers):
                problems.append(f"superpotential term {t} is not holomorphic")
                continue
            r = sum(p * f.kcharge for p, f in zip(term.powers, self.fields))
            if r != self.korder:
                problems.append(f"superpotential term {t} does not have R-charge 1")
            for j in range(self.ndiscrete):
                q = sum(p * f.charges[j] for p, f in zip(term.powers, self.fields))
                if q % self.orders[j] != 0:
                    problems.append(f"superpotential term {t} is not invariant under discrete symmetry {j}")
        return problems

    @property
    def is_fermat(self) -> bool:
        """True if W is a sum of pure powers, one x_i^{n_i} per field (the case the fast
        zero-mode truncation of Sector assumes)."""
        seen = set()
        for term in self.superpotential:
            nz = [k for k, p in enumerate(term.powers) if p != 0]
            if len(nz) != 1:
                return False
            seen.add(nz[0])
        return len(seen) == self.nfields

    def blocks(self) -> List[List[int]]:
        """Connected components of the fields, two fields being linked if they share a term of W."""
        parent = list(range(self.nfields))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for term in self.superpotential:
            nz = [k for k, p in enumerate(term.powers) if p != 0]
            for k in nz[1:]:
                parent[find(k)] = find(nz[0])
        comp = {}
        for i in range(self.nfields):
            comp.setdefault(find(i), []).append(i)
        return sorted(comp.values())

    def jacobian_basis(self, block: Sequence[int]):
        """Monomial basis (exponent tuples over `block`) of the Jacobian ring of W restricted to the
        fields of `block`. Pure Python: degree by degree (weights kcharge, W of degree KOrder) the
        ideal (dW) is row reduced over the rationals; the non-pivot monomials are the basis (Fermat-box
        monomials preferred, see below, then grevlex standard monomials). Returns None if the ring is
        not finite dimensional (dW not a regular sequence: Hilbert function differs from
        prod (1 - t^(K - w)) / (1 - t^w), or the quotient does not vanish above the socle)."""
        from fractions import Fraction
        key = tuple(block)
        cache = self.__dict__.setdefault("_jac_cache", {})
        if key in cache:
            return cache[key]
        n, K = len(block), self.korder
        w = [self.fields[k].kcharge for k in block]
        W = {}
        for term in self.superpotential:
            if any(term.powers[k] != 0 for k in range(self.nfields) if k not in block):
                continue
            e = tuple(term.powers[k] for k in block)
            W[e] = W.get(e, 0) + Fraction(term.coeff)
        derivs = []
        for j in range(n):
            d = {}
            for e, c in W.items():
                if e[j] > 0:
                    f = list(e); f[j] -= 1
                    d[tuple(f)] = d.get(tuple(f), 0) + c * e[j]
            derivs.append({e: c for e, c in d.items() if c != 0})

        mono_cache = {}

        def monos(D, j=0):
            if (D, j) in mono_cache:
                return mono_cache[(D, j)]
            if j == n - 1:
                out = [(D // w[j],)] if D % w[j] == 0 else []
            else:
                out = [(a,) + r for a in range(D // w[j] + 1) for r in monos(D - a * w[j], j + 1)]
            mono_cache[(D, j)] = out
            return out

        def grevlex_key(e):          # larger key = larger in grevlex (same weighted degree)
            return (sum(e),) + tuple(-x for x in reversed(e))

        # Preferred representatives: the Fermat box e_j <= n_j - 2 for every field with a pure power
        # x_j^{n_j} in W. They are put last, so they stay non-pivot whenever they span the quotient
        # (a deformed Fermat block then keeps the same representatives as the Fermat point, which the
        # state matching of `deform`/`diff` relies on); otherwise grevlex picks the rest.
        cap = [None] * n
        for e in W:
            nz = [j for j in range(n) if e[j]]
            if len(nz) == 1:
                cap[nz[0]] = e[nz[0]] - 2

        def in_box(e):
            return all(c is None or x <= c for x, c in zip(e, cap))

        def quotient(D):
            cols = sorted(monos(D), key=lambda e: (not in_box(e), grevlex_key(e)), reverse=True)
            idx = {e: i for i, e in enumerate(cols)}
            rows = []
            for j in range(n):
                for m in monos(D - (K - w[j])) if D >= K - w[j] else []:
                    rows.append({idx[tuple(a + b for a, b in zip(m, e))]: c for e, c in derivs[j].items()})
            pivots = {}
            for r in rows:
                r = dict(r)
                while r:
                    p = min(r)
                    if p in pivots:
                        f = r[p]
                        for c, v in pivots[p].items():
                            nv = r.get(c, 0) - f * v
                            if nv == 0:
                                r.pop(c, None)
                            else:
                                r[c] = nv
                    else:
                        f = r[p]
                        pivots[p] = {c: v / f for c, v in r.items()}
                        break
            return [cols[i] for i in range(len(cols)) if i not in pivots]

        # expected Hilbert function prod (1 - t^(K-w)) / (1 - t^w), coefficients up to the socle
        socle = sum(K - 2 * x for x in w)
        expected = [0] * (socle + 1)
        expected[0] = 1
        for x in w:
            for D in range(x, socle + 1):        # divide by (1 - t^x)
                expected[D] += expected[D - x]
        for x in w:
            for D in range(socle, K - x - 1, -1):  # multiply by (1 - t^(K-x))
                expected[D] -= expected[D - (K - x)]
        basis = []
        for D in range(socle + 1):
            q = quotient(D)
            if len(q) != expected[D]:
                cache[key] = None
                return None
            basis.extend(q)
        if any(quotient(D) for D in range(socle + 1, socle + max(w) + 1)):
            cache[key] = None
            return None
        cache[key] = basis
        return basis

    def fermat_exponent(self, i: int) -> Optional[int]:
        """n_i such that x_i^{n_i} is a term of W, or None."""
        for term in self.superpotential:
            nz = [k for k, p in enumerate(term.powers) if p != 0]
            if nz == [i]:
                return term.powers[i]
        return None

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_file(cls, path) -> "Model":
        path = Path(path)
        text = path.read_text(errors="replace")
        m = cls.from_string(text)
        m.filename = str(path)
        return m

    @classmethod
    def from_string(cls, text: str) -> "Model":
        name = "NoName"
        korder = None
        orders: List[int] = []
        fields: List[Field] = []
        spot: List[SuperpotentialTerm] = []
        u1s, rcorr, dcorr = [], None, {}
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            m = re.match(r"([A-Za-z]+)", line)
            if not m:
                continue
            key = m.group(1)
            rest = line[m.end():]
            if key == "Name":
                q = re.search(r'"([^"]*)"', rest)
                if q:
                    name = q.group(1)
            elif key in ("KOrder", "Korder"):
                ints = re.findall(r"\d+", rest)
                if ints:
                    korder = int(ints[0])
            elif key == "Order" and not fields:
                orders = [int(x) for x in re.findall(r"\d+", rest)]
            elif key == "Field" and not spot:
                q = re.search(r'"([^"]*)"', rest)
                if not q:
                    raise ValueError(f"line {lineno}: field without a quoted name: {raw!r}")
                nums = [int(x) for x in re.findall(r"-?\d+", rest[q.end():])]
                if len(nums) < 1 + len(orders):
                    raise ValueError(f"line {lineno}: field needs 1 + {len(orders)} charges: {raw!r}")
                fields.append(Field(q.group(1), nums[0], nums[1:1 + len(orders)]))
            elif key == "U1":
                q = re.search(r'"([^"]*)"', rest)
                nm = q.group(1) if q else f"U1_{len(u1s) + 1}"
                nums = re.findall(r"-?\d+(?:/\d+)?", rest[q.end():] if q else rest)
                u1s.append((nm, nums))
            elif key == "RCorrection":
                rcorr = [int(x) for x in re.findall(r"-?\d+", rest)]
            elif key == "DiscreteCorrection":
                nums = [int(x) for x in re.findall(r"-?\d+", rest)]
                dcorr[nums[0] - 1] = nums[1:]
            elif key == "SPotTerm":
                nums = [int(x) for x in re.findall(r"-?\d+", rest)]
                if len(nums) < 1 + len(fields):
                    raise ValueError(f"line {lineno}: superpotential term needs 1 + {len(fields)} numbers: {raw!r}")
                spot.append(SuperpotentialTerm(nums[0], nums[1:1 + len(fields)]))
        if korder is None:
            raise ValueError("no KOrder given")
        if not fields:
            raise ValueError("no fields given")
        if not spot:
            raise ValueError("no superpotential given")
        from fractions import Fraction
        n = len(fields)
        for nm, c in u1s:
            if len(c) != n:
                raise ValueError(f"U1 {nm} needs {n} coefficients")
        if rcorr is not None and len(rcorr) != n:
            raise ValueError(f"RCorrection needs {n} coefficients")
        for j, c in dcorr.items():
            if len(c) != n or not 0 <= j < len(orders):
                raise ValueError("DiscreteCorrection needs: symmetry number (1, 2, ...) and one coefficient per field")
        return cls(name, korder, orders, fields, spot, None,
                   [(nm, [Fraction(x) for x in c]) for nm, c in u1s], rcorr, dcorr)

    def to_string(self) -> str:
        lines = [f'Name = "{self.name}"', f"KOrder = {self.korder}",
                 "Order = " + " ".join(str(o) for o in self.orders)]
        for f in self.fields:
            lines.append(f'Field "{f.name}" ' + " ".join(str(x) for x in [f.kcharge] + f.charges))
        for t in self.superpotential:
            lines.append("SPotTerm " + " ".join(str(x) for x in [t.coeff] + t.powers))
        for nm, c in self.u1_generators:
            lines.append(f'U1 "{nm}" ' + " ".join(str(x) for x in c))
        if self.r_correction is not None:
            lines.append("RCorrection " + " ".join(map(str, self.r_correction)))
        for j, c in sorted(self.discrete_corrections.items()):
            lines.append(f"DiscreteCorrection {j + 1} " + " ".join(map(str, c)))
        return "\n".join(lines) + "\n"

    def describe(self) -> str:
        out = [f"Model {self.name}: {self.nfields} fields, KOrder {self.korder}, "
               f"discrete orders {self.orders or 'none'}, {len(self.superpotential)} superpotential terms"]
        hdr = "  field   q   " + "  ".join(f"Z{o}" for o in self.orders)
        out.append(hdr)
        for f in self.fields:
            out.append(f"  {f.name:<6} {f.kcharge:>2}   " + "  ".join(f"{c:>2}" for c in f.charges))
        return "\n".join(out)


def _compute(self, processes: int = 1, **kwargs):
    """Compute the massless spectrum; returns a :class:`lgor.spectrum.ModelSpectrum`."""
    from .spectrum import compute_model
    return compute_model(self, processes=processes, **kwargs)


Model.compute = _compute
