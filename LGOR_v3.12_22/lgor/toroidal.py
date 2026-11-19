"""Toroidal orbifold phases of Fermat LG orbifolds (new in 2.7).

A Fermat c=3 block is an elliptic curve at a symmetric point.  There are three:

    A = (1)(1)(1)      x^3 + y^3 + z^3        hexagonal torus, Z3 rotation, size modulus  x y z
    B = (2)(2)(+z^2)   x^4 + y^4 (+ z^2)      square torus,    Z4 rotation, size modulus  x^2 y^2
    C = (4)(1)(+z^2)   x^6 + y^3 (+ z^2)      hexagonal torus, Z6 rotation, size modulus  x^4 y

(x^2 fields have c = 0 and are ignored for the geometry.)  If the factors of a c = 9 model split into
three such blocks and every element of the orbifold group G acts inside each block as a power of that
block's own J, the model is the Greene-Plesser mirror of the toroidal orbifold T^6/Gamma with

    Gamma = { r in Z_n1 x Z_n2 x Z_n3 : sum_i r_i b_i / n_i in Z  for all b in G }     (rotations),

and the three size moduli above are polynomial (27) moduli of the LG model = Kahler (size) moduli of
the tori in T^6/Gamma [inferred; reproduces the Hodge numbers of all 29 abelian T^6/Gamma we built and
the Z6-II twist (1/6,1/2,1/3)].  If instead the model's GP mirror has this property, the model itself is
T^6/Gamma at the symmetric point and the analysis is done on its mirror (same CFT, left N=2 flipped).

What this module computes (pure Python, no spectrum needed):
  * detection, block assignment, Gamma (generators, twist vectors, name, N=1 vs K3 x T^2 type);
  * the torus size moduli;
  * remnant symmetries at the Gepner point, along each size modulus and at generic finite size:
    phases of W + moduli modulo G, permutations normalising G, R vs non-R from det = sign(sigma) e^{2 pi i sum theta}
    (det != 1 means the element acts on Omega, i.e. a 4D R-symmetry [inferred]);
  * which Verlinde lines (and products) commute with each size modulus (tdl.keeps), with tags for
    odd-r lines (fermion-parity dressing open) and the torus reading of the line (class algebra).
The full spectrum along a family (gauge group, LGOR discrete symmetries) is `family` / `toroidal --family`.
"""

from __future__ import annotations

import itertools
from fractions import Fraction as Fr
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Model

BLOCK_N = {"A": 3, "B": 4, "C": 6}
BLOCK_TORUS = {"A": "hexagonal T^2 (Z3)", "B": "square T^2 (Z4)", "C": "hexagonal T^2 (Z6)"}

# Hodge numbers (h11, h21) of T^6/Gamma on factorised lattices, for orientation only [literature;
# lattice-dependent for Z4, Z6-II, Z2xZ4: the value listed is the one on the product of A2/Z2 tori]
KNOWN_HODGE = {
    "Z3": (36, 0), "Z4": (31, 7), "Z6-I": (29, 5), "Z6-II": (35, 11), "Z2xZ2": (51, 3),
    "Z3xZ3": (84, 0), "Z2xZ4": (61, 1), "Z4xZ4": (90, 0), "Z2xZ6": (51, 3), "Z2xZ6'": (36, 0),
    "Z3xZ6": (73, 1), "Z6xZ6": (84, 0),
}


# ----------------------------------------------------------------------------- small helpers
def _gens(model: Model) -> List[Tuple[int, ...]]:
    """Generators of the orbifold group as exponent vectors a (phase exp(2 pi i a_f / n_f) on field f)."""
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    gens = [tuple(f.kcharge * ni // model.korder for f, ni in zip(model.fields, n))]
    for j, o in enumerate(model.orders):
        v = [f.charges[j] * ni for f, ni in zip(model.fields, n)]
        if any(x % o for x in v):
            raise ValueError(f"discrete symmetry {j + 1} does not preserve W")
        gens.append(tuple((x // o) % ni for x, ni in zip(v, n)))
    return gens


def _span(gens: Sequence[Tuple[int, ...]], n: Sequence[int]) -> set:
    S = {tuple(0 for _ in n)}
    frontier = list(S)
    while frontier:
        new = []
        for s in frontier:
            for g in gens:
                t = tuple((x + y) % m for x, y, m in zip(s, g, n))
                if t not in S:
                    S.add(t)
                    new.append(t)
        frontier = new
    return S


def _abelian_invariants(elements: Sequence[Tuple[int, ...]], sub: set, n: Sequence[int]) -> List[int]:
    """Invariant factors of the finite abelian group <elements> / sub (elements and sub: vectors mod n)."""
    P = list(elements)
    order = len(P) // len(sub)
    if order == 1:
        return []

    def mult(x, d):
        return tuple((d * a) % m for a, m in zip(x, n))
    # for each prime p: a_k = log_p |{x : p^k x = 0}| determines the p-part
    primes, m = [], order
    q = 2
    while m > 1:
        if m % q == 0:
            primes.append(q)
            while m % q == 0:
                m //= q
        q += 1
    cyc: Dict[int, List[int]] = {}
    for p in primes:
        sizes, k = [0], 1
        while True:
            cnt = sum(1 for x in P if mult(x, p ** k) in sub) // len(sub)
            e = 0
            while cnt > 1:
                cnt //= p
                e += 1
            sizes.append(e)
            if e == sizes[-2]:
                break
            k += 1
        # number of cyclic factors of order >= p^k is sizes[k] - sizes[k-1]
        parts = []
        for k in range(1, len(sizes)):
            ge = sizes[k] - sizes[k - 1]
            parts.append(ge)
        exps = []
        for k in range(len(parts)):
            nxt = parts[k + 1] if k + 1 < len(parts) else 0
            exps += [k + 1] * (parts[k] - nxt)
        cyc[p] = sorted(exps, reverse=True)
    # combine into invariant factors d1 | d2 | ...
    L = max(len(v) for v in cyc.values())
    inv = []
    for i in range(L):
        d = 1
        for p, v in cyc.items():
            if i < len(v):
                d *= p ** v[i]
        inv.append(d)
    return sorted(inv)


def group_name(inv: Sequence[int]) -> str:
    return " x ".join(f"Z{d}" for d in inv) if inv else "trivial"


# ----------------------------------------------------------------------------- detection
class TorusPhase:
    """Result of the detection: blocks, Gamma, moduli.  `side` is 'model' (the model is the GP mirror of
    T^6/Gamma) or 'mirror' (the model's mirror is, i.e. the model is T^6/Gamma itself)."""

    def __init__(self, model: Model, blocks: List[Tuple[str, Tuple[int, ...]]], xsq: List[int], side: str):
        self.model, self.blocks, self.xsq, self.side = model, blocks, xsq, side
        self.n = [model.fermat_exponent(i) for i in range(model.nfields)]
        self.G = _span(_gens(model), self.n)
        self.bn = [BLOCK_N[t] for t, _ in blocks]
        # block exponents of G
        self.Gb = set()
        for g in self.G:
            self.Gb.add(tuple(g[fs[0]] % BLOCK_N[t] if t != "C" else g[fs[0]] % 6 for t, fs in blocks))
        self.Gamma = [r for r in itertools.product(*[range(m) for m in self.bn])
                      if all(sum(Fr(ri * bi, m) for ri, bi, m in zip(r, b, self.bn)).denominator == 1 for b in self.Gb)]

    # -- Gamma
    def gamma_invariants(self) -> List[int]:
        return _abelian_invariants(self.Gamma, {tuple(0 for _ in self.bn)}, self.bn)

    def _order(self, r) -> int:
        k = 1
        while any((k * x) % m for x, m in zip(r, self.bn)):
            k += 1
        return k

    def twist(self, r) -> Tuple[Fr, ...]:
        v = [Fr(x, m) for x, m in zip(r, self.bn)]
        return tuple(x if x <= Fr(1, 2) else x - 1 for x in v)

    def gamma_generators(self) -> List[Tuple[int, ...]]:
        inv = self.gamma_invariants()
        gens: List[Tuple[int, ...]] = []
        cur = {tuple(0 for _ in self.bn)}
        for d in sorted(inv, reverse=True):
            for r in sorted(self.Gamma, key=lambda r: (-self._order(r), r)):
                if self._order(r) == d and r not in cur:
                    new = _span(gens + [r], self.bn)
                    if len(new) == len(cur) * d:
                        gens.append(r)
                        cur = new
                        break
        return gens

    def fixed_tori(self) -> List[int]:
        """Blocks on which every element of Gamma acts trivially (then T^6/Gamma = (T^4/Gamma) x T^2)."""
        return [i for i in range(3) if all(r[i] == 0 for r in self.Gamma)]

    def name(self) -> str:
        inv = self.gamma_invariants()
        base = "x".join(f"Z{d}" for d in inv) if inv else "trivial"
        if inv == [6]:
            g = self.gamma_generators()[0]
            s = sorted(min(x * 6 // m % 6, 6 - x * 6 // m % 6) for x, m in zip(g, self.bn))
            base = "Z6-I" if s == [1, 1, 2] else "Z6-II" if s == [1, 2, 3] else base
        if inv == [2, 6]:
            fixes = any(self._order(r) == 6 and 0 in r for r in self.Gamma)
            base = "Z2xZ6" if fixes else "Z2xZ6'"
        if self.fixed_tori():
            base += " on T^4 only (K3 x T^2, N=2)"
        return base

    # -- moduli
    def moduli(self) -> List[Tuple[str, Tuple[int, ...]]]:
        out = []
        for i, (t, fs) in enumerate(self.blocks):
            e = [0] * self.model.nfields
            if t == "A":
                for f in fs:
                    e[f] = 1
            elif t == "B":
                e[fs[0]] = e[fs[1]] = 2
            else:
                e[fs[0]], e[fs[1]] = 4, 1
            out.append((f"T{i + 1}", tuple(e)))
        return out

    def mono_text(self, e: Sequence[int]) -> str:
        parts = []
        for f, x in enumerate(e):
            if x:
                parts.append(f"x{self.model.fields[f].name}" + (f"^{x}" if x > 1 else ""))
        return " ".join(parts) if parts else "1"

    def deform_terms(self, which: Sequence[int] = (0, 1, 2), coeffs=(3, 5, 7)) -> List[str]:
        """The size moduli as `deform` terms, with distinct coefficients (generic sizes).  2.7.2: not 2, since
        x^4 + y^4 + 2 x^2 y^2 = (x^2 + y^2)^2 is singular (B torus degenerates); A is singular at a^3 = -27, C at 4a^3 = -27."""
        return [f"{coeffs[i % len(coeffs)]} " + self.mono_text(self.moduli()[i][1]) for i in which]


def _assignments(n: Sequence[int]):
    idx3 = [i for i, x in enumerate(n) if x == 3]
    idx4 = [i for i, x in enumerate(n) if x == 4]
    idx6 = [i for i, x in enumerate(n) if x == 6]
    xsq = [i for i, x in enumerate(n) if x == 2]
    if any(x not in (2, 3, 4, 6) for x in n):
        return
    if len(idx4) % 2 or len(idx3) < len(idx6) or (len(idx3) - len(idx6)) % 3:
        return
    if len(idx4) // 2 + len(idx6) + (len(idx3) - len(idx6)) // 3 != 3:
        return

    def pairings(s):
        if not s:
            yield []
            return
        a = s[0]
        for j in range(1, len(s)):
            for rest in pairings(s[1:j] + s[j + 1:]):
                yield [(a, s[j])] + rest

    def triples(s):
        if not s:
            yield []
            return
        a = s[0]
        for j, k in itertools.combinations(range(1, len(s)), 2):
            rest = [x for t, x in enumerate(s) if t not in (0, j, k)]
            for r in triples(rest):
                yield [(a, s[j], s[k])] + r

    for Bs in pairings(idx4):
        for partners in itertools.permutations(idx3, len(idx6)):
            Cs = list(zip(idx6, partners))
            rest = [i for i in idx3 if i not in partners]
            for As in triples(rest):
                blocks = [("A", t) for t in As] + [("B", p) for p in Bs] + [("C", c) for c in Cs]
                yield sorted(blocks, key=lambda b: ("CBA".index(b[0]), b[1])), xsq


def _block_diagonal(gens, blocks, n) -> bool:
    for g in gens:
        for t, fs in blocks:
            b = g[fs[0]]
            m = BLOCK_N[t]
            if any((g[f] - b) % n[f] for f in fs):
                return False
            if t == "C" and (g[fs[1]] - b) % 3:
                return False
            _ = m
    return True


def detect_blocks(model: Model) -> Optional[TorusPhase]:
    """TorusPhase with side='model' if the model's orbifold group is block-diagonal on three torus blocks."""
    if not model.is_fermat:
        return None
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    gens = _gens(model)
    for blocks, xsq in _assignments(n):
        if _block_diagonal(gens, blocks, n):
            return TorusPhase(model, blocks, xsq, "model")
    return None


def torus_blocks_only(model: Model) -> Optional[List[Tuple[str, Tuple[int, ...]]]]:
    """Block types if the factors split into three c=3 blocks at all (whatever G does)."""
    if not model.is_fermat:
        return None
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    for blocks, _ in _assignments(n):
        return blocks
    return None


def detect(model: Model) -> Tuple[Optional[TorusPhase], str]:
    """(phase, note).  Tries the model, then its GP mirror."""
    if not model.is_fermat:
        return None, "not a Fermat model (toroidal detection needs Fermat blocks)"
    if torus_blocks_only(model) is None:
        n = sorted(model.fermat_exponent(i) for i in range(model.nfields))
        return None, f"exponents {n} do not split into three c=3 torus blocks (A=(3,3,3), B=(4,4), C=(6,3))"
    ph = detect_blocks(model)
    if ph is not None:
        return ph, "the model is the GP mirror of T^6/Gamma; its size moduli are polynomial deformations"
    from .mirror import mirror_model
    try:
        mir = mirror_model(model)
    except Exception as e:                       # pragma: no cover
        return None, f"blocks fit, G is not block-diagonal, mirror failed: {e}"
    ph = detect_blocks(mir)
    if ph is not None:
        ph.side = "mirror"
        return ph, ("the model itself is T^6/Gamma at the symmetric point; analysis done on its GP mirror "
                    f"{mir.name} (same CFT, left N=2 flipped; lines map across mirror)")
    return None, ("the factors split into torus blocks, but neither G nor the mirror group acts block-diagonally "
                  "(a non-geometric / asymmetric orbifold of T^6) [inferred]")


# ----------------------------------------------------------------------------- remnant symmetries
class Remnants:
    """Symmetries of W + (chosen size moduli) in the orbifold theory: phases mod G and field permutations."""

    def __init__(self, ph: TorusPhase, on: Sequence[int], equal_sizes: bool = False):
        self.ph, self.on, self.equal = ph, list(on), equal_sizes
        n = ph.n
        mons = [ph.moduli()[i][1] for i in self.on]
        L = 1
        for m in n:
            L = L * m // gcd(L, m)
        w = [L // m for m in n]
        rows = [[wi * e for wi, e in zip(w, mon)] for mon in mons]
        self.P = [a for a in itertools.product(*[range(m) for m in n])
                  if all(sum(x * c for x, c in zip(a, row)) % L == 0 for row in rows)]
        self.inv = _abelian_invariants(self.P, ph.G, n)
        self.phase_order = len(self.P) // len(ph.G)
        # R-type phases: det = exp(2 pi i sum a_f / n_f)
        dets = {sum(x * wi for x, wi in zip(a, w)) % L for a in self.P}
        self.phase_R = max(L // gcd(L, d) for d in dets)
        self.perms, self.perm_R = self._perms(mons)

    def _perms(self, mons):
        """Permutations of equal-exponent fields normalising G and fixing each modulus (generic sizes) or
        the set of moduli (equal sizes); at the Gepner point all of them (they may mix blocks)."""
        from .perms import normalizer_perms, sign
        perms = normalizer_perms(self.ph.model, mons, each=not self.equal, G=self.ph.G)
        return perms, sum(1 for p in perms if sign(p) < 0)

    def order(self) -> int:
        return self.phase_order * len(self.perms)

    def r_order(self) -> int:
        """Order of the image of det (phases and permutation signs together)."""
        d = self.phase_R
        if self.perm_R:
            d = d * 2 // gcd(d, 2)
        return d

    def text(self) -> str:
        p = len(self.perms)
        s = (f"phases mod G: {group_name(self.inv)} (order {self.phase_order}); "
             f"permutations normalising G: {p}" + (f" ({self.perm_R} odd)" if self.perm_R else ""))
        s += f"; total order {self.order()}; det image Z{self.r_order()}"
        s += " -> R-symmetry part present" if self.r_order() > 1 else " -> no R-type element"
        return s


# ----------------------------------------------------------------------------- lines
LINE_TAG = {
    (1, 1): "invertible (-1)^l",
    (2, 1): "k=2 L1: Ising sigma, TY(Z2), d=sqrt2",
    (4, 1): "k=4 L1: d=sqrt3",
    (4, 2): "k=4 L2: Rep(S3) fusion with L4, d=2",
}
TORUS_READING = {
    ("B", 2, 1): "on the Z4 torus: Z2 fixed-point class sum (Ising) [inferred]",
    ("C", 4, 2): "on the Z6 torus: class algebra of S3 of the hexagonal lattice [inferred, cf. Z6-II]",
    ("C", 4, 1): "on the Z6 torus: no class-algebra reading found [missing]",
}


def phase_dressing(ph: TorusPhase, signs: Sequence[Optional[int]]) -> Optional[Tuple[int, ...]]:
    """A phase symmetry g of W with eigenvalue -1 on the size moduli where signs[j] == -1 and +1 where
    signs[j] == +1 (None: ignored), or None."""
    from .family import phase_dressing as pd
    return pd(ph.n, [m for _, m in ph.moduli()], signs)


def line_table(ph: TorusPhase, max_factors: int = 2) -> List[dict]:
    """Products of non-invertible Verlinde lines (at most `max_factors` non-trivial factors, r <= k/2) and
    their action on each size modulus: +1 kept, -1 kept after a phase dressing, None broken."""
    from .family import line_rows
    block_of = {f: t for t, fs in ph.blocks for f in fs}
    K = [m - 2 for m in ph.n]
    rows = line_rows(ph.model, [m for _, m in ph.moduli()], max_factors)
    for r in rows:
        tags = []
        for f, x in zip(r["factors"], r["rs"]):
            tags.append(LINE_TAG.get((K[f], x), f"k={K[f]} r={x}"))
            rd = TORUS_READING.get((block_of.get(f), K[f], x))
            if rd:
                tags.append(rd)
        r["tags"] = sorted(set(tags))
    return rows


# ----------------------------------------------------------------------------- report
def _survives(r) -> bool:
    return all(a is not None for a in r["act"]) and (r["dress"] is not None or all(a == 1 for a in r["act"]))


def _dress_text(ph: TorusPhase, a) -> str:
    from .family import dress_text
    return dress_text(ph.model, a)


def _fmt_act(a):
    return {1: "kept", -1: "kept*", None: "broken"}[a]


def report(model: Model, max_factors: int = 2, family: bool = False, processes: int = 1, progress=None) -> str:
    ph, note = detect(model)
    out = [f"TOROIDAL PHASE of {model.name}", "=" * (18 + len(model.name))]
    if ph is None:
        out.append("No toroidal orbifold phase: " + note)
        return "\n".join(out) + "\n"
    out.append(note + ".")
    out.append("")
    out.append("Blocks (fields numbered from 1):")
    for i, (t, fs) in enumerate(ph.blocks):
        out.append(f"  T{i + 1}: {t} = {BLOCK_TORUS[t]}, fields " + ", ".join(f"x{ph.model.fields[f].name}" for f in fs))
    if ph.xsq:
        out.append("  x^2 fields (c=0, ignored for geometry): " + ", ".join(f"x{ph.model.fields[f].name}" for f in ph.xsq))
    inv = ph.gamma_invariants()
    nm = ph.name()
    out.append("")
    out.append(f"Gamma = {nm}   (|Gamma| = {len(ph.Gamma)}, abstract {group_name(inv)})")
    for g in ph.gamma_generators():
        tw = ", ".join(str(x) for x in ph.twist(g))
        out.append(f"  generator r = {g}  twist ({tw})")
    base = nm.split(" ")[0]
    if base in KNOWN_HODGE:
        out.append(f"  T^6/{base} on factorised lattices: (h11,h21) = {KNOWN_HODGE[base]} [literature]; "
                   "LGOR's (h21,h11) of this model should match it" + (" after the mirror swap" if ph.side == "model" else ""))
    out.append("")
    out.append("Torus size moduli (polynomial deformations of the model analysed):")
    for lab, e in ph.moduli():
        out.append(f"  {lab}: {ph.mono_text(e)}")
    out.append("  (off-diagonal untwisted moduli of T^6/Gamma, when present, sit in twisted LG sectors; not listed)")
    out.append("")
    out.append("REMNANT SYMMETRIES (of W + moduli in the orbifold theory; R-type = det != 1 [inferred])")
    cases = [("Gepner point", [], False)] + [(f"along {lab}", [i], False) for i, (lab, _) in enumerate(ph.moduli())]
    cases += [("generic finite size (T1,T2,T3)", [0, 1, 2], False), ("equal sizes of equal tori", [0, 1, 2], True)]
    for lab, on, eq in cases:
        R = Remnants(ph, on, eq)
        out.append(f"  {lab:32s} {R.text()}")
    out.append("  Some of these phases may sit inside the gauge group or the E6 centre; `family` gives LGOR's")
    out.append("  gauge-reduced discrete symmetries for the same points.")
    out.append("")
    out.append("LINES vs SIZE MODULI (kept = eigenvalue +d; kept* = -d, needs a phase dressing g with g = -1 there; broken)")
    rows = line_table(ph, max_factors)
    if not rows:
        out.append("  no non-invertible Verlinde lines (all factors k <= 1): only invertible symmetries.")
    else:
        mods = [lab for lab, _ in ph.moduli()]
        out.append(f"  {'line':28s} {'d':>6s}  " + "  ".join(f"{m:>7s}" for m in mods) + "   all")
        surv = []
        for r in rows:
            allk = _survives(r)
            out.append(f"  {r['line']:28s} {r['d']:6.3f}  " + "  ".join(f"{_fmt_act(a):>7s}" for a in r["act"])
                       + ("   SURVIVES" + (f" with phase dressing {_dress_text(ph, r['dress'])}"
                                         if any(a == -1 for a in r["act"]) else "") if allk else
                          ("   broken (no phase dressing gives the signs)" if all(a is not None for a in r["act"]) else "")))
            if allk:
                surv.append(r)
        out.append("")
        if surv:
            out.append("Non-invertible lines surviving generic finite torus sizes:")
            for r in surv:
                dr = f", phase-dressed by {_dress_text(ph, r['dress'])}" if any(a == -1 for a in r["act"]) else ""
                out.append(f"  {r['line']} (d={r['d']:.3f}{dr})" + ("  [odd r: fermion-parity dressing open]" if r["odd"] else ""))
            out.append("  Caveat: survival is tree-level and only shows the line commutes with these moduli; whether it")
            out.append("  forbids anything beyond invertible symmetries needs `lineonly` (lattice + u1_forced tests).")
        else:
            out.append("No non-invertible line survives generic finite torus sizes.")
        tags = sorted({t for r in rows for t in r["tags"]})
        if tags:
            out.append("")
            out.append("Line types present:")
            for t in tags:
                out.append("  - " + t)
    if family:
        from .family import family_text
        out.append("")
        out.append(family_text(ph.model, ph.deform_terms(), processes=processes, progress=progress,
                               max_factors=max_factors))
    return "\n".join(out) + "\n"


def summary_dict(model: Model, max_factors: int = 2) -> dict:
    ph, note = detect(model)
    if ph is None:
        return {"model": model.name, "toroidal": False, "note": note}
    rows = line_table(ph, max_factors)
    rem = {}
    for lab, on, eq in [("gepner", [], False), ("finite", [0, 1, 2], False), ("equal", [0, 1, 2], True)]:
        R = Remnants(ph, on, eq)
        rem[lab] = {"phases": group_name(R.inv), "phase_order": R.phase_order, "perms": len(R.perms),
                    "order": R.order(), "det_order": R.r_order()}
    return {"model": model.name, "toroidal": True, "side": ph.side, "gamma": ph.name(),
            "blocks": "".join(t for t, _ in ph.blocks),
            "moduli": {lab: ph.mono_text(e) for lab, e in ph.moduli()},
            "remnants": rem,
            "lines_surviving": [r["line"] + (" (dressed)" if any(a == -1 for a in r["act"]) else "") for r in rows if _survives(r)],
            "lines_total": len(rows)}
