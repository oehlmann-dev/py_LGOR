"""Permutation symmetries of Fermat LG orbifolds (new in 2.7).

A permutation sigma of fields with equal exponents is a symmetry of the orbifold theory iff it maps the
orbifold group G onto itself (sigma G sigma^-1 = G; elements of G are phase vectors).  Along a deformation
it must also preserve the deforming monomials: each one separately (generic coefficients) or the set of
them (equal coefficients).  sigma acts on the holomorphic 3-form (spectral-flow operator) by its sign, so an
odd permutation is an R-symmetry in 4D unless a phase with det -1 compensates it [inferred; the same
rule gives the Delta(54) R-part of T^6/Z3].  Typical outcomes: S_n of identical factors, S3 inside a
cubic torus (Delta(27) x| S3 / Delta(54) with the Z3 phases), swaps in quartic tori (D4), S5 of the quintic.
"""

from __future__ import annotations

import itertools
from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Model

Perm = Tuple[int, ...]


def _gens(model: Model):
    from .toroidal import _gens as g
    return g(model)


def _span(gens, n):
    from .toroidal import _span as s
    return s(gens, n)


def sign(p: Perm) -> int:
    sg, seen = 1, set()
    for i in range(len(p)):
        if i in seen:
            continue
        j, L = i, 0
        while j not in seen:
            seen.add(j)
            j = p[j]
            L += 1
        sg *= (-1) ** (L - 1)
    return sg


def candidate_count(model: Model) -> int:
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    c = 1
    for e in set(n):
        c *= factorial(n.count(e))
    return c


def normalizer_perms(model: Model, mons: Sequence[Sequence[int]] = (), each: bool = True,
                     cap: int = 4_000_000, G: Optional[set] = None) -> List[Perm]:
    """All permutations s (field f -> field s[f]) of fields with equal Fermat exponents with s G s^-1 = G and
    mapping every monomial in `mons` to itself (each=True) or the set `mons` to itself (each=False)."""
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    if candidate_count(model) > cap:
        raise ValueError(f"{candidate_count(model)} candidate permutations (> cap {cap}); pass a larger cap")
    gens = _gens(model)
    G = G if G is not None else _span(gens, n)
    classes: Dict[int, List[int]] = {}
    for f, e in enumerate(n):
        classes.setdefault(e, []).append(f)
    keys = sorted(classes)
    monset = {tuple(m) for m in mons}
    out = []
    for choice in itertools.product(*[itertools.permutations(classes[e]) for e in keys]):
        s = [0] * len(n)
        for e, img in zip(keys, choice):
            for a, b in zip(classes[e], img):
                s[a] = b
        ok = True
        for m in mons:
            h = [0] * len(n)
            for f in range(len(n)):
                h[s[f]] = m[f]
            h = tuple(h)
            if (each and h != tuple(m)) or (not each and h not in monset):
                ok = False
                break
        if not ok:
            continue
        for g in gens:
            h = [0] * len(n)
            for f in range(len(n)):
                h[s[f]] = g[f]
            if tuple(h) not in G:
                ok = False
                break
        if ok:
            out.append(tuple(s))
    return out


def cycle_text(model: Model, p: Perm) -> str:
    seen, cyc = set(), []
    for i in range(len(p)):
        if i in seen or p[i] == i:
            seen.add(i)
            continue
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(f"x{model.fields[j].name}")
            j = p[j]
        cyc.append("(" + " ".join(c) + ")")
    return "".join(cyc) or "id"


def structure(perms: List[Perm], max_classes: int = 5000) -> dict:
    """Order, number of odd elements, abelian or not, conjugacy-class count and a small-group guess."""
    from . import fingroups as fg
    info = {"order": len(perms), "odd": sum(1 for p in perms if sign(p) < 0)}
    if len(perms) <= max_classes:
        info["abelian"] = fg.is_abelian(perms)
        cl = fg.conj_classes(perms)
        info["classes"] = len(cl)
        sizes = sorted(len(c) for c in cl)
        info["class_sizes"] = sizes
        names = {(1, (1,)): "1", (2, (1, 1)): "Z2", (6, (1, 2, 3)): "S3", (6, (1,) * 6): "Z6",
                 (8, (1, 1, 2, 2, 2)): "D4 or Q8", (24, (1, 3, 6, 6, 8)): "S4", (12, (1, 3, 4, 4)): "A4",
                 (120, (1, 10, 15, 20, 20, 24, 30)): "S5", (4, (1, 1, 1, 1)): "Z4 or Z2xZ2",
                 (12, (1, 1, 2, 2, 3, 3)): "D6 = S3xZ2", (36, (1, 2, 2, 4, 6, 6, 6, 9)): "S3xS3",
                 (72, (1, 4, 4, 6, 6, 9, 12, 12, 18)): "S3 wr Z2 = (S3xS3) x| Z2"}
        info["guess"] = names.get((len(perms), tuple(sizes)))
    return info


def perms_text(model: Model, mons: Sequence[Sequence[int]] = (), label: str = "") -> str:
    out = [f"PERMUTATION SYMMETRIES of {model.name}" + (f" ({label})" if label else "")]
    try:
        P = normalizer_perms(model, mons, each=True)
        Pset = normalizer_perms(model, mons, each=False) if mons else None
    except ValueError as e:
        return "\n".join(out + [f"  {e}"]) + "\n"
    for lab, perms in [("generic coefficients" if mons else "Fermat point", P),
                       ("equal coefficients", Pset)]:
        if perms is None:
            continue
        st = structure(perms)
        g = f" = {st['guess']}" if st.get("guess") else ""
        ab = "" if "abelian" not in st else (", abelian" if st["abelian"] else ", non-abelian")
        cl = f", {st['classes']} classes" if "classes" in st else ""
        out.append(f"  {lab:22s} order {st['order']}{g}{ab}{cl}; odd (R-type unless phase-compensated): {st['odd']}")
        gens = []
        cur = {tuple(range(model.nfields))}
        from . import fingroups as fg
        for p in sorted(perms, key=lambda p: sum(1 for i, x in enumerate(p) if i != x)):
            if p not in cur:
                gens.append(p)
                cur = set(fg.closure(gens, model.nfields))
            if len(cur) == len(perms):
                break
        if gens:
            out.append("    generated by " + ", ".join(cycle_text(model, p) + (" [odd]" if sign(p) < 0 else "") for p in gens))
    return "\n".join(out) + "\n"
