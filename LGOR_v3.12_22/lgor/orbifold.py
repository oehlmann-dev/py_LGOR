"""Build LG orbifold (.lgo) files for toroidal orbifolds T^6/Gamma (new in 2.7; port of make_lgo.py).

Three torus blocks, each A = (1)^3 (Z3 torus), B = (2)^2 + z^2 (Z4 torus) or C = (4)(1) + z^2 (Z6 torus);
Gamma <= Z_n1 x Z_n2 x Z_n3 is given by generators r (rotation units per torus) with sum_i r_i / n_i in Z.
The A-side orbifold group is G_A = annihilator of Gamma under <r, a> = sum_i r_i a_i / n_i; it contains the
total J.  LGOR's orbifold group is J times a direct-product complement H of <J> in G_A; the charges of H are
written as extra discrete columns (shifted so that each column sums to exactly 0).  The resulting model is
the Greene-Plesser mirror of T^6/Gamma at the symmetric point [inferred; all 29 cases reproduce the Hodge
numbers of T^6/Gamma].  A z^2 field is kept for every B and C block, so there are always 9 fields.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product
from math import lcm
from pathlib import Path
from typing import List, Sequence, Tuple

BLOCK = {"A": (3, [F(1, 3)] * 3), "B": (4, [F(1, 4)] * 2 + [F(1, 2)]), "C": (6, [F(1, 6), F(1, 3), F(1, 2)])}

# the 29 abelian N=1 (and K3 x T^2) toroidal orbifolds with these blocks used in the 2026-10-30 study
LIST29 = [
    ("AAA", [(1, 1, 1)]), ("AAA", [(0, 1, 2), (1, 0, 2)]),
    ("BBB", [(1, 2, 1)]), ("BBB", [(0, 2, 2), (2, 0, 2)]), ("BBB", [(0, 2, 2), (1, 0, 3)]), ("BBB", [(0, 1, 3), (1, 0, 3)]),
    ("CAA", [(2, 1, 1)]), ("CAA", [(0, 1, 2), (2, 0, 2)]),
    ("CBA", [(1, 2, 1)]),
    ("CBB", [(0, 2, 2), (3, 0, 2)]), ("CBB", [(3, 3, 3)]), ("CBB", [(0, 1, 3), (3, 0, 2)]),
    ("CCA", [(4, 4, 2)]), ("CCA", [(5, 5, 1)]), ("CCA", [(3, 5, 2)]), ("CCA", [(0, 2, 2), (2, 0, 2)]), ("CCA", [(0, 2, 2), (1, 1, 2)]),
    ("CCB", [(0, 3, 2), (3, 0, 2)]), ("CCB", [(2, 1, 2)]), ("CCB", [(0, 3, 2), (1, 2, 2)]),
    ("CCC", [(4, 4, 4)]), ("CCC", [(0, 3, 3), (3, 0, 3)]), ("CCC", [(5, 5, 2)]), ("CCC", [(5, 4, 3)]), ("CCC", [(0, 2, 4), (2, 0, 4)]),
    ("CCC", [(0, 3, 3), (1, 2, 3)]), ("CCC", [(0, 3, 3), (1, 1, 4)]), ("CCC", [(0, 2, 4), (1, 0, 5)]), ("CCC", [(0, 1, 5), (1, 0, 5)]),
]


def subgroup(gens, ns):
    S = {tuple(0 for _ in ns)}
    frontier = list(S)
    while frontier:
        new = []
        for s in frontier:
            for g in gens:
                t = tuple((x + y) % n for x, y, n in zip(s, g, ns))
                if t not in S:
                    S.add(t)
                    new.append(t)
        frontier = new
    return S


def default_name(blocks: str, gamma_gens: Sequence[Tuple[int, ...]]) -> str:
    ns = [BLOCK[b][0] for b in blocks]
    n = len(subgroup(gamma_gens, ns))
    return f"T_{blocks}_G{n}_" + "_".join("".join(map(str, g)) for g in gamma_gens)


def parse_gens(text: str) -> List[Tuple[int, ...]]:
    """'1,2,1' or '0,2,2;2,0,2' or '122 202' -> list of 3-tuples."""
    out = []
    for part in text.replace(";", " ").split():
        part = part.strip("()")
        v = tuple(int(x) for x in (part.split(",") if "," in part else list(part)))
        if len(v) != 3:
            raise ValueError(f"generator {part!r} needs three entries (one rotation unit per torus)")
        out.append(v)
    return out


def build(blocks: str, gamma_gens: Sequence[Tuple[int, ...]], name: str = None) -> Tuple[str, dict]:
    """(.lgo text, info dict).  blocks: three letters from A, B, C."""
    blocks = blocks.upper()
    if len(blocks) != 3 or any(b not in BLOCK for b in blocks):
        raise ValueError("blocks must be three letters from A (Z3 torus), B (Z4 torus), C (Z6 torus)")
    ns = [BLOCK[b][0] for b in blocks]
    for g in gamma_gens:
        if sum(F(r, n) for r, n in zip(g, ns)).denominator != 1:
            raise ValueError(f"generator {g} does not preserve Omega (sum r_i/n_i must be an integer)")
    name = name or default_name(blocks, gamma_gens)
    Gam = subgroup(gamma_gens, ns)
    GA = [a for a in product(*[range(n) for n in ns])
          if all(sum(F(r * x, n) for r, x, n in zip(g, a, ns)).denominator == 1 for g in Gam)]
    J = tuple(1 for _ in ns)
    Jgrp = subgroup([J], ns)

    def order(g):
        return len(subgroup([g], ns))
    target, gens, cur = set(GA), [], Jgrp
    while cur != target:
        best = None
        for g in sorted(target - cur, key=lambda g: -order(g)):
            H = subgroup(gens + [g], ns)
            free = 1
            for h in gens + [g]:
                free *= order(h)
            if (len(H) == free and len(H & Jgrp) == 1
                    and len(subgroup([J] + gens + [g], ns)) == len(Jgrp) * len(H)):
                best = g
                break
        if best is None:
            raise ValueError("no direct-product complement of <J> in G_A")
        gens.append(best)
        cur = subgroup([J] + gens, ns)
    fields = [(i, q) for i, b in enumerate(blocks) for q in BLOCK[b][1]]
    K = lcm(*[q.denominator for _, q in fields])
    lines = [f'Name = "{name}"', f"KOrder = {K}", "Order = " + " ".join(str(order(g)) for g in gens)]
    extra = []
    for g in gens:
        N = order(g)
        col = []
        for i, q in fields:
            c = F(g[i]) * q * N
            if c.denominator != 1:
                raise ValueError(f"element {g} has a non-integral charge")
            col.append(int(c) % N)
        tot = sum(col)
        if tot % N:
            raise ValueError(f"element {g} has det != 1")
        for j in sorted(range(len(col)), key=lambda j: -col[j])[:tot // N]:
            col[j] -= N
        extra.append(col)
    for f, (i, q) in enumerate(fields):
        cols = [str(int(q * K))] + [str(extra[gi][f]) for gi in range(len(gens))]
        lines.append(f'Field "{f + 1}" ' + " ".join(cols))
    nf = len(fields)
    for f, (i, q) in enumerate(fields):
        e = [0] * nf
        e[f] = int(1 / q)
        lines.append("SPotTerm  1 " + " ".join(map(str, e)))
    info = {"name": name, "blocks": blocks, "gamma_order": len(Gam), "gamma_gens": [list(g) for g in gamma_gens],
            "GA_order": len(GA), "extra_generators": [list(g) for g in gens]}
    return "\n".join(lines) + "\n", info


def write(blocks: str, gamma_gens, outdir=".", name: str = None) -> Path:
    text, info = build(blocks, gamma_gens, name)
    p = Path(outdir) / (info["name"] + ".lgo")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
