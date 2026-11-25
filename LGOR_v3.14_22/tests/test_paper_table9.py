"""Check the A_1^9 Fermat quotients against Table 9 of Blaszczyk & Oehlmann,
JHEP04(2016)068 (arXiv:1512.03055).

Each entry: expected (S, 27, 27bar, Adj) at q+ = -3/2 (vectors) and at q+ = -1/2
(chiral multiplets), and the Z_3 charge vectors of one or more models with that
spectrum.  Run with ``python -m pytest tests`` or directly with python.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgor import Model, compute_model
from lgor.model import Field, SuperpotentialTerm

TABLE9 = [
    # N = 4
    ((32, 3, 3, 1), (96, 9, 9, 3), [
        [(1,1,1,0,0,0,0,0,0), (0,0,0,1,1,1,0,0,0)],
        [(1,1,1,0,0,0,0,0,0), (0,0,0,1,1,1,0,0,0), (0,0,0,0,0,0,1,-1,0)],
        [(1,1,1,0,0,0,0,0,0), (0,0,0,1,1,1,0,0,0), (0,0,0,0,1,-1,1,-1,0)],
    ]),
    # N = 2
    ((14, 1, 1, 1), (194, 21, 21, 1), [
        [(1,1,1,0,0,0,0,0,0)],
        [(1,2,0,0,0,0,0,0,0), (0,0,0,1,1,1,0,0,0)],
        [(1,2,0,1,2,0,0,0,0), (0,0,0,0,0,0,1,1,1)],
        # [(1,2,0,0,0,0,0,0,0), (0,0,1,1,1,1,1,1,1)],  # as printed: charges sum to 7, not 0 mod 3 (typo in the table)
        [(1,1,1,0,0,0,0,0,0), (0,0,1,1,1,0,0,0,0), (0,0,0,0,0,1,1,1,0)],
        [(1,2,0,0,0,0,0,0,0), (0,0,0,1,2,0,0,0,0), (0,0,0,0,0,0,1,1,1)],
        [(1,2,0,0,0,0,0,0,0), (0,2,1,0,0,0,0,0,0), (0,0,0,1,2,0,0,0,0)],
        [(1,2,0,1,2,0,0,0,0), (0,2,1,1,2,0,0,0,0), (0,0,0,0,0,0,1,2,0)],
    ]),
    ((14, 1, 1, 1), (194, 3, 3, 1), [
        [(1,2,0,1,1,1,0,0,0), (0,0,0,1,1,1,1,1,1)],
        [(1,1,1,0,0,0,0,0,0), (0,0,0,1,1,1,0,0,0), (0,0,1,1,1,0,1,2,0)],
    ]),
    ((14, 1, 1, 1), (194, 9, 9, 1), [
        [(1,1,1,0,0,0,0,0,0), (1,2,0,1,2,0,1,2,0)],
    ]),
    ((32, 1, 1, 1), (248, 9, 9, 1), [
        [(1,1,1,0,0,0,0,0,0), (1,2,0,1,2,0,0,0,0)],
    ]),
    # N = 1, chi = 0
    ((8, 0, 0, 1), (252, 13, 13, 0), [
        [(1,1,1,1,2,0,0,0,0), (0,1,0,0,0,0,1,1,0)],
        [(1,2,0,0,0,0,0,0,0), (0,1,0,0,1,0,0,1,0), (0,0,0,1,1,1,0,0,0)],
    ]),
    ((8, 0, 0, 1), (252, 9, 9, 0), [
        [(1,2,0,1,2,0,0,0,0), (0,1,0,0,1,1,1,2,0), (0,0,0,0,1,0,0,1,1)],
    ]),
    ((14, 0, 0, 1), (270, 9, 9, 0), [
        [(1,1,1,1,2,0,0,0,0), (0,1,0,2,1,0,1,1,0), (1,0,0,0,0,0,0,1,1)],
    ]),
    # N = 1 mirror pairs
    ((8, 0, 0, 1), (252, 84, 0, 0), [
        [],
        [(1,2,0,0,0,0,0,0,0)],
        [(1,2,0,0,0,0,0,0,0), (0,0,0,1,2,0,0,0,0)],
    ]),
    ((8, 0, 0, 1), (252, 0, 84, 0), [
        [(1,1,1,0,0,0,0,0,0), (0,1,0,0,1,1,0,0,0), (0,1,0,0,0,0,0,1,1)],
    ]),
    ((8, 0, 0, 1), (252, 40, 4, 0), [
        [(1,1,1,1,2,0,0,0,0)],
    ]),
    ((8, 0, 0, 1), (252, 4, 40, 0), [
        [(1,1,1,0,0,0,0,0,0), (0,1,0,1,1,0,0,0,0)],
    ]),
    ((14, 0, 0, 1), (270, 36, 0, 0), [
        [(1,2,0,1,2,0,0,0,0)],
    ]),
    ((14, 0, 0, 1), (270, 0, 36, 0), [
        [(1,1,1,0,0,0,0,0,0), (0,1,0,1,1,0,0,0,0), (0,0,0,0,1,0,1,1,0)],
    ]),
    ((32, 0, 0, 1), (324, 36, 0, 0), [
        [(1,2,0,1,2,0,1,2,0), (2,1,2,2,1,0,0,0,1)],
    ]),
    ((32, 0, 0, 1), (324, 0, 36, 0), [
        [(1,0,0,0,2,1,1,1,0), (0,1,0,1,1,1,0,2,0), (0,0,1,2,1,1,1,0,0)],
    ]),
    ((8, 0, 0, 1), (252, 24, 12, 0), [
        [(1,1,1,2,2,2,0,0,0)],
    ]),
    ((8, 0, 0, 1), (252, 12, 24, 0), [
        [(1,2,0,1,2,0,0,0,0), (0,1,0,0,1,1,2,1,0)],
    ]),
]


def normalize(vec):
    """Shift Z_3 charges by multiples of 3 so that they sum to exactly zero."""
    v = [x % 3 for x in vec]
    s = sum(v)
    assert s % 3 == 0, vec
    i = 0
    while s > 0:
        if v[i] > 0:
            v[i] -= 3
            s -= 3
        i = (i + 1) % len(v)
    return v


def a19_model(generators, name="A1^9"):
    gens = [normalize(g) for g in generators]
    fields = [Field(str(i + 1), 1, [g[i] for g in gens]) for i in range(9)]
    W = [SuperpotentialTerm(1, [3 if j == i else 0 for j in range(9)]) for i in range(9)]
    return Model(name, 3, [3] * len(gens), fields, W)


def cases():
    for vec, chi, models in TABLE9:
        for gens in models:
            yield vec, chi, gens


def check(gens, vec, chi, processes=1):
    s = compute_model(a19_model(gens), processes=processes).total
    got_v = (s.singlets[0], s.n27[0], s.n27bar[0], s.n78[0])
    got_c = (s.singlets[1], s.n27[1], s.n27bar[1], s.n78[1])
    return got_v, got_c


def test_table9():
    for vec, chi, gens in cases():
        if [tuple(g) for g in gens] in [[tuple(g) for g in m] for m in TABLE9_MISPRINTS]:
            continue
        got_v, got_c = check(gens, vec, chi)
        assert (got_v, got_c) == (vec, chi), (gens, got_v, got_c)


if __name__ == "__main__":
    import time
    bad = 0
    for vec, chi, gens in cases():
        t = time.time()
        got_v, got_c = check(gens, vec, chi, processes=4)
        ok = (got_v, got_c) == (vec, chi)
        bad += not ok
        print(("ok  " if ok else "FAIL"), f"{time.time() - t:5.1f}s", vec, chi,
              "" if ok else f"got {got_v} {got_c}", gens)
    print("failures:", bad)


# ---------------------------------------------------------------------------
# Independent cross-check: Vafa's Euler number formula for LG orbifolds,
#   chi = (1/|H|) sum_{g,h in H} prod_{i fixed by g and h} (1 - 1/q_i),
# with H generated by J and the discrete Z_3's (q_i = 1/3 -> factor -2), and
# chi = -2 (n27 - n27bar).  It shares no code with LGOR.
#
# These Table 9 entries disagree with both LGOR (C++ and Python, which agree
# with each other) and with Vafa's formula, so the printed table has them in the
# wrong row:
TABLE9_MISPRINTS = [
    [(1,2,0,1,2,0,0,0,0), (0,2,1,1,2,0,0,0,0), (0,0,0,0,0,0,1,2,0)],   # code: 8/252, 24, 12
    [(1,1,1,0,0,0,0,0,0), (1,2,0,1,2,0,1,2,0)],                        # code: 32/248, 9, 9 (rows swapped)
    [(1,1,1,0,0,0,0,0,0), (1,2,0,1,2,0,0,0,0)],                        # code: 14/194, 9, 9 (rows swapped)
    [(1,2,0,1,2,0,0,0,0), (0,1,0,0,1,1,1,2,0), (0,0,0,0,1,0,0,1,1)],   # code: 8/252, 4, 16
    [(1,1,1,1,2,0,0,0,0)],                                             # code: 8/252, 24, 12
    [(1,2,0,1,2,0,0,0,0)],                                             # code: 8/252, 40, 4
    [(1,2,0,1,2,0,1,2,0), (2,1,2,2,1,0,0,0,1)],                        # code: 8/252, 16, 4
    [(1,1,1,2,2,2,0,0,0)],                                             # code: 14/270, 36, 0
]


def vafa_chi(generators):
    import itertools
    from fractions import Fraction
    G = [[1] * 9] + [normalize(g) for g in generators]
    els = {tuple(sum(c * g[i] for c, g in zip(cs, G)) % 3 for i in range(9))
           for cs in itertools.product(range(3), repeat=len(G))}
    tot = sum((-2) ** sum(1 for i in range(9) if g[i] == 0 and h[i] == 0) for g in els for h in els)
    return Fraction(tot, len(els))


def test_vafa_euler_number():
    for vec, chi, gens in cases():
        s = compute_model(a19_model(gens)).total
        assert vafa_chi(gens) == -2 * (s.n27[1] - s.n27bar[1]), gens
