"""Geometric heterotic T6/Z_N standard-embedding spectra (lgor/hetorb.py) against literature Hodge numbers."""
from collections import Counter
from lgor import hetorb

EXPECT = {"Z3": (36, 0), "Z4": (31, 7), "Z6-I": (29, 5), "Z6-II": (35, 11), "Z7": (24, 0)}


def _tot(name):
    v, _ = hetorb.parse_twist(name)
    spec = hetorb.spectrum(v)
    c = Counter()
    for lst in spec.values():
        for kind, Q, dyn, osc, m in lst:
            c[kind] += m
    return c, spec


def test_hodge_numbers():
    for name, (h11, h21) in EXPECT.items():
        c, _ = _tot(name)
        assert (c["27"], c["27bar"]) == (h11, h21), name


def test_z3_twisted_su3():
    c, spec = _tot("Z3")
    assert c["1"] == 252
    tw = [(dyn, m) for kind, Q, dyn, osc, m in spec[1] if kind == "1"]
    assert hetorb.su3_multiplets(tw) == Counter({"3bar": 81})


def test_z6II_sectors_and_lgor_singlets():
    c, spec = _tot("Z6-II")
    assert c["1"] == 224  # = LGOR mirror LGO with all three torus moduli on
    per = {k: Counter() for k in spec}
    for k, lst in spec.items():
        for kind, Q, dyn, osc, m in lst:
            per[k][kind] += m
    assert [(per[k]["27"], per[k]["27bar"], per[k]["1"]) for k in range(6)] == \
        [(3, 1, 7), (12, 0, 84), (6, 3, 42), (8, 4, 52), (6, 3, 39), (0, 0, 0)]
