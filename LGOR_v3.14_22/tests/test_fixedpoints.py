"""3.13 `fixedpoints` on the Z6-II Gepner model (4)(2)^2(1)^4 (bundled LGOR 3.12.2 run, tests/data_313)."""
import os
from collections import Counter
from lgor import fixedpoints as fp

D = os.path.join(os.path.dirname(__file__), "data_313")
PRE = os.path.join(D, "Z6II_gepner")
_cache = {}


def setup():
    if "S" not in _cache:
        _cache["S"] = fp.Setup(PRE, fp.parse_blocks("0,3;1,2;4,5,6"), fp.parse_twist("1/6,-1/2,1/3"),
                               fp.read_r_rule(os.path.join(D, "Z6II_T2.summary.txt")))
    return _cache["S"]


def test_sectors():
    S = setup()
    c = Counter(r["k"] for r in S.rows)
    assert [c[k] for k in range(6)] == [18, 96, 51, 64, 48, 0]
    assert S.types == ["C", "B", "A"]


def test_rep_s3_per_orbit():
    S = setup()
    lg, orb = fp.rep_s3(S)
    for k in (2, 4):
        for rep in ("27", "27bar", "1"):
            assert lg[(k, rep, "e (origin)")] == sum(v for (kk, rr, c, p), v in orb.items()
                                                    if kk == k and rr == rep and p == "origin")
            assert lg[(k, rep, "3-cycle (pair)")] == sum(v for (kk, rr, c, p), v in orb.items()
                                                        if kk == k and rr == rep and p == "orbit")
    assert lg[(2, "1", "e (origin)")] == 15 and lg[(4, "1", "3-cycle (pair)")] == 27


def test_a4_split():
    S = setup()
    lg, orb = fp.a4_split(S)
    for key in set(lg) | set(orb):
        assert lg[key] == orb[key], key
    o = sum(v for k, v in lg.items() if k[1] == "1" and k[3] == "origin")
    t = sum(v for k, v in lg.items() if k[1] == "1" and k[3] == "triple")
    assert (o, t) == (12, 40)


def test_g_label():
    exc, split = fp.label_test(setup())
    assert exc == 0
    assert split == Counter({(2, 2): 22, (6, 6): 6})


def test_r_quartets_and_fit():
    S = setup()
    Q = fp.r_quartets(S)
    assert len(Q) == 28 and all(v is not None for v in Q.values())
    sols, rep = fp.r_fit(S)
    assert {s[0] for s in sols} == {(0, 3, 1)}
    assert all(s[4] == fp.F(1, 2) for s in sols)
