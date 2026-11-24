"""grouplines: 2D fixed-point tables of the torus blocks and the A4 / S3 lines of the G2 block."""
from lgor import grouplines as gl


def test_fixed_point_tables():
    K, tab, ok = gl.fixed_point_table((6, 3))
    assert ok and tab[0] == [0, 1, 3, 4, 3, 1] and tab[3] == [4, 1, 1, 4, 1, 1] and tab[2] == [3, 1, 3, 1, 3, 1]
    K, tab, ok = gl.fixed_point_table((4, 4))
    assert ok and tab[0] == [0, 2, 4, 2] and tab[2] == [4, 2, 4, 2]
    K, tab, ok = gl.fixed_point_table((3, 3, 3))
    assert ok and tab == [[0, 3, 3], [3, 3, 3], [3, 3, 3]]


def test_euler_totals():
    assert abs(gl.euler_total((4, 4, 4, 4)) - 24) < 1e-4
    assert abs(gl.euler_total((5, 5, 5, 5, 5)) - 200) < 1e-4


def test_g2_block_lines():
    rep, txt = gl.analyse_block((6, 3), nmax=4)
    groups = {c["group"]: c for c in rep["candidates"]}
    a4 = [c for g, c in groups.items() if "A4" in g][0]
    s3 = [c for g, c in groups.items() if "S3" in g][0]
    assert a4["status"] == "certified" and len(a4["certB"]["solutions"]) == 3
    assert s3["status"] == "certified" and len(s3["certB"]["solutions"]) == 4
    assert "Verlinde L_r, r = (2, 0)" in txt
