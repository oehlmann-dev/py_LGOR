"""General defect lines (lgor.defects, lgor.rcft, lgor.fsymbols, lgor.fingroups, lgor.zgroups).

Checks against known results: Greene-Plesser mirrors as simple-current invariants, the quintic
self-duality defect (Z5^3, unique), Fibonacci F-symbols, su(2)_k pentagon, Rep(S3) and C(S3, Z2),
the anomaly of the su(2)_k Z2.  Runs with pytest or as a script."""
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor import fingroups as fg
from lgor import fsymbols as fs
from lgor import zgroups as zg
from lgor.defects import DefectAnalysis
from lgor.mirror import mirror_model
from lgor.rcft import Factor

EX = HERE.parent / "examples"


def test_monodromy_formula():
    import random
    rnd = random.Random(0)
    for k in (1, 2, 3, 4, 6):
        F = Factor(k)
        L = F.labels()
        assert len(L) == 2 * F.n * (k + 1)
        for _ in range(200):
            a = rnd.choice(L)
            mJ = rnd.randrange(2 * F.n)
            sJ = 2 * rnd.randrange(2) + (mJ % 2)
            Ja = F.canon(a[0], a[1] + mJ, a[2] + sJ)
            q = (F.h((0, mJ, sJ)) + F.h(a) - F.h(Ja)) % 1
            assert q == (Fr(mJ * a[1], 2 * F.n) - Fr(sJ * a[2], 4)) % 1


def test_lattice_tools():
    S, ok = zg.subgroups([5, 5, 5])
    assert ok and len(S) == 1 + 31 + 31 + 1
    S, ok = zg.subgroups([4, 2])
    assert sorted(zg.subgroup_order(H, [4, 2]) for H in S) == [1, 2, 2, 2, 4, 4, 4, 8]
    b, o = zg.smith_basis([[2, 0], [0, 3]], [[4, 0], [0, 6]], 2)
    assert sorted(o) == [2, 2]


def test_su2_pentagon_and_fib():
    for k in (1, 2, 3, 4, 5):
        assert fs.SU2k(k).pentagon_check(100) < 1e-12
        assert [fs.SU2k(k).fs_indicator(a) for a in range(k + 1)] == [(-1) ** a for a in range(k + 1)]
    F = fs.fib_from_su2_3()
    phi = (1 + 5 ** 0.5) / 2
    assert abs(F[0][0] - 1 / phi) < 1e-12 and abs(abs(F[0][1]) - phi ** -0.5) < 1e-12


def test_pointed_anomaly_su2_z2():
    # Z2 = <L_k> of su(2)_k has spin k/4: anomalous iff k odd
    for k in range(1, 9):
        assert fs.pointed_anomaly([2], [Fr(k, 4) % 1]) == [k % 2 == 0]


def test_group_theoretical_s3():
    G = fg.symmetric_group(3)
    T = fg.GroupTheoretical(G, fg.closure([(1, 0, 2)], 3))
    T.fusion()
    assert sorted(round(d) for d in T.dims()) == [1, 1, 2]                  # C(S3, Z2) has Rep(S3) dims
    T1 = fg.GroupTheoretical(G, [(0, 1, 2)])
    assert sorted(round(d) for d in T1.dims()) == [1] * 6                    # Vec_S3
    if fg.have_numpy():
        rd = fg.ring_character_data(T.fusion(), T.dims())
        assert "S3" in fg.match_rep_groups(*rd)
        chars = fg.irreducible_characters(fg.symmetric_group(4))
        assert sorted(round(c[(0, 1, 2, 3)].real) for c in chars) == [1, 1, 2, 3, 3]


def _keys(name):
    m = Model.from_file(EX / name)
    A, B = DefectAnalysis(m), DefectAnalysis(mirror_model(m))
    PA = A._pairs_lattice(A._base_currents())
    PB = B._pairs_lattice(B._base_currents())
    idp = tuple(range(A.gd.r))
    return A._apply_left(PA, True, idp), A._apply_left(PA, False, idp), tuple(map(tuple, PB))


def test_gp_mirror_is_omega():
    for name in ("Q1Fermat2.lgo", "G44441.lgo", "G66222.lgo"):
        om, ident, mir = _keys(name)
        assert mir == om, name          # Greene-Plesser: Z_mirror(lam, mu) = Z(omega lam, mu)
        assert mir != ident, name
        assert om != ident, name


def test_quintic_selfduality_unique():
    A = DefectAnalysis(Model.from_file(EX / "Q1Fermat2.lgo"), lgor_check=False)
    A.section2()
    found = A.data["selfdual"]
    assert [(k, h) for k, h, _, _ in found] == [("phase", 125)]
    assert all(om for om, _ in found[0][3])          # only with the left charge conjugation


def test_quintic_lines():
    A = DefectAnalysis(Model.from_file(EX / "Q1Fermat2.lgo"), lgor_check=False)
    txt = A.section1()
    assert "single symmetric lines: 31;  alpha^+/alpha^- pairs: 0" in txt       # Fib^5 minus the identity


def test_44441_rep_s3_ring():
    if not fg.have_numpy():
        return
    A = DefectAnalysis(Model.from_file(EX / "G44441.lgo"), lgor_check=False)
    A.section1()
    txt = A.section3()
    assert "rank 3, dims {1.0: 2, 2.0: 1}" in txt and "'S3'" in txt


def test_mirror_quantum_lines():
    # quintic phase Z5^3 <-> mirror quantum Z5^3
    A = DefectAnalysis(Model.from_file(EX / "Q1Mirror.lgo"), lgor_check=False)
    txt = A.section5()
    assert "group modulo the internal currents: trivial" in txt and "dual group of order 125" in txt


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
