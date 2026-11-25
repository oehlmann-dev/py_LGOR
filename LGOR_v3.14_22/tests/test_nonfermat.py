"""v2.4: non-Fermat superpotentials (Kazama-Suzuki, E7, D blocks). Reference numbers are the Hodge
numbers from the generalised Vafa formula (Jacobian Poincare polynomial of the fixed fields), computed
independently in gepner_survey/new_tdl_candidates/scripts/vafa_gen.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model, compute_model

EX = HERE.parent / "examples"
NF = EX / "nonfermat"


def counts(path):
    t = compute_model(Model.from_file(path)).total
    return t.n27[1], t.n27bar[1]


def test_jacobian_dimensions():
    # Gr(2,6): C(6,2) = 15; E7: 7; D8: 8; CP3_6 = Gr(3,9): 84
    cases = [("K_CP2_4_CP2_4_5.lgo", [0, 1], 15), ("K_E7_E7_E7_1.lgo", [0, 1], 7),
             ("K_D8_5_5_5.lgo", [0, 1], 8), ("K_CP3_6_3_3.lgo", [0, 1, 2], 84)]
    for f, blk, dim in cases:
        m = Model.from_file(NF / f)
        assert blk in m.blocks()
        assert len(m.jacobian_basis(blk)) == dim, f


def test_degenerate_block():
    # x^3 + x y^3 restricted to (x, y) is E7; dropping x^3 leaves x y^3, whose Jacobian ring is infinite
    m = Model.from_file(NF / "K_E7_E7_E7_1.lgo")
    m.superpotential = [t for t in m.superpotential if not (t.powers[0] == 3 and t.powers[1] == 0)]
    assert m.jacobian_basis([0, 1]) is None


def test_fermat_unchanged():
    m = Model.from_file(EX / "Q1Fermat2.lgo")
    assert m.is_fermat
    assert counts(EX / "Q1Fermat2.lgo") == (101, 1)


def test_d4_equals_a1_squared():
    # x^3 + x y^2 is (1)(1) after a linear change of variables: D4 (1)^7 = (1)^9 = (84, 0)
    assert counts(NF / "D4_A1_7.lgo") == (84, 0)


def test_kazama_suzuki_models():
    for f, ref in [("K_CP2_4_CP2_4_5.lgo", (95, 2)), ("K_CP2_2_CP2_2_CP2_2_3.lgo", (85, 1)),
                   ("K_E7_E7_E7_1.lgo", (35, 8)), ("K_D8_5_5_5.lgo", (122, 2)),
                   ("K_CP3_6_3_3.lgo", (75, 3))]:
        assert counts(NF / f) == ref, f


def test_unreduced_korder_warning():
    # the same D8 (5)^3 written with KOrder 14: an extra Z2 orbifold, doubled spectrum, and a warning
    s = compute_model(Model.from_file(NF / "K_D8_5_5_5_unreduced.lgo"))
    assert (s.total.n27[1], s.total.n27bar[1]) == (244, 4)
    assert any("share the factor 2" in w for w in s.warnings)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
    print("all non-Fermat tests passed")
