"""Symmetry pipeline: gauge group, R-symmetry and discrete charges.

Reference: Blaszczyk & Oehlmann, JHEP04(2016)068, section 4.1 (SU(3)^4 mirror LGO,
Tables 4 and 5) and the Fermat quintic.
"""
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor.symmetries import SymmetryAnalysis

EX = HERE.parent / "examples"
M32, M12 = Fraction(-3, 2), Fraction(-1, 2)


def analysis(name):
    return SymmetryAnalysis(Model.from_file(EX / name).compute())


def test_quintic():
    a = analysis("Q1Fermat2.lgo")
    assert a.gauge_group == "U(1)^4" and not a.warnings
    assert a.n_R == [0] * 5                          # Q_R = 3 k0 - 2 q-  mod 30
    R27 = {s.R_superfield for s in a.states if s.state.q_plus == M12 and s.state.q_minus == M12}
    assert R27 == {Fraction(28)}                     # R(27) = -2: 27^3 has R(W) = -6
    assert all(s.R == 3 for s in a.states if s.state.q_plus == M32)


def test_su3_4_mirror():
    a = analysis("A19Z33Mirror20.lgo")
    assert a.gauge_group == "SU(3) x SU(3) x SU(3) x SU(3)" and not a.warnings
    g = [s for s in a.states if s.state.q_plus == M32]
    assert all(s.R == 3 for s in g) and all(d == 0 for s in g for d in s.discrete)
    # Tables 4 and 5 list R in the convention R(Phi) = R(psi) + 3: per SU(3) factor
    # the 27-triplets have (6, 6, 0) and the singlets 216 x 6, 108 x 0.
    paper = lambda s: (s.R + 3) % 18
    c27 = Counter(paper(s) for s in a.states if s.state.q_plus == M12 and s.state.q_minus == M12)
    c1 = Counter(paper(s) for s in a.states if s.state.q_plus == M12 and s.state.q_minus == 0)
    assert c27 == {6: 24, 0: 12}
    assert c1 == {6: 216, 0: 108}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
