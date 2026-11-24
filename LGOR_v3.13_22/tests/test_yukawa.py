"""Yukawa selection rules: the allowed 27^3 couplings of the Fermat quintic must be
exactly the non-zero chiral-ring products x^a x^b x^c = prod x_i^3."""
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor.monomial import B
from lgor.symmetries import SymmetryAnalysis
from lgor.yukawa import YukawaAnalysis


def test_quintic_27_cubed():
    mons = [a for a in itertools.product(range(4), repeat=5) if sum(a) == 5]
    ring = {t for t in itertools.combinations_with_replacement(range(len(mons)), 3)
            if tuple(sum(mons[i][k] for i in t) for k in range(5)) == (3, 3, 3, 3, 3)}
    y = YukawaAnalysis(SymmetryAnalysis(Model.from_file(HERE.parent / "examples" / "Q1Fermat2.lgo").compute()))
    idx = {m: i for i, m in enumerate(mons)}

    def mono(f):
        (c, m), = f.info.state.terms
        return idx[tuple(m[B * 5 + i] for i in range(5))]
    allowed = {tuple(sorted(mono(f) for f in t)) for t in y.results["27.27.27"]["allowed"]}
    assert len(ring) == 566 and allowed == ring
    assert len(y.results["27bar.27bar.27bar"]["allowed"]) == 1


if __name__ == "__main__":
    test_quintic_27_cubed()
    print("ok test_quintic_27_cubed")
