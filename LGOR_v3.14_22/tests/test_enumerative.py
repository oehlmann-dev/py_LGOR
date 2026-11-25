"""v2.6: lines versus enumerative data (`enumerative`). Calibration: the quintic Fibonacci line acts on the
narrow-sector brane charges as 1 + A + A^-1 (= -A^2 - A^3 modulo the narrow annihilator); P(1,1,2,2,2)[8] has
6 = 2 + 2 h11 narrow sectors and no phase-allowed 4-point Kahler coupling; P(2,2,2,3,9)[18] has one 4-point
Kahler zero forbidden by the k = 4, r = 2 line and by no abelian charge assignment, but forced by the U(1) of factor 4
(one entry avoids it); the quintic 4- and 5-point data calibrate `u1_forced`."""
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor.enumerative import (KahlerCouplings, charge_polynomial, enumerative_text, forbidding_lines, narrow_sectors,
                              u1_forced)

EX = HERE.parent / "examples"


def test_quintic_charges():
    m = Model.from_file(EX / "Q1Fermat2.lgo")
    assert [nu for nu, _ in narrow_sectors(m)] == [1, 2, 3, 4]
    cp = charge_polynomial(m, 0, 2)
    assert cp["integer"] and cp["coeffs"] == [0, 0, -1, -1]          # -A^2 - A^3 = 1 + A + A^4 on A^5 = 1, A != 1


def test_p11222_8():
    m = Model.from_file(EX / "G66222.lgo")
    assert [nu for nu, _ in narrow_sectors(m)] == [1, 2, 3, 5, 6, 7]
    assert charge_polynomial(m, 0, 2)["coeffs"] == [1, 0, 0, -1, 0, -1]
    assert not charge_polynomial(m, 2, 1)["rational"]                  # odd Ising line: needs its dressing
    kc = KahlerCouplings(m)
    assert len(kc.moduli) == 2 and len(kc.three_point()) == 1 and kc.four_point() == []
    assert kc.survival(2, 1)["kept"] == [(4, 4, 0, 0, 0)]


def test_p22239_18():
    m = Model.from_file(EX / "G77749.lgo")
    kc = KahlerCouplings(m)
    an = kc.analyse(3)
    assert an["consistent"] and len(an["rows"]) == 2 and len(an["beyond_phase"]) == 1 and an["genuine"] == []
    q = an["beyond_phase"][0][0]
    assert an["u1"][q]
    assert forbidding_lines(4, [a[3] for a in q], True) == [1, 2]
    assert sorted(kc.survival(3, 2)["kept"]) == [(1, 1, 1, 4, 0), (3, 3, 3, 0, 0)]


def test_text_runs():
    txt = enumerative_text(Model.from_file(EX / "G66222.lgo"), 2, 1, h11=2)
    assert "narrow sectors: 6" in txt and "GENUINE line zeros of genus-0 enumerative data at the Gepner point: 0" in txt


def test_u1_forced_quintic():
    # 4-point: Fibonacci zero of the write-up (worked example, x3 x4 x5^3 avoids factor 1) and a non-zero coupling
    n, H = [5] * 5, [3] * 5
    assert u1_forced(((2, 3, 0, 0, 0), (3, 0, 2, 0, 0), (0, 0, 1, 1, 3), (3, 0, 0, 2, 0)), n, H)
    # 5-point Fibonacci zero with no entry avoiding factor 3: a genuine line zero at the Fermat point
    assert not u1_forced(((0, 0, 3, 1, 1), (0, 2, 3, 0, 0), (1, 0, 3, 0, 1), (1, 0, 3, 0, 1), (1, 1, 1, 2, 0)), n, H)
    # non-zero 5-point coupling is not forced
    assert not u1_forced(((1, 0, 3, 1, 0), (1, 1, 1, 1, 1), (2, 0, 2, 0, 1), (2, 0, 2, 0, 1), (2, 2, 0, 1, 0)), n, H)
