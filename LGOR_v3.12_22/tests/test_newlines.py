"""LGOR 3.9 newlines: discfull, permlines, fplines, mfdefects."""
import os
import pytest
from lgor.newlines import mfdefects, zfull
from lgor.newlines.mfdefects import single_factor_table

HERE = os.path.dirname(__file__)


def test_mf_quintic_fibonacci_locus():
    n = [5] * 5
    monos = [(3, 1, 1, 0, 0), (2, 2, 1, 0, 0), (1, 1, 1, 1, 1), (0, 3, 1, 1, 0), (0, 2, 2, 1, 0)]
    v = single_factor_table(n, 0, [(4, 0, 1)], monos)[(4, 0, 1)]
    assert v == [True, False, False, True, True]          # Fib line W_1 survives iff x1-exponent in {0, 3}
    ident = single_factor_table(n, 0, [(0,)], monos)[(0,)]
    assert all(ident)


def test_kernel_and_invariants():
    # Z^2 / span{(2,0),(0,4)} = Z2 x Z4
    assert zfull.invariant_factors([[2, 0], [0, 4]], 2) == [2, 4]
    # x with 2x = 0 mod 4 -> 2Z
    ker = zfull.kernel_mod([[2]], 1, 4)
    assert all(v[0] % 2 == 0 for v in ker) and any(v[0] % 4 == 2 for v in ker)


def _small_model():
    p = os.path.join(HERE, "data_newlines", "G4_4_4_2_2_H4_5.lgor.json")
    if not os.path.exists(p):
        pytest.skip("test spectrum not present")
    from lgor import results
    from lgor.symmetries import SymmetryAnalysis
    return SymmetryAnalysis(results.load(p)), SymmetryAnalysis(results.load(p.replace(".lgor.json", "_mirror.lgor.json")))


def test_discfull_mirror_embedding():
    A, B = _small_model()
    fa, fb = zfull.FullDiscrete(A), zfull.FullDiscrete(B)
    assert fa.invariants == fb.invariants
    ok, _ = zfull.embed_check(fa, fb)
    assert ok


def test_permlines_control():
    from lgor.newlines.permlines import PermLines
    A, _ = _small_model()
    P = PermLines(A)
    assert len(P.normalising) == 4
    ok, _ = P.control()
    assert ok
    txt = P.report()
    assert "(2 3)" in txt and "d=|H:K|" in txt
