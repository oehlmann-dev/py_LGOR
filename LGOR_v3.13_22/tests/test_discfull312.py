"""3.12: discfull finds the SU(3)-plane Z3 of Z6-II at finite tori (3.9 found only Z2)."""
import os, pytest
from lgor.newlines.runner import _load
from lgor.newlines.zfull import FullDiscrete
J = os.path.join(os.path.dirname(__file__), "data_312", "Z6II_torus.lgor.json")


@pytest.mark.skipif(not os.path.exists(J), reason="data missing")
def test_z6II_torus_z3():
    sym = _load(J)
    assert FullDiscrete(sym, reps_only=False).group_name() == "Z2"
    assert FullDiscrete(sym, reps_only=True).group_name() == "Z6"
