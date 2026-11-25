"""3.12: the A4 lines of the (6,3) block (G2 plane of T2/Z6) do not close under fusion; the
remainder is the predicted pair of non-commuting orbit lines (eigenvalues +-2 sqrt3 i on x1x4, x1^3)."""
import numpy as np
from lgor.gtlines.closure import closure_report


def test_g2_block_closure():
    txt, missing = closure_report((6, 3), nmax=4)
    assert len(missing) == 3
    assert "3.4641j" in txt
    assert txt.count("S3 irrep2 sol") >= 4
