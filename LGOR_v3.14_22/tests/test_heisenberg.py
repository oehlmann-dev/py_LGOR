"""LGOR 3.14: Heisenberg orbit lines and the completed fusion closure."""
import numpy as np
import pytest
from lgor.gtlines.groups import AbGroup
from lgor.gtlines.heisenberg import orbit_operators, invariant_matrix


def test_heisenberg_orbits_z2z2_order3():
    A = AbGroup((2, 2))
    sigma = (0, 3, 1, 2)                     # order-3 automorphism of Z2 x Z2 (theta^3 fixed points of the G2 plane)
    orbs = orbit_operators(A, sigma)
    assert len(orbs) == 5 and all(o["size"] == 3 for o in orbs)
    assert sum(o["commuting"] for o in orbs) == 3          # the three A4-type (Rep(G)) lines
    reps = [(0, 0), (0, 1)]
    for o in orbs:
        R = invariant_matrix(A, sigma, o["L"], reps)
        ev = sorted(np.round(np.abs(np.linalg.eigvals(R)), 6))
        assert ev == ([1.0, 3.0] if o["commuting"] else [pytest.approx(np.sqrt(3))] * 2)


def test_heisenberg_orbits_z3_order2_all_commute():
    A = AbGroup((3,))
    sigma = (0, 2, 1)                        # x -> -x: theta^2 fixed points of the G2 plane (S3 lines)
    orbs = orbit_operators(A, sigma)
    assert len(orbs) == 4 and all(o["commuting"] for o in orbs)


def test_completed_closure_63_closes_with_heisenberg():
    pytest.importorskip("scipy")
    from lgor.gtlines.heisenberg import completed_closure
    txt, opened = completed_closure((6, 3), nmax=5)
    assert opened, "A4 x A4^g should need lines beyond the certified ones"
    assert all(st.startswith("closed") for _, _, st, _ in opened)
    assert all(any("Heis" in n and n.endswith("+i^g('0', '0')") or "Heis" in n for n in dec) for _, _, _, dec in opened)


def test_heisenberg_line_is_mf_bound_state():
    """the non-commuting Heisenberg line H of the (6,3) block (eigenvalues +-i sqrt3 on theta^3 states) is the simple
    bound state Cone(P_{1,2}(x1) P_{1}(x4) -> P_{0,5}(x1) P_{0}(x4)) of rank-one factorisations"""
    from lgor.gtlines.pipeline import analyse_block
    from lgor.gtlines.heisenberg import heisenberg_lines
    from lgor.mfcat.core import LG
    from lgor.mfcat.search import rank_one_defect
    from lgor.mfcat.lines import bound_state
    from lgor.mfcat.hom import bulk_action, is_simple
    rep, _ = analyse_block((6, 3), nmax=5, a_picture=False)
    hl, _ = heisenberg_lines(rep, (6, 3))
    H = [h for h in hl if h["name"].endswith("+i")][0]
    lg = LG.fermat([6, 3])
    bi = lg.jac().bindex
    T = np.zeros((10, 10), complex)
    for q, (sts, Op) in H["ops"].items():
        for a, sa in enumerate(sts):
            for b, sb in enumerate(sts):
                T[bi[sa], bi[sb]] = Op[b, a]
    src = rank_one_defect(lg, [(1, 2), (1,)])
    tgt = rank_one_defect(lg, [(0, 5), (0,)])
    X, dim, chk = bound_state([src], [tgt], seed=0)
    assert chk[0] and dim >= 1 and is_simple(X)
    assert np.allclose(bulk_action(X), T, atol=1e-7)
