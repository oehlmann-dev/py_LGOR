"""LGOR 3.14 mfcat: graded matrix-factorisation defects of LG blocks."""
import cmath
from fractions import Fraction as Fr
import numpy as np
from lgor.mfcat import LG, phase_defect, identity_defect, bulk_action, hom_dims, is_simple, parse_ideal, ideal_defect
from lgor.mfcat.lines import a4_demo, realise_block


def test_jacobian_and_identity():
    lg = LG.fermat([6, 3])
    assert lg.jac().mu == 10
    assert np.allclose(bulk_action(identity_defect(lg)), np.eye(10))
    assert LG.fermat([4, 4]).jac().mu == 9


def test_phase_defect_is_det_twisted_symmetry():
    lg = LG.fermat([6, 3])
    ph = [Fr(1, 6), Fr(1, 3)]
    A = bulk_action(phase_defect(lg, ph))
    for k, m in enumerate(lg.jac().basis):
        # RR action of g: g^{-1} on x^m times det(g)^{-1}
        expect = cmath.exp(-2j * cmath.pi * float(sum((a + 1) * p for a, p in zip(m, ph))))
        assert abs(A[k, k] - expect) < 1e-8


def test_swap_defect_and_hom():
    lg = LG.fermat([6, 3])
    lg.names = ["x1", "x4"]
    S = ideal_defect(lg, parse_ideal(lg, ["x1^2 - x4p", "x4 - x1p^2"]), "S")
    assert S.check()[0] and is_simple(S)
    assert hom_dims(identity_defect(lg), S) == {(2, 1): 1, (4, 1): 1, (6, 1): 1, (8, 1): 1}


def test_a4_lines_of_z6II_block():
    for z in range(3):
        out, V, S = a4_demo(z, verbose=False)
        for key in ("S_check", "V_check", "V_action_additive", "V_equals_LGOR_A4_line", "fusion_V2_on_actions",
                    "V_absorbs_1prime", "V_simple"):
            assert out[key], (z, key)
        assert abs(out["vacuum_eigenvalue"] - 3) < 1e-8


def test_realise_block_63():
    lines = realise_block((6, 3), nmax=5, max_terms=4, verbose=False)
    a4 = [l for l in lines if "A4" in l["group"]]
    assert len(a4) == 3 and all(l["realised"] and l["simple"] and l["kind"] == "cone" for l in a4)
    verl = [l for l in lines if "S3" in l["group"] and l.get("realised")]
    assert verl and all(l["simple"] for l in verl)          # L_2 Verlinde = bound state of 1 + g + g^-1


def test_mf_search_finds_s3_mixing_component():
    """general MF search (grading of Koszul charges (4,8), arbitrary entries) finds a rank-2 MF mixing x4 with x1^2,
    and the S3 line sol 1 of gtlines is Cone(M_h -> 1) for a phase twist M_h of it"""
    from lgor.mfcat.search import mf_search, twisted_closure, conjugate, fourier_library
    from lgor.mfcat.lines import gtlines_targets, library, realise_fast, realise_bound_state
    lg, targets = gtlines_targets((6, 3), 5)
    lib = library(lg)
    found = mf_search(lg, (0, 0), (-2, 2), starts=8, seed=1, known=[x[3] for x in lib])
    bi = lg.jac().bindex
    mix = [X for X, A in found if abs(A[bi[(0, 1)], bi[(2, 0)]]) > 1e-6]
    assert mix, "no x4 <-> x1^2 mixing component found"
    Xs = mix + [conjugate(X) for X in mix]
    lib += [("mf", None, Y, A) for Y, A in twisted_closure(lg, Xs, known=[x[3] for x in lib])]
    s3 = [t for t in targets if "S3" in t["group"] and abs(t["target"][bi[(0, 1)], bi[(2, 0)]]) > 1e-6]
    assert len(s3) == 3
    for t in s3:
        sols = realise_fast(t["target"], lib, max_terms=3)
        assert sols
        sols.sort(key=lambda so: (len(so), all(c > 0 for c, _ in so)))
        X, rep = realise_bound_state(sols[0], lib)
        assert rep["check"] and rep["simple"] and abs(rep["vacuum_eigenvalue"] - 2) < 1e-6


def test_fourier_and_rank_one_defects():
    from lgor.mfcat.search import fourier_library, rank_one_defect
    lg = LG.fermat([4, 4])
    for kind, data, X in fourier_library(lg):
        assert X.check()[0] and is_simple(X)
        A = bulk_action(X)
        assert abs(abs(A[4, 4]) - 4) < 1e-8 and np.count_nonzero(np.abs(A) > 1e-8) == 1   # acts only on x1 x2
    P = rank_one_defect(lg, [(0, 1), (0,)])          # Verlinde L1 of x1^4 (x) identity
    assert P.check()[0]
    A = bulk_action(P)
    lam = [1 + (-1j) ** (j + 1) for j in range(3)]    # sum_{a in S} eta^{-a(j+1)}
    for k, m in enumerate(lg.jac().basis):
        assert abs(A[k, k] - lam[m[0]]) < 1e-8
