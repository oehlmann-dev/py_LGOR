"""mirrorlines: Chiodo FJRW 4-point = A_{r-1} block correlators; A/B existence agree on the Z6-II G2 block."""
import itertools
from fractions import Fraction as F
from lgor import mirrorlines as ml
from lgor.frobenius import frob


def test_chiodo_equals_saito():
    for r in (3, 4, 5, 6):
        for a in itertools.combinations_with_replacement(range(r - 1), 4):
            ths = [F(x + 1, r) for x in a]
            if F(2, r) - sum(ths) == -2:
                assert ml.chiodo4(r, ths) == frob(r).corr(a)


def test_group_closure_and_membership():
    G = ml.group_closure([(F(1, 6), F(1, 3))])
    assert len(G) == 6 and (F(2, 3), F(1, 3)) in G and (F(1, 3), F(1, 3)) not in G
