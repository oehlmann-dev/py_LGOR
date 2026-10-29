"""V2 features: anomalies, deformations, VEV proposal and flatness.

References: Fermat quintic and its x1..x5 deformation; SU(3)^4 mirror model of
Blaszczyk & Oehlmann, JHEP04(2016)068, Table 6 (B_i deformations)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor.anomalies import AnomalyAnalysis
from lgor.deform import Deformation, parse_terms
from lgor.symmetries import SymmetryAnalysis

EX = HERE.parent / "examples"


def setup(name):
    spec = Model.from_file(EX / name).compute(processes=2)
    return spec, SymmetryAnalysis(spec)


def test_anomalies():
    for name in ("Q1Fermat2.lgo", "Q1Mirror.lgo", "Q1NonFermat.lgo", "A19Z33Mirror20.lgo"):
        assert AnomalyAnalysis(setup(name)[1]).anomaly_free, name


def test_parse():
    m = Model.from_file(EX / "Q1Fermat2.lgo")
    assert [(t.coeff, t.powers) for t in parse_terms(m, "5*x1*x2*x3*x4*x5")] == [(5, [1, 1, 1, 1, 1])]
    assert [(t.coeff, t.powers) for t in parse_terms(m, "x1^2 x2 x3 x4 + 2 x2^3 x3 x5")] == \
        [(1, [2, 1, 1, 1, 0]), (2, [0, 3, 1, 0, 1])]


def test_quintic_deformation():
    spec, sym = setup("Q1Fermat2.lgo")
    d = Deformation(spec, sym, "5 x1 x2 x3 x4 x5", processes=2)
    v = d.vev
    assert len(v.vevs) == 5 and len({y for _, y, _ in v.vevs}) == 1          # equal VEVs of the five s_i
    assert v.d_flat and v.f_flat and v.f_flat_all_orders_by_R
    assert v.pred_group == d.sym.gauge_group == "none"
    assert v.pred_remnant == [5, 5, 5] and v.ws_remnant == [5, 5, 5]
    assert v.R_preserved
    assert d.stats == {"lost_vectors": 4, "lost_chiral": 4}


def test_su3_table6():
    spec, sym = setup("A19Z33Mirror20.lgo")
    for terms, (S, V), group in (("x1 x4 x7", (264, 12), "SU(3) x U(1)^4"),
                                 ("x1 x4 x7 + x2 x5 x8 + x3 x6 x9", (252, 8), "SU(3)")):
        d = Deformation(spec, sym, terms, processes=2)
        assert (d.spec.total.singlets[1], len(d.sym.extra_vectors) - 1) == (S, V)
        assert d.sym.gauge_group == group == d.vev.pred_group
        assert d.vev.d_flat and d.vev.f_flat and d.vev.R_preserved
        assert d.vev.pred_remnant == d.vev.ws_remnant


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
