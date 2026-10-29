"""v2.5: consistent lines of non-Fermat blocks (`blocklines`). Calibration: CP2_2 = D6 (both Fib x Fib),
level-rank PSU(N)_k = PSU(k)_N, D4 pointed, D8 = PSU(3)_4, Fermat blocks give PSU(2)_k."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor.blocklines import (KNOWN, NEW, NEW_LEVEL, ModelLines, d_extension, identify, load_model, psun,
                             ring_isomorphic, sun)

EX = HERE.parent / "examples"
NF = EX / "nonfermat"


def test_smatrix_unitary():
    for N, k in [(2, 3), (2, 8), (3, 2), (3, 4), (4, 3), (5, 2)]:
        assert sun(N, k).check_unitary(), (N, k)


def test_level_rank():
    for (N, k), (N2, k2) in [((3, 2), (2, 3)), ((4, 2), (2, 4)), ((3, 4), (4, 3)), ((3, 5), (5, 3)), ((4, 5), (5, 4))]:
        assert ring_isomorphic(psun(N, k)[0], psun(N2, k2)[0]), (N, k)


def test_d_series():
    d4 = d_extension(4)[0]
    assert d4.is_pointed() and d4.rank == 3                       # D4 = (1)(1): only Z3
    d6 = d_extension(8)[0]
    fibfib = psun(3, 2)[0].product(psun(2, 3)[0])                  # CP2_2 core
    assert ring_isomorphic(d6, fibfib)                            # CP2_2 = D6
    assert ring_isomorphic(d_extension(12)[0], psun(3, 4)[0])     # D8 = PSU(3)_4
    assert identify(d_extension(16)[0])[1] == NEW                 # D10 (also E7)


def test_identify():
    assert identify(psun(2, 3)[0]) == ("fusion ring of Fib (= PSU(2)_3)", KNOWN)
    assert identify(psun(2, 50)[0])[1] == NEW_LEVEL
    assert identify(psun(3, 4)[0])[1] == NEW


def test_models():
    cases = {"K_CP2_2_CP2_2_CP2_2_3.lgo": [], "K_D8_5_5_5.lgo": ["D8"], "K_E7_E7_E7_1.lgo": ["E7"] * 3,
             "K_CP2_4_CP2_4_5.lgo": ["CP2_4"] * 2, "D4_A1_7.lgo": []}
    for f, new in cases.items():
        ml = ModelLines(Model.from_file(NF / f))
        assert ml.recognised and ml.checks_ok and ml.c3 == 3, f
        assert [x.split(":")[0] for x in ml.new_types()] == new, (f, ml.new_types())
    ml = ModelLines(Model.from_file(EX / "Q1Fermat2.lgo"))
    assert ml.new_types() == [] and "PSU(2)_3=Fib" in ml.oneline()


def test_reads_saved_results(tmp=None):
    import tempfile
    from lgor import compute_model, results
    with tempfile.TemporaryDirectory() as d:
        m = Model.from_file(NF / "K_E7_E7_E7_1.lgo")
        p = Path(d) / "K_E7_E7_E7_1.lgor.json"
        results.save(compute_model(m), p)
        assert ModelLines(load_model(p)).oneline() == ModelLines(m).oneline()
        assert ModelLines(load_model(Path(d))).oneline() == ModelLines(m).oneline()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
    print("all blocklines tests passed")
