"""Basic regression tests (reference numbers from the original C++ LGOR, which agree)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model, compute_model
from lgor import results

EX = HERE.parent / "examples"


def spectrum(name):
    return compute_model(Model.from_file(EX / name))


def test_quintic_fermat():
    s = spectrum("Q1Fermat2.lgo")
    t = s.total
    assert (t.n27[1], t.n27bar[1], t.singlets[1], t.singlets[0], t.n78[0]) == (101, 1, 330, 4, 1)
    assert not s.warnings


def test_quintic_deformed():
    # x1 x2 x3 x4 x5 deformation: the 4 extra U(1)s are Higgsed, eating 4 singlets
    t = spectrum("Q1NonFermat.lgo").total
    assert (t.n27[1], t.n27bar[1], t.singlets[1], t.singlets[0]) == (101, 1, 326, 0)


def test_a19_mirror():
    s = spectrum("A19Z33Mirror20.lgo")
    t = s.total
    assert (t.singlets[0], t.n78[0], t.singlets[1], t.n27[1], t.n27bar[1]) == (32, 1, 324, 36, 0)
    mult = sorted(len(tw) for _, tw in s.grouped())
    assert mult == [1, 8]
    # the nine Cartan gauginos of the paper's appendix B
    sec = s.sector(1, (0, 0))
    cartan = sec.states(q_minus=0, q_plus="-3/2")
    assert len(cartan) == 9
    assert all(sorted(c for c, _ in st.terms) == [-2, 1] for st in cartan)


def test_json_roundtrip(tmp_path=None):
    import tempfile
    s = spectrum("A19Z33Mirror20.lgo")
    d = Path(tmp_path or tempfile.mkdtemp())
    p = results.save(s, d / "x.lgor.json")
    s2 = results.load(p)
    assert s2.total.key() == s.total.key()
    for b1, b2 in zip(s.bunches, s2.bunches):
        for x, y in zip(b1.sectors, b2.sectors):
            assert [st.terms for st in x.states()] == [st.terms for st in y.states()]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
