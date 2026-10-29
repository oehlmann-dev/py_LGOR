"""Verlinde-line analysis (lgor.tdl) and the general Greene-Plesser mirror (lgor.mirror).

References: quintic Fibonacci lines (3180 vanishing 27^3.1 couplings), (4)^4(1) and (6)^2(2)^3
Gepner models; see MatchingTDLsAcrossMirrorSymmetry/LGO_models/TDL_SUMMARY.md."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor import tdl
from lgor.mirror import general_mirror_model, generator_overlap, mirror_model, orbifold_group

EX = HERE.parent / "examples"


def counts(name):
    m = Model.from_file(EX / name)
    mons = tdl.marginal_monomials(m)
    st = tdl.classify_couplings(m, mons)["stats"]
    zero_weak = sum(v for (w, s, i, nz), v in st.items() if not nz and not w)
    zero_strict_only = sum(v for (w, s, i, nz), v in st.items() if not nz and w)
    nonzero = sum(v for (w, s, i, nz), v in st.items() if nz)
    # a coupling vanishes iff the strict fusion rule forbids it, iff the special factor has an identity
    assert all(s == nz and i == (not nz) for (w, s, i, nz) in st)
    return len(mons), zero_weak, zero_strict_only, nonzero


def test_line_dims():
    assert abs(tdl.line_dim(3, 2) - (1 + 5 ** 0.5) / 2) < 1e-12      # Fibonacci
    assert abs(tdl.line_dim(6, 2) - (1 + 2 ** 0.5)) < 1e-12
    assert abs(tdl.line_dim(2, 1) - 2 ** 0.5) < 1e-12               # Ising-type
    assert [round(tdl.line_dim(4, 2, l)) for l in range(5)] == [2, 0, -1, 0, 2]


def test_quintic():
    assert counts("Q1Fermat2.lgo") == (101, 3180, 0, 6515)
    m = Model.from_file(EX / "Q1Fermat2.lgo")
    loci = tdl.line_loci(m, tdl.marginal_monomials(m))
    assert all(cnt == 50 and ex == [0, 3] for f, k, r, cnt, ex in loci)


def test_gepner_44441_66222():
    assert counts("G44441.lgo") == (103, 1452, 1588, 7845)
    assert counts("G66222.lgo") == (83, 458, 1278, 4047)


def test_labels_well_defined():
    spec = Model.from_file(EX / "Q1Fermat2.lgo").compute(processes=2)
    reps, bad = tdl.massless_labels(spec)
    assert bad == 0
    assert sum(v for (r, p), v in reps.items() if r == "27") == 101


def test_general_mirror():
    for name in ("G44441.lgo", "G66222.lgo"):
        m = Model.from_file(EX / name)
        mm = mirror_model(m)                  # mixed / non-prime exponents -> BHK construction
        H, ab = orbifold_group(mm)
        assert len(H) == ab and generator_overlap(mm) is None
        H0, _ = orbifold_group(m)
        n = [m.fermat_exponent(i) for i in range(m.nfields)]
        size = 1
        for x in n:
            size *= x
        assert len(H) * len(H0) == size      # |H| |H^v| = |prod Z_n|
    # for the quintic the general construction gives the same group as the F_5 one
    q = Model.from_file(EX / "Q1Fermat2.lgo")
    assert orbifold_group(general_mirror_model(q))[0] == orbifold_group(mirror_model(q))[0]


def test_overlap_warning():
    m = Model.from_file(EX / "G66222.lgo")
    m.orders = [8]
    for f, c in zip(m.fields, (1, -1, 0, 0, 0)):
        f.charges = [c]
    m.__post_init__()
    assert "overcounts by 2" in generator_overlap(m)
