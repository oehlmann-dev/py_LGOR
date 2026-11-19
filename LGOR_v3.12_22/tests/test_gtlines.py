"""gtlines (LGOR 3.8): group-theoretical lines of arbitrary LGOs on both mirror sides."""
from fractions import Fraction as F
from pathlib import Path

from lgor import gtlines as gt
from lgor.model import Model
from lgor.gtlines import groups

ROOT = Path(__file__).resolve().parents[1]
Z6 = Path("/mnt/project-files/gepner_survey/toroidal_candidates/z6II/runs/Z6II_gepner.lgo")
KS = ROOT / "examples" / "nonfermat" / "K_CP2_2_CP2_2_CP2_2_3.lgo"


def _z6():
    if Z6.exists():
        return Model.from_file(Z6)
    from lgor.model import Field, SuperpotentialTerm
    ks = [2, 3, 3, 4, 4, 4, 4]                  # KOrder 12: q = 1/6, 1/4, 1/4, 1/3 x 4
    ns = [6, 4, 4, 3, 3, 3, 3]
    W = [SuperpotentialTerm(1, [n if j == i else 0 for j in range(7)]) for i, n in enumerate(ns)]
    return Model("Z6II_gepner", 12, [], [Field(str(i + 1), k, []) for i, k in enumerate(ks)], W)


def test_blocks_of_z6II():
    bl, sk = gt.find_blocks(_z6())
    by = {}
    for b in bl:
        by.setdefault(int(b.chat), []).append(b)
    assert len(by[1]) == 9 and len(by[2]) == 9 and len(by[3]) == 1 and not sk
    assert sorted({b.ns for b in by[1]}) == [(3, 3, 3), (4, 4), (6, 3)]


def test_tables_and_charges():
    K, tab, ok = gt.fixed_point_table((6, 3))
    assert ok and tab[3] == [4, 1, 1, 4, 1, 1] and tab[2] == [3, 1, 3, 1, 3, 1]
    K2, tab2, ok2 = gt.fixed_point_table((F(1, 6), F(1, 3)))      # charge input = Fermat input
    assert tab2 == tab
    assert abs(gt.euler_total((5, 5, 5, 5, 5)) - 200) < 1e-4


def test_candidate_guard():
    assert groups.candidate_actions([10, 6, 10, 6], 4) == []      # 6 does not divide 10: no automorphism


def test_g2_block_both_pictures():
    rep, txt = gt.analyse_block((6, 3), nmax=4)
    c = {x["group"]: x for x in rep["candidates"]}
    a4 = [v for k, v in c.items() if "A4" in k][0]
    s3 = [v for k, v in c.items() if "S3" in k][0]
    assert len(a4["certB"]["solutions"]) == 3 and len(s3["certB"]["solutions"]) == 4
    assert None not in a4["matchAB"] and None not in s3["matchAB"]
    assert "Verlinde L_r, r = (2, 0)" in txt
    assert a4["certB"]["line_only"] == {3: 1, 4: 1}


def test_z6II_mirror_agreement_g2():
    txt, rows, _ = gt.analyse_model(_z6(), mirror="auto", nmax=4, chats=(1,), only=[(6, 3)])
    sol = [r for r in rows if "solution" in r]
    assert len(sol) == 4 * 7 and all(r["mirror_agree"] for r in sol)
    assert all(r["B_X"] for r in sol)
    a_x = [r for r in sol if r["A_X"]]
    assert len(a_x) == 4 and all("Verlinde" in r["lines"][0][1] and "not" not in r["lines"][0][1] for r in a_x)


def test_nonfermat_model_runs():
    txt, rows, _ = gt.analyse_model(KS, mirror="auto", nmax=3)
    assert "no mirror" in txt and "non-Fermat" in txt


def test_gtsurvey_small(tmp_path):
    """run + mirror of G8_3_3_2_2 ((4,4) block), then gtsurvey at 3 points: everything mirror-consistent"""
    import shutil
    import subprocess
    import sys
    src = Path("/mnt/project-files/gepner_survey/models/G8_3_3_2_2.lgo")
    if not src.exists():
        return
    shutil.copy(src, tmp_path)
    m = str(tmp_path / "G8_3_3_2_2.lgo")
    for args in (["run", m, "-j", "1"], ["mirror", m, "-j", "1", "--from-json"]):
        subprocess.run([sys.executable, "-m", "lgor"] + args, cwd=ROOT, check=True, capture_output=True)
    from lgor.gtlines.survey import run_survey
    txt = run_survey(str(tmp_path), ["G8_3_3_2_2"], str(tmp_path / "out"), nmax=3, log=lambda *_: None)
    assert "spectra mirror ok" in txt and "spectrum mirror failures 0; TFT mirror failures 0" in txt
