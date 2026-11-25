"""v2.7: checkpoints and sector parallelism, toroidal phases, orbifold builder, families, permutations,
line-only search, batch.  Calibration: the 29 T^6/Gamma LGOs of the 2026-10-30 study (Gamma names, Z6-II
twist), the Z6-II Gepner model G4_2_2_1_1_1_1 (L2 survives, single L1 broken, 27^3.1 with T insertions all
U(1)-forced) and the quintic 5-point line-only zeros (269 for the line of factor 1)."""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lgor import Model
from lgor import orbifold, toroidal, perms, lineonly, family
from lgor.spectrum import compute_model

EX = HERE.parent / "examples"


def _z6ii():
    text, _ = orbifold.build("CBA", [(1, 2, 1)])
    return Model.from_string(text)


def test_checkpoint_resume_parallel():
    m = _z6ii()
    serial = compute_model(m).total.key()
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck"
        par = compute_model(m, processes=2, checkpoint=ck, log=Path(d) / "log").total.key()
        assert len(list(ck.glob("*.pkl"))) == 2 * m.korder
        again = compute_model(m, checkpoint=ck).total.key()      # all sectors from the checkpoint
    assert serial == par == again


def test_gamma_names():
    expect = {"T_CBA_G6_121": "Z6-II", "T_CCA_G6_551": "Z6-I", "T_CCC_G12_033_123": "Z2xZ6",
              "T_CCC_G12_033_114": "Z2xZ6'", "T_BBB_G16_013_103": "Z4xZ4", "T_AAA_G3_111": "Z3"}
    for blocks, gens in orbifold.LIST29:
        text, info = orbifold.build(blocks, gens)
        ph, _ = toroidal.detect(Model.from_string(text))
        assert ph is not None and ph.side == "model"
        assert len(ph.Gamma) == info["gamma_order"]
        if info["name"] in expect:
            assert ph.name() == expect[info["name"]], info["name"]


def test_z6ii_twist_and_lines():
    ph, _ = toroidal.detect(_z6ii())
    g = ph.gamma_generators()[0]
    assert sorted(abs(x) for x in ph.twist(g)) == sorted([toroidal.Fr(1, 6), toroidal.Fr(1, 2), toroidal.Fr(1, 3)])
    rows = {r["line"]: family.survives(r) for r in toroidal.line_table(ph)}
    assert rows["L2(x1)"] and not rows["L1(x1)"]            # Rep(S3) line survives, sqrt3 line broken
    assert rows["L1(x4) x L1(x5)"]


def test_mirror_side_detection():
    from lgor.mirror import mirror_model
    mir = mirror_model(_z6ii())
    ph, note = toroidal.detect(mir)
    assert ph is not None and ph.side == "mirror" and ph.name() == "Z6-II"


def test_not_toroidal():
    ph, note = toroidal.detect(Model.from_file(EX / "Q1Fermat2.lgo"))
    assert ph is None and "torus blocks" in note


def test_perms_quintic_s5():
    p = perms.normalizer_perms(Model.from_file(EX / "Q1Fermat2.lgo"))
    assert len(p) == 120 and perms.structure(p)["guess"] == "S5"


def test_lineonly_z6ii():
    m = _z6ii()
    ph, _ = toroidal.detect(m)
    res = lineonly.search(m, 0, 2, 3, [mm for _, mm in ph.moduli()], 1)
    assert res["stats"]["line_forbidden"] == res["stats"]["u1_forced"] > 0 and res["stats"]["genuine"] == 0


def test_orbifold_cli_and_toroidal_cli():
    with tempfile.TemporaryDirectory() as d:
        env = {"PYTHONPATH": str(HERE.parent)}
        import os
        env = dict(os.environ, **env)
        subprocess.run([sys.executable, "-m", "lgor", "orbifold", "CBA", "121", "-o", d], check=True, env=env,
                       capture_output=True)
        lgo = Path(d) / "T_CBA_G6_121.lgo"
        out = subprocess.run([sys.executable, "-m", "lgor", "toroidal", str(lgo), "-o", d], check=True, env=env,
                             capture_output=True, text=True).stdout
        assert "Gamma = Z6-II" in out and "SURVIVES" in out


def test_orbifold_generators_independent():
    # 2.7.1: the extra generators must generate a group of order prod(orders), else LGOR over-counts sectors
    # (CCC with Gamma = Z3 = <(4,4,4)> came out as 108 27s instead of 36)
    text, info = orbifold.build("CCC", [(4, 4, 4)])
    assert "Order = 6 2" in text and info["GA_order"] == 72
    assert compute_model(Model.from_string(text)).total.n27[1] == 36
