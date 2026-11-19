"""Driver for ``lgor newlines`` and the single commands (discfull, permlines, fplines, mfdefects)."""
from __future__ import annotations

import os
from pathlib import Path


def _load(path):
    from .. import results
    from ..symmetries import SymmetryAnalysis
    return SymmetryAnalysis(results.load(path))


def json_for(path: str) -> str:
    p = Path(path)
    if p.suffix == ".lgo":
        p = p.with_suffix(".lgor.json")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `lgor run` (and `lgor mirror`) first")
    return str(p)


def mirror_json(path: str):
    p = Path(json_for(path))
    q = p.with_name(p.name.replace(".lgor.json", "_mirror.lgor.json"))
    return str(q) if q.exists() else None


def run(path: str, parts=("discfull", "permlines", "fplines", "mfdefects"), mirror: bool = True,
        max_exponent: int = 12) -> str:
    from . import zfull, permlines, fplines, mfdefects
    js = json_for(path)
    symA = _load(js)
    name = symA.model.name
    out = [f"LGOR 3.12 newlines: {name}", "=" * 100]
    mj = mirror_json(path) if mirror else None
    symB = _load(mj) if mj else None
    if "discfull" in parts:
        out.append(zfull.report(symA, symB, name, (name + "_mirror") if symB else "B"))
    if "permlines" in parts:
        try:
            out.append(permlines.PermLines(symA).report())
        except Exception as e:
            out.append(f"PERMUTATION LINES: not computed ({e}) [missing]\n")
    if "fplines" in parts:
        out.append(fplines.report(symA))
    if "mfdefects" in parts:
        try:
            out.append(mfdefects.report(symA.model, max_exponent=max_exponent))
        except Exception as e:
            out.append(f"MF DEFECTS: not computed ({e}) [missing]\n")
    return "\n".join(out)
