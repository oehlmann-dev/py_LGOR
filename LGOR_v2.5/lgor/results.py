"""Saving and loading results as JSON (replaces the .lrt / .lrs files).

The JSON file holds the model, the summary, and for every sector its vacuum
quantum numbers, its oscillator modes and all cohomology states.  Each state is
stored as a list of terms ``[coeff, [[osc, field, power], ...]]`` with osc one of
"phi", "phibar", "psi", "psibar", together with a readable ``text`` version.

Results are written to a temporary file and renamed at the end, so an
interrupted run never leaves a half-written or empty results file behind (the
C++ code did leave one).
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

from .model import Field, Model, SuperpotentialTerm
from .monomial import KIND_NAMES_PHYS
from .sector import Complex, Sector
from .spectrum import Bunch, ModelSpectrum, SpectrumSum

FORMAT = "lgor-results"
VERSION = 1


def _frac(x: Fraction) -> str:
    return str(Fraction(x))


def model_to_dict(m: Model) -> Dict:
    return {
        "name": m.name, "korder": m.korder, "orders": m.orders,
        "fields": [{"name": f.name, "kcharge": f.kcharge, "charges": f.charges} for f in m.fields],
        "superpotential": [{"coeff": t.coeff, "powers": t.powers} for t in m.superpotential],
    }


def model_from_dict(d: Dict) -> Model:
    return Model(d["name"], d["korder"], list(d["orders"]),
                 [Field(f["name"], f["kcharge"], list(f["charges"])) for f in d["fields"]],
                 [SuperpotentialTerm(t["coeff"], list(t["powers"])) for t in d["superpotential"]])


def _mono_to_list(mono, model: Model):
    n = model.nfields
    out = []
    for kind in range(4):
        for i in range(n):
            p = mono[kind * n + i]
            if p:
                out.append([KIND_NAMES_PHYS[kind], model.fields[i].name, p])
    return out


def _mono_from_list(lst, model: Model):
    n = model.nfields
    names = {f.name: i for i, f in enumerate(model.fields)}
    kinds = {k: j for j, k in enumerate(KIND_NAMES_PHYS)}
    v = [0] * (4 * n)
    for kind, name, p in lst:
        v[kinds[kind] * n + names[name]] = p
    return tuple(v)


def sector_to_dict(sec: Sector, charges=None) -> Dict:
    vac = sec.vacuum
    states = []
    for st in sec.states():
        d = {
            "q_minus": _frac(st.q_minus), "q_plus": _frac(st.q_plus),
            "terms": [[c, _mono_to_list(m, sec.model)] for c, m in st.terms],
            "text": st.to_string("modes"),
        }
        info = charges.get(id(st.terms)) if charges else None
        if info is not None:
            d["charges"] = {
                "cartan_q": [_frac(x) for x in info.q] if info.q is not None else None,
                "u1": [_frac(x) for x in info.u1],
                "R_fermion": _frac(info.R) if info.R is not None else None,
                "R_superfield": _frac(info.R_superfield) if info.R_superfield is not None else None,
                "discrete": [None if x is None else _frac(x) for x in info.discrete],
            }
        states.append(d)
    return {
        "k0": sec.k0, "twist": list(sec.twist), "label": sec.label, "ramond": sec.ramond,
        "vacuum": {"energy": _frac(vac["energy"]), "q_minus": _frac(vac["q_minus"]),
                   "q_plus": _frac(vac["q_plus"]), "discrete_charges": list(vac["discrete_charges"])},
        "oscillators": [{k: (_frac(v) if isinstance(v, Fraction) else v) for k, v in row.items()}
                        for row in sec.oscillator_table()],
        "raw_dimensions": sec.raw_dimensions(),
        "cohomology_dimensions": sec.counts(),
        "gravitini_q_plus": [_frac(q) for q in sec.gravitini],
        "warnings": sec.warnings,
        "states": states,
    }


def spectrum_to_dict(spec: ModelSpectrum, sym=None) -> Dict:
    total = spec.total
    charges = {id(info.state.terms): info for info in sym.states} if sym is not None else None
    symd = None
    if sym is not None:
        symd = {
            "gauge_group": sym.gauge_group,
            "u1_generators": {n: [_frac(x) for x in c] for n, c in zip(sym.u1_names, sym.U1)},
            "R_symmetry": {"formula": "Q_R = 3 k0 - 2 q- + n.q mod M", "n": sym.n_R, "M": sym.M,
                           "R_superfield_minus_R_fermion": sym.R_THETA},
            "discrete": [{"order": o, "formula": f"Q = k{j + 1} + n.q mod {o}", "n": nj}
                         for j, (o, nj) in enumerate(zip(spec.model.orders, sym.n_disc))],
            "description": sym.describe(),
        }
    return {
        "format": FORMAT, "version": VERSION,
        "model": model_to_dict(spec.model),
        "summary": {
            "total": total.to_dict(),
            "bunch_groups": [{"multiplicity": len(tw), "twists": [list(t) for t in tw], "spectrum": s.to_dict()}
                             for s, tw in spec.grouped()],
        },
        "symmetries": symd,
        "warnings": spec.warnings + (sym.warnings if sym is not None else []),
        "bunches": [{"twist": list(b.twist), "spectrum": b.sum.to_dict(), "warnings": b.warnings,
                     "sectors": [sector_to_dict(s, charges) for s in b.sectors]} for b in spec.bunches],
    }


def save(spec: ModelSpectrum, path, sym=None) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w") as fh:
        json.dump(spectrum_to_dict(spec, sym), fh, indent=1)
    os.replace(tmp, path)
    return path


def load(path) -> ModelSpectrum:
    d = json.loads(Path(path).read_text())
    if d.get("format") != FORMAT:
        raise ValueError(f"{path} is not an LGOR results file")
    model = model_from_dict(d["model"])
    bunches = []
    for bd in d["bunches"]:
        b = Bunch(model, bd["twist"])
        for sd in bd["sectors"]:
            sec = Sector(model, sd["k0"], sd["twist"])
            sec.compute_twists()
            sec.compute_vacuum()
            sec.warnings = list(sd.get("warnings", []))
            sec.gravitini = [Fraction(q) for q in sd.get("gravitini_q_plus", [])]
            qm_values = sec.q_minus_values()
            for st in sd["states"]:
                a = qm_values.index(Fraction(st["q_minus"]))
                q = int(Fraction(st["q_plus"]) + Fraction(3, 2))
                sec.complexes[a].cohomology[q].append(
                    [(c, _mono_from_list(m, model)) for c, m in st["terms"]])
            sec.computed = True
            b.sectors.append(sec)
        b.assemble()
        bunches.append(b)
    return ModelSpectrum(model, bunches)


def summary_text(spec: ModelSpectrum) -> str:
    return spec.summary() + "\n"
