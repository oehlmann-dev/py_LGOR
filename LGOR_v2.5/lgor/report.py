"""Human-readable output: a summary file and a states file (plus a CSV).

* ``MODEL.summary.txt``  4D spectrum, gauge group, symmetry generators, charge
  tables, per-sector overview, warnings.
* ``MODEL.states.txt``   every massless state as an oscillator polynomial on its
  twisted vacuum, with q-, q+, Cartan charges q^i, U(1) charges, R-charge and
  discrete charges, organised by 4D multiplet.
* ``MODEL.states.csv``   the same, one row per state (for spreadsheets / pandas).

Only q+ = -3/2 (gauginos) and q+ = -1/2 (chiral fermions) are listed; the states
with q+ = +3/2, +1/2 are their CPT conjugates.
"""

from __future__ import annotations

import csv
import datetime
from collections import Counter, OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

from .spectrum import ModelSpectrum
from .symmetries import StateInfo, SymmetryAnalysis, fmt

QM = Fraction
HALF = Fraction(1, 2)


def _poly_W(model) -> str:
    out = []
    for t in model.superpotential:
        mono = " ".join(f"x{model.fields[i].name}" + (f"^{p}" if p > 1 else "")
                        for i, p in enumerate(t.powers) if p)
        out.append((f"{t.coeff} " if t.coeff != 1 else "") + mono)
    return " + ".join(out)


def classify(s: StateInfo, sym: SymmetryAnalysis):
    """(section, label) of a state, or None for CPT conjugates."""
    qp, qm = s.state.q_plus, s.state.q_minus
    if qp == QM(-3, 2):
        if abs(qm) == QM(3, 2):
            return "V1", "E6 gaugino, 78 component " + ("16_-3/2" if qm < 0 else "16bar_+3/2")
        if qm == 0:
            if s.weight_zero:
                return "V2", "Cartan gaugino"
            w = tuple(s.u1)
            for f in sym.factors:
                if w in f.roots:
                    return "V3", f"root of {f.label}"
            return "V3", "root"
        return "V4", "E6-charged gaugino: " + s.rep
    if qp == -HALF:
        if qm == -HALF:
            return "C1", "27 (16_-1/2 component)"
        if qm == HALF:
            return "C2", "27bar (16bar_+1/2 component)"
        if abs(qm) == QM(3, 2):
            return "C3", "chiral 78 (" + ("16_-3/2" if qm < 0 else "16bar_+3/2") + ")"
        if qm == 0:
            return "C4", "E6 singlet"
        return "C5", s.rep
    return None


SECTIONS = OrderedDict([
    ("V1", "VECTOR MULTIPLETS: E6 gauginos (78; the 45+1 part sits on the T(1;0) vacuum)"),
    ("V2", "VECTOR MULTIPLETS: Cartan gauginos of the extra gauge group (q- = 0, zero weight)\n"
           "   (one combination, the trace sum_i, is the E6 Cartan U(1) inside the 78)"),
    ("V3", "VECTOR MULTIPLETS: roots of the non-Abelian extra gauge group"),
    ("V4", "VECTOR MULTIPLETS: E6-charged gauginos (enhancement of E6)"),
    ("C1", "CHIRAL MULTIPLETS: 27-plets (listed by their 16_-1/2 component)"),
    ("C2", "CHIRAL MULTIPLETS: 27bar-plets (listed by their 16bar_+1/2 component)"),
    ("C3", "CHIRAL MULTIPLETS: adjoint 78-plets"),
    ("C4", "CHIRAL MULTIPLETS: E6 singlets (q- = 0)"),
    ("C5", "OTHER COMPONENTS of the 27 / 27bar chiral multiplets (1_-2, 10_+1, 10_-1, 1_+2), for reference"),
])


def _charge_str(s: StateInfo, sym: SymmetryAnalysis, latex=False) -> List[str]:
    lines = []
    if s.q is not None and not s.mixed:
        lines.append("q^i   : (" + ", ".join(fmt(x) for x in s.q) + ")")
    elif s.mixed:
        lines.append("q^i   : not definite (the state mixes monomials with different q^i; W breaks them)")
    if sym.U1:
        lines.append("U(1)  : " + "  ".join(f"{n}={fmt(v)}" for n, v in zip(sym.u1_names, s.u1)))
    if s.R is not None:
        r = f"R     : {fmt(s.R)} (fermion)"
        if s.state.q_plus == -HALF:
            r += f",  {fmt(s.R_superfield)} (chiral superfield)"
        lines.append(r + f"   mod {sym.M}")
    if sym.remnants:
        lines.append("remnant: " + "  ".join(f"{n}={'?' if v is None else fmt(v)}"
                                             for (n, _, _), v in zip(sym.remnants, s.remnant)))
    if s.discrete:
        lines.append("Z_n   : " + "  ".join(
            f"Z{o}#{j + 1}={'?' if d is None else fmt(d)}" for j, (o, d) in enumerate(zip(sym.model.orders, s.discrete))))
    return lines


def write_states(spec: ModelSpectrum, sym: SymmetryAnalysis, path, style: str = "modes") -> Path:
    path = Path(path)
    groups: Dict[str, List] = {k: [] for k in SECTIONS}
    for s in sym.states:
        c = classify(s, sym)
        if c:
            groups[c[0]].append((c[1], s))
    out = [f"States of model {spec.model.name}  (LGOR Python, {datetime.date.today()})",
           "Each state: oscillator polynomial acting on the twisted vacuum |k0; k1..kN>.",
           "Oscillators: phiI[m], phibarI[m], psiI[m], psibarI[m] = modes of field I with mode number m.",
           "q+ = +1/2, +3/2 states (CPT conjugates) are not listed.",
           ""]
    for key, title in SECTIONS.items():
        items = groups[key]
        if not items:
            continue
        out.append("=" * 100)
        out.append(f"{title}:  {len(items)} state(s)")
        out.append("=" * 100)
        for idx, (label, s) in enumerate(items, 1):
            st = s.state
            out.append(f"[{key}-{idx}] {st.sector.label}   q- = {fmt(st.q_minus)}   q+ = {fmt(st.q_plus)}   ({label})")
            out.append("    state : " + st.to_string(style))
            for l in _charge_str(s, sym):
                out.append("    " + l)
        out.append("")
    path.write_text("\n".join(out) + "\n")
    return path


def write_csv(spec: ModelSpectrum, sym: SymmetryAnalysis, path) -> Path:
    path = Path(path)
    names = [f.name for f in spec.model.fields]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "type", "sector", "k0"] + [f"k{j + 1}" for j in range(spec.model.ndiscrete)]
                   + ["q_minus", "q_plus"] + [f"q_{n}" for n in names] + sym.u1_names
                   + ["R_fermion", "R_superfield"] + [f"Z{o}_{j + 1}" for j, o in enumerate(spec.model.orders)]
                   + [n for n, _, _ in sym.remnants] + ["mixed_q", "state", "state_latex"])
        for s in sym.states:
            c = classify(s, sym)
            if not c:
                continue
            st = s.state
            w.writerow([c[0], c[1], st.sector.label, st.sector.k0] + list(st.sector.twist)
                       + [fmt(st.q_minus), fmt(st.q_plus)]
                       + ([fmt(x) for x in s.q] if (s.q and not s.mixed) else [""] * len(names))
                       + [fmt(x) for x in s.u1]
                       + [fmt(s.R) if s.R is not None else "",
                          fmt(s.R_superfield) if (s.R_superfield is not None and st.q_plus == -HALF) else ""]
                       + [("" if d is None else fmt(d)) for d in s.discrete]
                       + [("" if v is None else fmt(v)) for v in s.remnant]
                       + ["yes" if s.mixed else "", st.to_string("modes"), st.to_string("latex")])
    return path


COMPONENTS = {
    "27": (Fraction(-1, 2), (("16_-1/2", Fraction(-1, 2)), ("1_-2", Fraction(-2)), ("10_+1", Fraction(1)))),
    "27bar": (Fraction(1, 2), (("16b_+1/2", Fraction(1, 2)), ("1_+2", Fraction(2)), ("10_-1", Fraction(-1)))),
}


def _key(s: StateInfo):
    return (tuple(s.u1), s.R_superfield, tuple(s.discrete), tuple(s.remnant))


def _sortkey(s: StateInfo):
    return (s.R_superfield if s.R_superfield is not None else -1,
            tuple(-1 if d is None else d for d in s.discrete), tuple(s.u1))


def build_multiplets(sym: SymmetryAnalysis):
    """Assemble E6 multiplets from their components (matched by charges).

    Returns (multiplets, problems) with multiplets = {"27": [[(label, StateInfo), ...], ...], ...}.
    """
    out, problems = {}, []
    chiral = [s for s in sym.states if s.state.q_plus == -HALF]
    for rep, (_, comps) in COMPONENTS.items():
        pools = {}
        for label, qm in comps:
            pool = {}
            for s in sorted((x for x in chiral if x.state.q_minus == qm), key=_sortkey):
                pool.setdefault(_key(s), []).append(s)
            pools[label] = pool
        primary_label = comps[0][0]
        mults = []
        for s in sorted((x for x in chiral if x.state.q_minus == comps[0][1]), key=_sortkey):
            entry = [(primary_label, s)]
            for label, _ in comps[1:]:
                lst = pools[label].get(_key(s), [])
                if lst:
                    entry.append((label, lst.pop(0)))
                else:
                    problems.append(f"no {label} component with the charges of a {rep} in {s.sector.label}")
            mults.append(entry)
        for label, _ in comps[1:]:
            left = sum(len(v) for v in pools[label].values())
            if left:
                problems.append(f"{left} {label} components of {rep}-plets left unmatched")
        out[rep] = mults
    out["singlets"] = sorted((s for s in chiral if s.state.q_minus == 0), key=_sortkey)
    out["78"] = sorted((s for s in chiral if s.state.q_minus == Fraction(-3, 2)), key=_sortkey)
    return out, problems


def _charges_cols(s: StateInfo, sym: SymmetryAnalysis, width: int, superfield=True) -> str:
    u1 = "(" + ", ".join(f"{fmt(x):>{width}}" for x in s.u1) + ")" if sym.U1 else ""
    R = s.R_superfield if superfield else s.R
    r = f"R={fmt(R):>3}" if R is not None else ""
    multi = len(sym.model.orders) > 1
    d = " ".join(f"Z{o}{('#' + str(j + 1)) if multi else ''}={'?' if v is None else fmt(v)}"
                 for j, (o, v) in enumerate(zip(sym.model.orders, s.discrete)))
    rem = ""
    if sym.remnants:
        rem = "(" + ",".join("?" if v is None else fmt(v) for v in s.remnant) + ")"
    return "  ".join(x for x in (u1, r, d, rem) if x)


def summary_text(spec: ModelSpectrum, sym: SymmetryAnalysis, style: str = "modes") -> str:
    m = spec.model
    t = spec.total
    mults, problems = build_multiplets(sym)
    width = max([len(fmt(x)) for s in sym.states for x in s.u1] + [1])
    out = []
    bar = "=" * 100
    out.append(f"LGOR spectrum summary for model {m.name}")
    out.append(f"(LGOR Python, {datetime.date.today()}{', input ' + Path(m.filename).name if m.filename else ''})")
    out.append("")
    out.append("MODEL")
    out.append(f"  KOrder = {m.korder}, {m.nfields} fields, discrete symmetries: "
               + (", ".join(f"Z{o}" for o in m.orders) if m.orders else "none"))
    for f in m.fields:
        out.append(f"    field {f.name:<4} alpha = {fmt(Fraction(f.kcharge, m.korder)):<5}"
                   + ("  discrete charges " + " ".join(f"{c:>2}" for c in f.charges) if f.charges else ""))
    out.append(f"  W = {_poly_W(m)}")
    out.append(f"  {2 * m.korder * len(spec.bunches)} twisted sectors T(k0; k) in {len(spec.bunches)} bunch(es)")
    out.append("")

    nvec = len(sym.extra_vectors) - (1 if sym.untwisted is not None else 0)
    out.append("4D SPECTRUM  (N=1 multiplets; gauge group E6 x E8 x extra)")
    out.append(f"  vector multiplets : E6 adjoint 78   {t.n78[0]}")
    out.append(f"                      extra vectors    {nvec}   -> extra gauge group {sym.gauge_group}")
    if sym.remnants:
        out.append(f"  discrete remnants of the broken U(1)s: " + " x ".join(f"Z{k}" for _, _, k in sym.remnants))
    if sym.charged_vectors:
        out.append(f"                      E6-charged       27: {t.n27[0]}, 27bar: {t.n27bar[0]}  (E6 is enhanced)")
    out.append(f"  chiral multiplets : 27              {t.n27[1]}")
    out.append(f"                      27bar           {t.n27bar[1]}")
    if t.n78[1]:
        out.append(f"                      78              {t.n78[1]}")
    out.append(f"                      E6 singlets     {t.singlets[1]}")
    out.append(f"  n27 - n27bar = {t.n27[1] - t.n27bar[1]}   (Calabi-Yau: h21 = {t.n27[1]}, h11 = {t.n27bar[1]})")
    out.append("")

    out.append("CHARGE GENERATORS")
    out.extend("  " + l for l in sym.describe().splitlines())
    out.append("  The charges below are the values of these generators. All components of an E6 multiplet")
    out.append("  carry the same charges (checked). R = R-charge of the chiral superfield (fermion R - 3);")
    out.append("  for the vector multiplets R = R-charge of the gaugino (= 3).")
    out.append("")
    cols = "U(1) charges (" + ", ".join(sym.u1_names) + ")" if sym.U1 else "no extra U(1)s"
    cols += ", R"
    if m.orders:
        cols += ", discrete charges"
    if sym.remnants:
        cols += ", (" + ", ".join(n for n, _, _ in sym.remnants) + ") remnant charges"

    for rep in ("27", "27bar"):
        out.append(bar)
        out.append(f"{rep}-PLETS: {len(mults[rep])}      [{cols}]")
        out.append("   components: " + ", ".join(l for l, _ in COMPONENTS[rep][1]))
        out.append(bar)
        for i, entry in enumerate(mults[rep], 1):
            out.append(f"{rep}-{i:03d}   {_charges_cols(entry[0][1], sym, width)}")
            for label, s in entry:
                out.append(f"          {label:<9} {s.sector.label:<14} {s.state.to_string(style)}")
        out.append("")

    out.append(bar)
    out.append(f"E6 SINGLETS (chiral): {len(mults['singlets'])}      [{cols}]")
    out.append(bar)
    for i, s in enumerate(mults["singlets"], 1):
        out.append(f"1-{i:03d}   {_charges_cols(s, sym, width)}   {s.sector.label:<14} {s.state.to_string(style)}")
    out.append("")
    if mults["78"]:
        out.append(bar)
        out.append(f"CHIRAL 78-PLETS: {len(mults['78'])} (16_-3/2 component)")
        out.append(bar)
        for i, s in enumerate(mults["78"], 1):
            out.append(f"78-{i:03d}  {_charges_cols(s, sym, width)}   {s.sector.label:<14} {s.state.to_string(style)}")
        out.append("")

    out.append(bar)
    out.append(f"VECTOR MULTIPLETS (gauginos, q+ = -3/2)      [{cols}]")
    out.append(bar)
    vec = [s for s in sym.states if s.state.q_plus == Fraction(-3, 2)]
    for i, s in enumerate(sorted(vec, key=lambda x: (x.state.q_minus != 0, x.weight_zero is False, tuple(x.u1))), 1):
        c = classify(s, sym)
        out.append(f"V-{i:03d}   {_charges_cols(s, sym, width, superfield=False)}   {c[1]:<36} "
                   f"{s.sector.label:<14} {s.state.to_string(style)}")
    out.append("   (the E6 gauginos of the 45 + 1 sit on the T(1;0) vacuum; the trace of the Cartan gauginos")
    out.append("    above is the E6 Cartan U(1).)")
    out.append("")

    try:
        from .anomalies import AnomalyAnalysis
        out.append(AnomalyAnalysis(sym).text())
    except Exception as err:          # never let the anomaly check break the summary
        out.append(f"ANOMALIES: could not be computed ({err})\n")
    out.append("TWISTED SECTORS  (vacuum quantum numbers; number of states at q+ = -3/2 | -1/2 per q-)")
    out.append(f"  {'sector':<20}{'type':<6}{'E_vac':>7}{'q-':>7}{'q+':>7}   states")
    for b in spec.bunches:
        for sec in b.sectors:
            v = sec.vacuum
            items = [f"q-={fmt(q)}: {row[0]}|{row[1]}" for q, row in zip(sec.q_minus_values(), sec.counts())
                     if row[0] or row[1]]
            if not items and not sec.gravitini:
                continue
            extra = "  + gravitino/E8xSO(10)" if sec.gravitini else ""
            out.append(f"  {sec.label:<20}{('R' if sec.ramond else 'NS'):<6}{fmt(v['energy']):>7}"
                       f"{fmt(v['q_minus']):>7}{fmt(v['q_plus']):>7}   " + "  ".join(items) + extra)
    out.append("")
    w = spec.warnings + sym.warnings + problems
    out.append("WARNINGS" + ("" if w else ": none"))
    out.extend("  " + x for x in w)
    return "\n".join(out) + "\n"


def write_all(spec: ModelSpectrum, sym: SymmetryAnalysis, stem) -> List[Path]:
    stem = Path(stem)
    p1 = stem.with_name(stem.name + ".summary.txt")
    p1.write_text(summary_text(spec, sym))
    p2 = write_states(spec, sym, stem.with_name(stem.name + ".states.txt"))
    p3 = write_csv(spec, sym, stem.with_name(stem.name + ".states.csv"))
    return [p1, p2, p3]
