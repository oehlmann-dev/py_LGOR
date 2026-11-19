"""Families of deformations (new in 2.7): deform a Fermat model by each modulus alone and by all of them
together, and report for every point the massless spectrum, the gauge group, LGOR's discrete symmetries
(modulo gauge) and which Verlinde lines and products of lines commute with the deformation (with a phase
dressing where the eigenvalue is -d, cf. toroidal.phase_dressing).

    python -m lgor family MODEL.lgo --terms "2 x1^4 x4; 3 x2^2 x3^2; 5 x5 x6 x7" [-j N] [--no-singles]
    python -m lgor family MODEL.lgo --torus        (the three torus size moduli of `toroidal`)
"""

from __future__ import annotations

import itertools
from math import gcd
from typing import List, Optional, Sequence, Tuple

from .deform import deformed_model, parse_terms, term_string
from .model import Model
from .tdl import keeps, line_dim


def _L(n):
    L = 1
    for m in n:
        L = L * m // gcd(L, m)
    return L


def phase_dressing(n: Sequence[int], mons: Sequence[Sequence[int]], signs: Sequence[Optional[int]]):
    """A phase symmetry of the Fermat W (exponent vector a, phase exp(2 pi i a_f / n_f)) with eigenvalue -1 on
    the monomials with sign -1 and +1 on those with sign +1 (None: ignored), or None if there is none."""
    L = _L(n)
    rows = [([L // m * e for m, e in zip(n, mon)], L // 2 if s == -1 else 0)
            for mon, s in zip(mons, signs) if s is not None]
    if all(t == 0 for _, t in rows):
        return tuple(0 for _ in n)
    if L % 2:
        return None
    for a in itertools.product(*[range(m) for m in n]):
        if all(sum(x * c for x, c in zip(a, row)) % L == t for row, t in rows):
            return a
    return None


def dress_text(model: Model, a) -> str:
    from fractions import Fraction as Fr
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    return " ".join(f"x{model.fields[f].name}->e^(2pi i {Fr(x, m)})x{model.fields[f].name}"
                    for f, (x, m) in enumerate(zip(a, n)) if x)


def line_rows(model: Model, mons: Sequence[Sequence[int]], max_factors: int = 2) -> List[dict]:
    """Every product of at most `max_factors` non-invertible Verlinde lines (r <= k/2) and its action on each
    monomial: +1 (eigenvalue d), -1 (eigenvalue -d) or None (broken); `dress` = a phase dressing fixing the signs."""
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    K = [m - 2 for m in n]
    cand = [f for f in range(len(n)) if K[f] >= 2]
    rows = []
    for size in range(1, max_factors + 1):
        for fs in itertools.combinations(cand, size):
            for rs in itertools.product(*[range(1, K[f] // 2 + 1) for f in fs]):
                r = [0] * len(n)
                for f, x in zip(fs, rs):
                    r[f] = x
                d = 1.0
                for f, x in zip(fs, rs):
                    d *= line_dim(K[f], x)
                act = [keeps(K, r, mon) for mon in mons]
                dress = None
                if all(a is not None for a in act) and any(a == -1 for a in act):
                    dress = phase_dressing(n, mons, act)
                rows.append({"line": " x ".join(f"L{x}(x{model.fields[f].name})" for f, x in zip(fs, rs)),
                             "factors": fs, "rs": rs, "d": d, "act": act, "dress": dress,
                             "odd": any(x % 2 for x in rs)})
    return rows


def survives(row) -> bool:
    return all(a is not None for a in row["act"]) and (row["dress"] is not None or all(a == 1 for a in row["act"]))


def _counts(spec) -> str:
    t = spec.total
    return (f"(27, 27bar) = ({t.n27[1]}, {t.n27bar[1]})  chiral singlets {t.chiral_singlets}  "
            f"extra vectors {t.extra_vectors}")


def family_text(model: Model, terms_text: Sequence[str], singles: bool = True, processes: int = 1,
                progress=None, max_factors: int = 2, spectra: bool = True) -> str:
    """terms_text: one deformation term per entry (in `deform` syntax)."""
    from .spectrum import compute_model
    from .symmetries import SymmetryAnalysis
    terms = [parse_terms(model, t)[0] for t in terms_text]
    mons = [t.powers for t in terms]
    out = [f"FAMILY of {model.name}", "=" * (10 + len(model.name))]
    out.append("moduli: " + "; ".join(f"M{i + 1} = {term_string(model, t)}" for i, t in enumerate(terms)))
    points = [("Fermat point", [])]
    if singles:
        points += [(f"M{i + 1} alone", [i]) for i in range(len(terms))]
    if len(terms) > 1 or not singles:
        points.append(("all together", list(range(len(terms)))))
    if spectra:
        specF = compute_model(model, processes=processes, progress=progress)
        symF = SymmetryAnalysis(specF)
    out.append("")
    for lab, idx in points:
        out.append(f"--- {lab}" + (": " + " + ".join(f"M{i + 1}" for i in idx) if idx else ""))
        if spectra:
            if idx:
                dm = deformed_model(model, [terms[i] for i in idx], f"{model.name}_" + "".join(f"M{i + 1}" for i in idx))
                spec = compute_model(dm, processes=processes, progress=progress)
                sym = SymmetryAnalysis(spec)
            else:
                spec, sym = specF, symF
            out.append("  spectrum   " + _counts(spec))
            out.append("  gauge      " + sym.gauge_group)
            grp, parts = sym.discrete_group_summary()
            out.append("  discrete   " + grp + (f"   ({'; '.join(x for x in parts if x.startswith('('))})"
                                                if any(x.startswith("(") for x in parts) else ""))
        if idx:
            rows = line_rows(model, [mons[i] for i in idx], max_factors)
            ok = [r for r in rows if survives(r)]
            out.append(f"  lines      {len(ok)} of {len(rows)} products (<= {max_factors} factors) commute")
            for r in ok:
                dr = f", dressed by {dress_text(model, r['dress'])}" if any(a == -1 for a in r["act"]) else ""
                out.append(f"             {r['line']} (d={r['d']:.3f}{dr})" + ("  [odd r: dressing open]" if r["odd"] else ""))
        else:
            out.append("  lines      all Verlinde lines (Gepner point)")
        out.append("")
    out.append("Line survival is the tree-level Cordova-Rizi criterion on the listed moduli (untwisted monomials);")
    out.append("twisted moduli and the fermion-parity dressing of odd-r lines are not checked here (`lines`, `defects`).")
    return "\n".join(out) + "\n"
