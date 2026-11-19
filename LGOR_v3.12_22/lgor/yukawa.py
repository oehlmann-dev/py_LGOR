"""Trilinear (tree-level Yukawa) superpotential couplings allowed by the symmetries.

A coupling Phi_a Phi_b Phi_c of chiral superfields is listed if it is invariant
under

* E6:            only 27.27.27, 27bar.27bar.27bar, 27.27bar.1, 1.1.1
                 (and 78.78.78, 78.78.1, 27.27bar.78 if chiral 78s are present),
* the extra U(1)s:            the charges add up to zero,
* discrete symmetries and the discrete remnants of broken U(1)s:  zero mod n,
* the R-symmetry:             R(a) + R(b) + R(c) = R(W) = -6  mod M
                              (R = R-charge of the chiral superfields).

These are selection rules: a listed coupling is allowed, its actual value
requires the CFT (for 27^3 at a Fermat point the allowed couplings are exactly
the non-zero chiral-ring products).  Couplings of a field with itself (e.g.
Phi_a^2 Phi_b) are included.  Non-Abelian enhanced gauge groups are only
imposed through their Cartan generators.
"""

from __future__ import annotations

import datetime
from collections import Counter, OrderedDict, defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Tuple

from .symmetries import SymmetryAnalysis, _mod, fmt

CATEGORIES = OrderedDict([
    ("27.27.27", ("27", "27", "27")),
    ("27bar.27bar.27bar", ("27bar", "27bar", "27bar")),
    ("27.27bar.1", ("27", "27bar", "1")),
    ("1.1.1 (singlets only)", ("1", "1", "1")),
    ("78.78.78", ("78", "78", "78")),
    ("78.78.1", ("78", "78", "1")),
    ("27.27bar.78", ("27", "27bar", "78")),
])


class Field:
    def __init__(self, label, rep, info):
        self.label, self.rep, self.info = label, rep, info


def collect_fields(sym: SymmetryAnalysis):
    from .report import build_multiplets
    mults, _ = build_multiplets(sym)
    fields = {"27": [], "27bar": [], "1": [], "78": []}
    for rep in ("27", "27bar"):
        for i, entry in enumerate(mults[rep], 1):
            fields[rep].append(Field(f"{rep}-{i:03d}", rep, entry[0][1]))
    for i, s in enumerate(mults["singlets"], 1):
        fields["1"].append(Field(f"1-{i:03d}", "1", s))
    for i, s in enumerate(mults["78"], 1):
        fields["78"].append(Field(f"78-{i:03d}", "78", s))
    return fields


class YukawaAnalysis:
    def __init__(self, sym: SymmetryAnalysis):
        self.sym = sym
        self.M = sym.M
        self.RW = 2 * sym.R_THETA          # R(W) = -6
        self.fields = collect_fields(sym)
        self.orders = list(sym.model.orders)
        self.rem_orders = [k for _, _, k in sym.remnants]
        self.results = OrderedDict()
        self.warnings: List[str] = []
        for name, reps in CATEGORIES.items():
            if all(self.fields[r] for r in reps):
                self.results[name] = self._category(reps)

    # charge components ----------------------------------------------------------
    def _parts(self, f: Field):
        s = f.info
        if s.R_superfield is None or any(d is None for d in s.discrete) or any(r is None for r in s.remnant):
            return None
        return (tuple(s.u1), s.R_superfield, tuple(s.discrete), tuple(s.remnant))

    def _check(self, ps, level):
        """level 1: U(1)s, 2: + discrete/remnants, 3: + R."""
        u = [sum(x) for x in zip(*[p[0] for p in ps])] if ps[0][0] else []
        if any(u):
            return False
        if level >= 2:
            for j, o in enumerate(self.orders):
                if sum(p[2][j] for p in ps) % o:
                    return False
            for j, o in enumerate(self.rem_orders):
                if sum(p[3][j] for p in ps) % o:
                    return False
        if level >= 3:
            if _mod(sum(p[1] for p in ps) - self.RW, self.M) != 0:
                return False
        return True

    def _category(self, reps):
        """Enumerate allowed couplings by grouping fields with identical charges."""
        groups = []   # per position: {charges: [fields]}
        for r in reps:
            g = defaultdict(list)
            for f in self.fields[r]:
                p = self._parts(f)
                if p is None:
                    self.warnings.append(f"{f.label}: charges not defined, skipped")
                    continue
                g[p].append(f)
            groups.append(g)
        counts = Counter()          # couplings allowed by: 0 E6 only, 1 +U(1), 2 +discrete, 3 +R
        rejected_R = Counter()      # R(a)+R(b)+R(c) mod M of couplings that only R forbids
        allowed: List[Tuple[Field, Field, Field]] = []
        same01 = reps[0] == reps[1]
        same12 = reps[1] == reps[2]
        keys = [list(g.keys()) for g in groups]

        def n_field_combos(c0, c1, c2):
            # number of unordered field triples for this class triple, respecting identical reps
            a, b, c = (len(groups[0][c0]), len(groups[1][c1]), len(groups[2][c2]))
            if same01 and same12:
                cls = Counter([c0, c1, c2])
                tot = 1
                for k, m in cls.items():
                    nk = len(groups[0][k])
                    tot *= _multichoose(nk, m)
                return tot
            if same01:
                if c0 == c1:
                    return _multichoose(a, 2) * c
                return a * b * c
            if same12:
                if c1 == c2:
                    return a * _multichoose(b, 2)
                return a * b * c
            return a * b * c

        # E6-only count without enumerating (combinatorics)
        sizes = [sum(len(v) for v in g.values()) for g in groups]
        if same01 and same12:
            counts[0] = _multichoose(sizes[0], 3)
        elif same01:
            counts[0] = _multichoose(sizes[0], 2) * sizes[2]
        elif same12:
            counts[0] = sizes[0] * _multichoose(sizes[1], 2)
        else:
            counts[0] = sizes[0] * sizes[1] * sizes[2]

        # U(1)-allowed class triples: use a lookup on the third charge to stay fast
        index2 = defaultdict(list)
        for c2 in keys[2]:
            index2[c2[0]].append(c2)
        seen = set()
        for c0, c1 in self._pairs(keys, same01, same12):
            need = tuple(-a - b for a, b in zip(c0[0], c1[0])) if c0[0] else ()
            for c2 in index2.get(need, []):
                trip = (c0, c1, c2)
                # canonical ordering for identical reps to avoid double counting
                if same01 and same12:
                    key = tuple(sorted(trip, key=repr))
                elif same01:
                    key = tuple(sorted(trip[:2], key=repr)) + (trip[2],)
                elif same12:
                    key = (trip[0],) + tuple(sorted(trip[1:], key=repr))
                else:
                    key = trip
                if key in seen:
                    continue
                seen.add(key)
                n = n_field_combos(*key)
                counts[1] += n
                if self._check(key, 2):
                    counts[2] += n
                    if self._check(key, 3):
                        counts[3] += n
                        allowed.extend(self._expand(key, groups, same01, same12))
                    else:
                        rejected_R[_mod(sum(p[1] for p in key), self.M)] += n
        allowed.sort(key=lambda t: tuple(f.label for f in t))
        return {"counts": counts, "allowed": allowed, "rejected_R": rejected_R}

    @staticmethod
    def _pairs(keys, same01, same12):
        if same01:
            for i, j in combinations_with_replacement(range(len(keys[0])), 2):
                yield keys[0][i], keys[0][j]
                if i != j:
                    yield keys[0][j], keys[0][i]
        else:
            for c0 in keys[0]:
                for c1 in keys[1]:
                    yield c0, c1

    @staticmethod
    def _expand(key, groups, same01, same12):
        out = []
        L = [groups[0][key[0]], groups[1][key[1]], groups[2][key[2]]]
        if same01 and same12:
            fields = sorted({f.label: f for c in set(key) for f in groups[0][c]}.values(), key=lambda f: f.label)
            need = Counter(key)
            for t in combinations_with_replacement(fields, 3):
                cnt = Counter(YukawaAnalysis._cls(f, groups[0]) for f in t)
                if cnt == need:
                    out.append(t)
            return out
        if same01:
            fields = sorted({f.label: f for c in set(key[:2]) for f in groups[0][c]}.values(), key=lambda f: f.label)
            need = Counter(key[:2])
            for a, b in combinations_with_replacement(fields, 2):
                if Counter([YukawaAnalysis._cls(a, groups[0]), YukawaAnalysis._cls(b, groups[0])]) == need:
                    for c in L[2]:
                        out.append((a, b, c))
            return out
        if same12:
            fields = sorted({f.label: f for c in set(key[1:]) for f in groups[1][c]}.values(), key=lambda f: f.label)
            need = Counter(key[1:])
            for a in L[0]:
                for b, c in combinations_with_replacement(fields, 2):
                    if Counter([YukawaAnalysis._cls(b, groups[1]), YukawaAnalysis._cls(c, groups[1])]) == need:
                        out.append((a, b, c))
            return out
        for a in L[0]:
            for b in L[1]:
                for c in L[2]:
                    out.append((a, b, c))
        return out

    @staticmethod
    def _cls(f, group):
        for k, v in group.items():
            if f in v:
                return k

    # output -------------------------------------------------------------------
    def text(self, max_list: int = None) -> str:
        sym = self.sym
        m = sym.model
        out = []
        bar = "=" * 100
        out.append(f"Tree-level Yukawa couplings (trilinear superpotential) allowed by the symmetries: {m.name}")
        out.append(f"(LGOR Python, {datetime.date.today()})")
        out.append("")
        out.append("Selection rules imposed:")
        out.append("  E6 invariance (27.27.27, 27bar^3, 27.27bar.1, 1.1.1)")
        if sym.U1:
            out.append("  extra U(1)s: " + ", ".join(f"{n} = {_lin(c, m)}" for n, c in zip(sym.u1_names, sym.U1)))
        for j, o in enumerate(m.orders):
            out.append(f"  discrete Z{o} #{j + 1}")
        for n, c, k in sym.remnants:
            out.append(f"  remnant {n}: {_lin(c, m)} mod {k}")
        out.append(f"  R-symmetry: sum of superfield R-charges = R(W) = {self.RW} mod {self.M}")
        out.append("Field labels (27-001, 1-123, ...) are those of the summary file, which lists their states and charges.")
        out.append("Allowed means allowed by these symmetries; the actual values need the CFT.")
        out.append("")
        out.append("COUPLINGS PER CATEGORY       E6 only     + U(1)s   + discrete      + R  (= allowed)")
        for name, res in self.results.items():
            c = res["counts"]
            out.append(f"  {name:<24}{c[0]:>11}  {c[1]:>10}  {c[2]:>10}  {c[3]:>10}")
        out.append("")
        for name, res in self.results.items():
            allowed = res["allowed"]
            out.append(bar)
            out.append(f"{name}:  {len(allowed)} allowed couplings")
            out.append(bar)
            # summary by charge pattern
            pat = Counter()
            for t in allowed:
                pat[" + ".join(sorted(fmt(f.info.R_superfield) for f in t))] += 1
            if pat:
                out.append("  by R-charges of the three fields: " + ", ".join(f"({k}): {v}" for k, v in sorted(pat.items())))
            rej = res["rejected_R"]
            if rej:
                out.append(f"  {sum(rej.values())} couplings are allowed by all other symmetries but forbidden by R:")
                out.append("    their total R-charge is " + ", ".join(f"{fmt(k)} ({v}x)" for k, v in sorted(rej.items()))
                           + f" instead of {fmt(_mod(self.RW, self.M))} mod {self.M}")
            lst = allowed if max_list is None else allowed[:max_list]
            for t in lst:
                ch = "   ".join(_short(f, sym) for f in t)
                out.append(f"  {t[0].label} {t[1].label} {t[2].label}      [{ch}]")
            if max_list is not None and len(allowed) > max_list:
                out.append(f"  ... {len(allowed) - max_list} more")
            out.append("")
        if self.warnings:
            out.append("WARNINGS")
            out.extend("  " + w for w in sorted(set(self.warnings)))
        return "\n".join(out) + "\n"


def _short(f: Field, sym) -> str:
    s = f.info
    parts = []
    if sym.U1:
        parts.append("(" + ",".join(fmt(x) for x in s.u1) + ")")
    parts.append(f"R={fmt(s.R_superfield)}")
    if s.discrete:
        parts.append("Z=" + ",".join(fmt(x) for x in s.discrete))
    if s.remnant:
        parts.append("rem=" + ",".join(fmt(x) for x in s.remnant))
    return " ".join(parts)


def _lin(c, m) -> str:
    terms = []
    for x, f in zip(c, m.fields):
        x = Fraction(x)
        if x:
            terms.append(("- " if x < 0 else "+ ") + ("" if abs(x) == 1 else f"{abs(x)} ") + f"q{f.name}")
    s = " ".join(terms)
    return s[2:] if s.startswith("+ ") else ("-" + s[2:] if s else "0")


def _multichoose(n: int, k: int) -> int:
    from math import comb
    return comb(n + k - 1, k)


def write_yukawa(sym: SymmetryAnalysis, path) -> Path:
    path = Path(path)
    path.write_text(YukawaAnalysis(sym).text())
    return path
