"""Fixed-point resolution lines of the Gepner simple-current extension (``lgor fplines``).

[literature: Schellekens-Yankielowicz 1989-90 (fixed points); Fuchs-Schellekens-Schweigert 1996
(S^J matrices); Fuchs-Runkel-Schweigert 2002-04 (defects = A-A bimodules)]
The internal theory of a Fermat LGO is a simple-current extension/orbifold of the product of N=2 minimal
models.  If the current group contains an element J_S that acts on a set S of factors (all with even k)
as the field-identification shift (m, s) -> (m + n, s + 2), then J_S maps the su(2) label l -> k - l on S
and every label with l_f = k_f / 2 for all f in S is a FIXED POINT.  `lgor defects` (section 4) detects
these sets; here the consequences for lines are worked out:

* Each product Verlinde line a = (l_1, .., l_r) with l_f = k_f/2 on S (a "core" line, local with all
  l-independent currents) is not simple in the extended theory: alpha(a) = (a, +) + (a, -), two lines of
  quantum dimension d_a / 2 each [literature, untwisted stabiliser Z_2 assumed].
* Eigenvalues on a bulk field b of the extended theory (FSS formula):
    b not fixed:           lambda_(a,eps)(b)        = lambda_a(b) / 2
    b fixed, resolved eps': lambda_(a,eps)(b, eps') = (lambda_a(b) + eps eps' lambda^J_a(b)) / 2,
  with lambda_a(b) = prod_f S_(a_f b_f)/S_(0 b_f) (su(2)_k) and lambda^J_a(b) = prod_(f in S) S^J/S_(0 b_f)
  x prod_(f not in S) S_(a_f b_f)/S_(0 b_f), |S^J| = 1 for su(2)_k [literature].  So (a, +) and (a, -)
  differ only on fixed fields, by |lambda^J_a(b)| = prod_(f in S) sqrt((k_f + 2)/2) x |rest|  [verified].
* Fixed fields come in resolved pairs (b, +), (b, -) with identical charges: prediction checked here on the
  LGOR spectrum (fixed 27/27bar states should sit in charge classes of multiplicity >= 2) [verified count].
* Which LGOR state is (b, +) and which (b, -) needs the microscopic identification of the resolution label
  [missing]; hence the selection rules of the split lines are not computed.  Note that the full discrete
  group of `discfull` contains Z_2 symmetries that split charge-degenerate states; their relation to the
  resolution label is open [missing].
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from typing import List, Sequence, Tuple

V, LIT, INF, MISS = "[verified]", "[literature]", "[inferred]", "[missing]"


def fixed_sets(model) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """(physical, internal) fixed-point factor sets, from `lgor defects` section 4."""
    from ..defects import DefectAnalysis
    da = DefectAnalysis(model)
    txt = da.section4()
    sets = [tuple(int(x) - 1 for x in re.findall(r"x(\d+)", l)) for l in txt.splitlines() if "labels with l = k/2" in l]
    return sets, txt


def su2_dim(k, l):
    return math.sin(math.pi * (l + 1) / (k + 2)) / math.sin(math.pi / (k + 2))


def report(sym, max_sets: int = 6) -> str:
    from ..tdl import state_labels
    model = sym.model
    ks = [model.korder // f.kcharge - 2 for f in model.fields]
    L = ["FIXED-POINT RESOLUTION LINES (simple-current extension)", "-" * 100]
    try:
        sets, _ = fixed_sets(model)
    except Exception as e:
        return "\n".join(L + [f"  not available for this model ({e}) " + MISS]) + "\n"
    if not sets:
        L.append("  no fixed points of the current group: no resolved lines " + V)
        return "\n".join(L) + "\n"
    L.append(f"  levels {ks}; fixed-point factor sets S (from `defects` section 4) " + V + ": "
             + ", ".join("{" + ",".join(f"x{f + 1}" for f in S) + "}" for S in sets[:max_sets]) + (" ..." if len(sets) > max_sets else ""))
    # charge classes of states
    cls = Counter((s.state.q_plus, s.state.q_minus, tuple(s.q)) for s in sym.states)
    labs = []
    for s in sym.states:
        ls = state_labels(s.sector, s.state)
        labs.append(next(iter(ls)) if len(ls) == 1 else None)
    L.append(f"  {'S':<16} {'#core lines split':>18} {'dims d_a/2 (min..max)':>24}   fixed 27/27bar states: total, in degenerate charge classes")
    for S in sets[:max_sets]:
        free = [f for f in range(len(ks)) if f not in S]
        nl = 1
        for f in free:
            nl *= ks[f] + 1
        dims = []
        for ls in itertools.product(*[range(ks[f] + 1) for f in free]):
            d = 1.0
            for f in S:
                d *= su2_dim(ks[f], ks[f] // 2)
            for f, l in zip(free, ls):
                d *= su2_dim(ks[f], l)
            dims.append(d / 2)
        n27 = deg = 0
        for s, l in zip(sym.states, labs):
            if l is None or s.state.q_minus not in (Fraction(-2), Fraction(2)):
                continue
            if all(l[f] == Fraction(ks[f], 2) for f in S):
                n27 += 1
                if cls[(s.state.q_plus, s.state.q_minus, tuple(s.q))] >= 2:
                    deg += 1
        L.append(f"  {'{' + ','.join(f'x{f + 1}' for f in S) + '}':<16} {nl:>18} {min(dims):>11.4f}..{max(dims):<11.4f}   {n27}, {deg}")
    L.append("  Each listed core line alpha(a) splits into (a,+) + (a,-) " + LIT + "; (a,+) and (a,-) act identically on")
    L.append("  non-fixed fields (half of alpha(a)) and differ on fixed fields by +-lambda^J/2 " + LIT + ".")
    L.append("  Their selection rules need the resolution label of each fixed LGOR state " + MISS + ".")
    return "\n".join(L) + "\n"
