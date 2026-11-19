"""Line-only zeros (new in 2.7; generalises toroidal_candidates/z6II/checks/cand4.py).

Flat n-point functions of untwisted marginal 27s at the Fermat point, F(e_1..e_N, M^k) with insertions of
chosen moduli M (Taylor coefficients of F along the family M).  A tuple counts as a genuine line-only zero
candidate if
  (i)   the line forbids it: in the line's factor the labels fuse to a single channel c (or to channels that
        all give the line the same eigenvalue [inferred]) whose eigenvalue differs from that of the target
        label t;  t = k if mu (the number of n-wraps in that factor, sum = H + mu n) is even, 0 if odd
        [inferred; mu = 0 and mu = 1 reproduce the 27^3.1 rules];
  (ii)  it is allowed by all Fermat phases and the R-charge: sum of exponents = H_i mod n_i, sum mu_i = N - 3;
  (iii) it is not forced by the exact U(1) of a decoupled block (enumerative.u1_forced, which uses the
        symmetry of C_abcd... and special geometry: a (2,2) tie).
Only (iii)-survivors are line-only; they are candidates until the correlator is computed (Yukawa check).
Phase symmetries of the deformed point and permutations are not needed for (ii): Taylor coefficients at the
Fermat point obey the Fermat-point selection rules.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from .enumerative import u1_forced
from .model import Model
from .tdl import channels, line_dim, marginal_monomials


def target_label(k: int, s: int) -> Tuple[int, int]:
    """(mu, target) for label sum s in a factor with level k (n = k + 2)."""
    mu = (s - k) // (k + 2)
    return mu, (k if mu % 2 == 0 else 0)


def line_forbids(k: int, r: int, labels: Sequence[int], target: int) -> bool:
    ch = channels(k, labels)
    if target in ch:
        return False
    vals = {round(line_dim(k, r, c), 9) for c in ch}
    return len(vals) == 1 and abs(next(iter(vals)) - line_dim(k, r, target)) > 1e-9


def phase_ok(entries, n, H) -> Optional[List[int]]:
    N = len(entries)
    m = [sum(e[i] for e in entries) for i in range(len(n))]
    if any((m[i] - H[i]) % n[i] or m[i] < H[i] for i in range(len(n))):
        return None
    mu = [(m[i] - H[i]) // n[i] for i in range(len(n))]
    return mu if sum(mu) == N - 3 else None


def search(model: Model, factor: int, r: int, npoints: int = 4, moduli: Sequence[Sequence[int]] = (),
           max_insert: int = 2, limit: Optional[int] = None) -> dict:
    """factor: 0-based.  Returns counts and the genuine candidates [(entries, insertion counts)]."""
    n = [model.fermat_exponent(i) for i in range(model.nfields)]
    H = [x - 2 for x in n]
    k = n[factor] - 2
    if not 0 < r < k:
        raise ValueError(f"r must be a non-invertible label 0 < r < k = {k}")
    mons = marginal_monomials(model)
    stats = Counter()
    genuine, forced = [], []
    mons = sorted(mons)
    pos = {m: i for i, m in enumerate(mons)}
    nv = len(n)

    def tuples():
        """(base, ks, ins): the last base entry is fixed by the phases (exponents < n_i, so last_i = residue)."""
        for ks in itertools.product(range(max_insert + 1), repeat=len(moduli)):
            ins = [tuple(m) for m, c in zip(moduli, ks) for _ in range(c)]
            sins = [sum(e[i] for e in ins) for i in range(nv)]
            for head in itertools.combinations_with_replacement(range(len(mons)), npoints - 1):
                last = tuple((H[i] - sins[i] - sum(mons[j][i] for j in head)) % n[i] for i in range(nv))
                j = pos.get(last)
                if j is not None and (not head or j >= head[-1]):
                    yield tuple(mons[x] for x in head) + (last,), ks, ins
    for base, ks, ins in tuples():
            ent = list(base) + ins
            mu = phase_ok(ent, n, H)
            if mu is None:
                continue
            stats["phase_allowed"] += 1
            labels = [e[factor] for e in ent]
            _, t = target_label(k, sum(labels))
            if not line_forbids(k, r, labels, t):
                continue
            stats["line_forbidden"] += 1
            if u1_forced(ent, n, H):
                stats["u1_forced"] += 1
                if len(forced) < 20:
                    forced.append((base, ks))
                continue
            stats["genuine"] += 1
            genuine.append((base, ks))
            if limit and len(genuine) >= limit:
                return {"stats": stats, "genuine": genuine, "forced_examples": forced, "truncated": True}
    return {"stats": stats, "genuine": genuine, "forced_examples": forced, "truncated": False}


def _mono(model: Model, e) -> str:
    s = " ".join(f"x{model.fields[f].name}" + (f"^{x}" if x > 1 else "") for f, x in enumerate(e) if x)
    return s or "1"


def lineonly_text(model: Model, factor: int, r: int, npoints: int = 4, moduli: Sequence[Sequence[int]] = (),
                  max_insert: int = 2, show: int = 30, limit: Optional[int] = None) -> str:
    res = search(model, factor, r, npoints, moduli, max_insert, limit)
    st = res["stats"]
    k = model.fermat_exponent(factor) - 2
    out = [f"LINE-ONLY ZEROS of {model.name}: line L{r} of factor x{model.fields[factor].name} (k={k}, "
           f"d={line_dim(k, r):.4f}), {npoints} marginal 27s"
           + (f" + up to {max_insert} insertions of " + ", ".join(_mono(model, m) for m in moduli) if moduli else "")]
    out.append(f"  phase-allowed tuples              {st['phase_allowed']}")
    out.append(f"  of which forbidden by the line    {st['line_forbidden']}")
    out.append(f"  of which U(1)-forced (2,2 tie)    {st['u1_forced']}")
    out.append(f"  genuine line-only candidates      {st['genuine']}" + ("  (search stopped at --limit)" if res["truncated"] else ""))
    for base, ks in res["genuine"][:show]:
        ins = " ".join(f"{_mono(model, m)}^{c}" for m, c in zip(moduli, ks) if c)
        out.append("    <" + ", ".join(_mono(model, e) for e in base) + ">" + (f"  with {ins}" if ins else ""))
    if st["genuine"] == 0 and st["line_forbidden"]:
        out.append("  Every zero this line predicts is already forced by an exact U(1) (no line-only content here).")
    out.append("  Tags: phase test and u1_forced [verified method]; multi-channel and mu>=2 target rule [inferred];")
    out.append("  candidates still need the correlator itself (Yukawa/Saito check) before they count as zeros.")
    return "\n".join(out) + "\n"
