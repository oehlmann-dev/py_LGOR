"""Action of certified block lines on the 27 / 27bar multiplets of an LGOR run (Fermat phase)."""
import cmath
import itertools
import math
from collections import Counter
from fractions import Fraction as Fr

from .fixedpoints import fixed_point_table, orbit_counts
from .groups import SpaceGroup, candidate_actions
from .tft import label_states, certify, line_on_monomials
from .blocks import gauged_generators

def _parse_state(row, nf):
    import re
    sec = row["sector"].strip()
    m = re.match(r"T\((\d+)(?:;\s*([\d,\s]+))?\)", sec)
    k0 = int(m.group(1))
    ts = [int(x) for x in m.group(2).split(",")] if m.group(2) else []
    zm = [0] * nf
    for f, p in re.findall(r"phi(\d+)\[0\](?:\^(\d+))?", row["state"]):
        zm[int(f) - 1] += int(p) if p else 1
    return k0, ts, zm


def twisted_class(ns, K, SG, th):
    """class of a block state that is twisted (fully or partially) in the B-picture: th = twists (0 for a fixed field,
    with its zero-mode power given separately).  Geometric sector from the block charges; q_L != q_R -> untwisted-type
    (class e); sigma^k = 1 -> theta-non-invariant fixed-point combination (the non-trivial orbit, needs a unique one);
    sigma^k != 1 -> the rotation class (must be unique)."""
    qL, qR = th["qL"], th["qR"]
    if qL != qR:
        return SG.cls(SG.A.els[0], 0)
    k = int(qL * K) % K
    if SG.sig[k % SG.m] == SG.e:
        orbs = SG.orbits()
        nontriv = [o for o in orbs if len(SG.classes[o]) > 1]
        return nontriv[0] if len(nontriv) == 1 else None
    cl = {SG.cls(a, k) for a in SG.A.els}
    return cl.pop() if len(cl) == 1 else None


def lgor_line_table(lgo_path, csv_path, block_fields, nmax=4, krawitz_swap=False, cands=None):
    """Action of every certified line of the block on the 27 / 27bar multiplets of an LGOR run (Fermat phase).
    Returns (text, eigenvalue multisets {(line, kind): Counter})."""
    import csv
    from collections import Counter, defaultdict
    from ..model import Model
    M = Model.from_file(lgo_path)
    nf = M.nfields
    gens = gauged_generators(M)
    ns = tuple(M.fermat_exponent(f) for f in block_fields)
    K, tab, ok = fixed_point_table(ns)
    q = [Fr(f.kcharge, M.korder) for f in M.fields]
    rows = [r for r in csv.DictReader(open(csv_path)) if r["section"] in ("C1", "C2")]
    states = []
    for r in rows:
        k0, ts, zm = _parse_state(r, nf)
        nu = []
        for i in range(nf):
            v = Fr(k0) * q[i] / 2
            for j, t in enumerate(ts):
                v += Fr(t) * gens[j + 1][1][i]
            nu.append(v % 1)
        kind = "27" if r["type"].startswith("27 ") else "27bar"
        # 'bulk': every massive-free field twisted (no fixed coordinate; volume-type Kähler moduli); otherwise the
        # state lives on a fixed locus (blow-up / exceptional-divisor type) [inferred classification]
        untw = all(v != 0 for i, v in enumerate(nu) if q[i] != Fr(1, 2))
        if krawitz_swap:
            # A-picture side: narrow block twist t <-> monomial n t - 1, broad monomial a <-> twist (a+1)/n
            for f in block_fields:
                n = M.fermat_exponent(f)
                if nu[f] == 0:
                    nu[f], zm[f] = Fr(zm[f] + 1, n), 0
                else:
                    nu[f], zm[f] = Fr(0), int(nu[f] * n) - 1
        states.append({"kind": kind, "nu": nu, "zm": zm, "sector": r["sector"], "state": r["state"],
                       "untwisted": untw})
    out = ["%s: G2-type block on fields %s, %d multiplets (27: %d, 27bar: %d)" % (
        M.name, [f + 1 for f in block_fields], len(states), sum(s["kind"] == "27" for s in states),
        sum(s["kind"] == "27bar" for s in states))]
    multisets = {}
    done = set()
    pre = cands is not None
    cands = list(cands or [])
    for kstar in ([] if pre else range(1, K)):
        oc = orbit_counts(tab[kstar], K)
        if not oc or max(oc) == 1 or tuple(tab[kstar]) in done:
            continue
        done.add(tuple(tab[kstar]))
        for A, sigma in candidate_actions(tab[kstar], K):
            SG = SpaceGroup(A, sigma)
            labels, tsec, tcls = label_states(ns, K, tab, SG)
            if labels is None:
                continue
            res = certify(ns, SG, labels, tsec, tcls, nmax=nmax)
            cands.append((SG, labels, tsec, tcls, res))
    for SG, labels, tsec, tcls, res in cands:
        irr = [(d, ch) for d, ch in SG.irreps() if d > 1]
        for si, sol in enumerate([x for x in res["solutions"] if not x.get("family")]):
            for d, ch in irr:
                name = "%s line %d (d=%d)" % (SG.name().split("= ")[-1], si, d)
                diag, eig = line_on_monomials(ns, SG, labels, tsec, tcls, sol["mats"], ch)
                cnt = {"27": Counter(), "27bar": Counter(), "27_T1": Counter(), "27bar_T1": Counter(),
                       "meta": (SG.name(), si, d)}
                notes = Counter()
                # group states with a mixing block part by 'rest' to form the eigen-pairs
                pend = defaultdict(list)
                for s in states:
                    t1 = s["untwisted"]
                    th = [s["nu"][f] for f in block_fields]
                    moved = [t != 0 for t in th]
                    if not any(moved):
                        # block untwisted: B-picture monomial = zero modes
                        a = tuple(s["zm"][f] for f in block_fields)
                        lab = labels.get(a)
                        if lab is None:
                            notes["block monomial outside ring"] += 1
                            continue
                        if isinstance(lab, tuple):          # mixing sector: pair by rest
                            rest = (s["kind"], tuple(v for i, v in enumerate(s["nu"]) if i not in block_fields),
                                    tuple(z for i, z in enumerate(s["zm"]) if i not in block_fields), t1)
                            pend[(lab[1], rest)].append(a)
                        else:
                            cnt[s["kind"]][round(ch[lab].real, 6)] += 1
                            if t1:
                                cnt[s["kind"] + "_T1"][round(ch[lab].real, 6)] += 1
                        continue
                    # block (partly) twisted: twisted-sector class from the block charges
                    qL = qR = Fr(0)
                    for t, n, f in zip(th, ns, block_fields):
                        if t == 0:
                            qL += Fr(s["zm"][f], n)
                            qR += Fr(s["zm"][f], n)
                        else:
                            qL += t - Fr(1, n)
                            qR += (1 - t) - Fr(1, n)
                    c = twisted_class(ns, K, SG, {"qL": qL, "qR": qR})
                    if c is None:
                        notes["twisted class ambiguous"] += 1
                        continue
                    cnt[s["kind"]][round(ch[c].real, 6)] += 1
                    if t1:
                        cnt[s["kind"] + "_T1"][round(ch[c].real, 6)] += 1
                for (k, rest), mons in pend.items():
                    sts = tsec[k]
                    if sorted(mons) != sorted(sts):
                        notes["incomplete mixing multiplet"] += 1
                        for a in mons:
                            cnt[rest[0]][("unpaired", a)] += 1
                        continue
                    for o in tcls[k]:
                        cnt[rest[0]][round(ch[o].real, 6)] += 1
                        if rest[-1]:
                            cnt[rest[0] + "_T1"][round(ch[o].real, 6)] += 1
                multisets[name] = cnt
                out.append("  %-26s 27: %s | 27bar: %s%s" % (name, dict(sorted(cnt["27"].items(), key=str)),
                                                           dict(sorted(cnt["27bar"].items(), key=str)),
                                                           ("  notes " + str(dict(notes))) if notes else ""))
    return "\n".join(out), multisets

