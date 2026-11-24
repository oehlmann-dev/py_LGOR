"""Geometric heterotic T^6/Z_N orbifolds with standard embedding (V = v): massless spectrum by twisted sector.

This is the target-space side of the toroidal-orbifold comparisons (`toroidal`, `family`): the
geometric orbifold computed directly in the E8 x E8 lattice language, to compare with the LGO/Gepner
phase.  Conventions (standard, e.g. Ibanez-Uranga ch. 8, Vaudrevange's thesis):

  twisted sector theta^k, shifted momenta p + kV (p in E8, first E8 only; the second E8 is untouched),
  q + kv (q in the SO(8) spinor class with sum(q) odd, q0 = -1/2: left-chiral fermions),
  masses   (p+kV)^2/2 + N_osc - 1 + dc = 0,   (q+kv)^2/2 - 1/2 + dc = 0,
           dc = sum_i w_i(1-w_i)/2,  w_i = {k v_i},
  phase    Delta = exp 2 pi i [ (p+kV).V - (q+kv).v + v.(N - N*) - k(V^2-v^2)/2 ],
  degeneracy D = (1/N) sum_l chi(theta^k, theta^l) Delta^l,
  chi(theta^k, theta^l) = prod over planes NOT fixed by theta^k of 4 sin^2(pi gcd(k,l,N) v_i)
  (Lefschetz count of common fixed points; exact for factorisable lattices with the rotation
  symmetry in each plane [literature]; for non-factorisable lattices the sectors with fixed planes
  can differ [inferred], a warning is printed).

E6 content: states with E6 weight 0 are singlets; a 27 contributes exactly one state with E6 weight
(s, 0,...,0), s = p1+p2+p3 != 0 (its SO(10) singlet); sign of s separates 27 from 27bar.  The geometric
convention is n(27) = h11, so the sign is fixed by the untwisted sector (one 27 per plane).
Hidden part of the SU(3) of the embedding: the roots e_i - e_j (i,j <= 3) with (e_i-e_j).V in Z survive;
the result is SU(3), SU(2)xU(1) or U(1)^2.  U(1)^2 charges (Q_a, Q_b) = (6(p1-p2), 2(p1+p2-2p3)) of the
shifted momentum (integral for N | 6 twists).

Validated [verified, tests/test_hetorb.py]: T6/Z3 (36,0,252 singlets, 27 (27,1)+81 (1,3bar) twisted),
Z4 (31,7), Z6-I (29,5), Z6-II (35,11; 224 singlets = LGOR mirror LGO at finite tori), Z7 (24,0).
"""
from fractions import Fraction as F
from itertools import product
from collections import Counter, defaultdict
import math
import cmath

NAMED = {
    "Z3": (F(1, 3), F(1, 3), F(-2, 3)),
    "Z4": (F(1, 4), F(1, 4), F(-1, 2)),
    "Z6-I": (F(1, 6), F(1, 6), F(-1, 3)),
    "Z6-II": (F(1, 6), F(1, 3), F(-1, 2)),
    "Z7": (F(1, 7), F(2, 7), F(-3, 7)),
}
FACTORISABLE_OK = {"Z3", "Z4", "Z6-I", "Z6-II", "Z7"}
# Z8 and Z12 need non-factorisable lattices with fixed planes: not supported (degeneracies come out non-integer).


def parse_twist(s):
    if s in NAMED:
        return NAMED[s], s
    parts = [F(x) for x in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError("twist needs three entries, e.g. 1/6,1/3,-1/2, or a name: " + ", ".join(NAMED))
    if sum(parts) != 0:
        raise ValueError("twist entries must sum to 0 (N=1 SUSY)")
    return tuple(parts), s


def _frac(x):
    return x - math.floor(x)


def order(v):
    return math.lcm(*[x.denominator for x in v])


def _e8_shifted(kV, maxnorm2=2):
    out = []
    r = math.sqrt(maxnorm2) + 1e-9
    for half in (F(0), F(1, 2)):
        ranges = []
        for x in kV:
            lo = math.ceil(-r - float(x) - float(half))
            hi = math.floor(r - float(x) - float(half))
            ranges.append([F(n) + half for n in range(lo, hi + 1)])
        for p in product(*ranges):
            if sum(p) % 2 == 0 and sum((a + b) ** 2 for a, b in zip(p, kV)) <= maxnorm2:
                out.append(p)
    return out


def nfix_plane(phi):
    """Fixed points of a rotation by 2 pi phi on a T^2 with that symmetry (Lefschetz)."""
    return round(4 * math.sin(math.pi * float(phi)) ** 2)


def hidden_su3(v):
    roots = [(i, j) for i in range(3) for j in range(3) if i != j and (v[i] - v[j]).denominator == 1]
    n = len(roots)
    return {6: "SU(3)", 2: "SU(2)xU(1)", 0: "U(1)^2"}[n], roots


def fixed_loci(v, k):
    N = order(v)
    planes = []
    for x in v:
        ph = _frac(k * x)
        n = 4 * math.sin(math.pi * float(ph)) ** 2
        planes.append("T2" if ph == 0 else (str(round(n)) if abs(n - round(n)) < 1e-9 else "%.3f" % n))
    return planes


def spectrum(v):
    """Return dict k -> list of (kind, charge(Qa,Qb), su3_dynkin, osc_label, multiplicity) and the gauge info."""
    N = order(v)
    V = list(v) + [F(0)] * 5
    V2 = sum(x * x for x in V)
    v2 = sum(x * x for x in v)
    out = {}
    for k in range(N):
        om = [_frac(k * x) for x in v]
        dc = sum(o * (1 - o) for o in om) / 2
        rstates = []
        cand = [F(2 * n + 1, 2) for n in range(-6, 6)]
        for s in product(cand, repeat=3):
            q = (F(-1, 2),) + s
            if int(sum(x + F(1, 2) for x in q)) % 2 != 1:
                continue
            qsh = [q[0]] + [q[i + 1] + k * v[i] for i in range(3)]
            if sum(x * x for x in qsh) / 2 - F(1, 2) + dc == 0:
                if k == 0 and q[1:] == (F(1, 2),) * 3:
                    continue  # gaugino
                rstates.append(tuple(qsh))
        freqs = []
        for i, o in enumerate(om):
            if o == 0:
                freqs += [(F(1), i, +1), (F(1), i, -1)]
            else:
                freqs += [(o, i, +1), (1 - o, i, -1)]
        lstates = []
        for p in _e8_shifted([k * x for x in V]):
            psh = tuple(p[j] + k * V[j] for j in range(8))
            m = sum(x * x for x in psh) / 2 - 1 + dc
            if m > 0:
                continue
            need = -m
            combos = []

            def rec(start, cur, lev):
                if lev == need:
                    combos.append(tuple(cur))
                    return
                for j in range(start, len(freqs)):
                    if lev + freqs[j][0] <= need:
                        rec(j, cur + [freqs[j]], lev + freqs[j][0])
            rec(0, [], F(0))
            for c in combos:
                lstates.append((psh, c))

        def chi(l):
            if k == 0:
                return 1
            gg = math.gcd(math.gcd(k, l), N)
            c = 1.0
            for x in v:
                if _frac(k * x) == 0:
                    continue
                c *= 4 * math.sin(math.pi * float(_frac(gg * x))) ** 2
            return round(c)

        recs = Counter()
        for psh, c in lstates:
            nosc = sum(f[2] * v[f[1]] for f in c)
            for qsh in rstates:
                ph = (sum(a * b for a, b in zip(psh, V)) - sum(a * b for a, b in zip(qsh[1:], v))
                      + nosc - F(k, 2) * (V2 - v2))
                D = sum(chi(l) * cmath.exp(2j * math.pi * float(ph) * l) for l in range(N)) / N
                d = round(D.real)
                if abs(D - d) > 1e-6 or d < 0:
                    raise RuntimeError("non-integer or negative degeneracy %s in sector %d" % (D, k))
                if d == 0:
                    continue
                s = psh[0] + psh[1] + psh[2]
                rest = psh[3:]
                if any(x != 0 for x in rest):
                    continue  # non-singlet E6 weight: counted through its SO(10)-singlet partner
                if s == 0:
                    kind = "1"
                else:
                    kind = "s+" if s > 0 else "s-"
                Q = (6 * (psh[0] - psh[1]), 2 * (psh[0] + psh[1] - 2 * psh[2]))
                dyn = (psh[0] - psh[1], psh[1] - psh[2])
                osc = " ".join(("a%d" % (f[1] + 1)) + ("" if f[2] > 0 else "*") for f in c)
                recs[(kind, Q, dyn, osc)] += d
        out[k] = recs
    # fix 27 vs 27bar: untwisted 27s (one per plane) carry the majority sign
    u = out[0]
    plus = sum(m for (kd, *_), m in u.items() if kd == "s+")
    minus = sum(m for (kd, *_), m in u.items() if kd == "s-")
    sign27 = "s+" if plus >= minus else "s-"
    final = {}
    for k, recs in out.items():
        lst = []
        for (kd, Q, dyn, osc), m in recs.items():
            kind = "1" if kd == "1" else ("27" if kd == sign27 else "27bar")
            lst.append((kind, Q, dyn, osc, m))
        final[k] = sorted(lst, key=lambda t: (["27", "27bar", "1"].index(t[0]), t[1], t[3]))
    return final


_SU3_WEIGHTS = {  # Dynkin-label weights of small irreps (multiplicities)
    "1": {(0, 0): 1},
    "3": {(1, 0): 1, (-1, 1): 1, (0, -1): 1},
    "3bar": {(0, 1): 1, (1, -1): 1, (-1, 0): 1},
    "8": {(1, 1): 1, (2, -1): 1, (-1, 2): 1, (0, 0): 2, (1, -2): 1, (-2, 1): 1, (-1, -1): 1},
}


def su3_multiplets(states):
    """Peel SU(3) irreps (1, 3, 3bar, 8) off a Counter of Dynkin weights; returns Counter of irreps or None."""
    w = Counter()
    for dyn, m in states:
        w[dyn] += m
    res = Counter()
    order_ = ["8", "3", "3bar", "1"]
    changed = True
    while sum(w.values()) > 0 and changed:
        changed = False
        for irr in order_:
            hw = max(_SU3_WEIGHTS[irr], key=lambda t: (t[0] + t[1], t[0]))
            if w[hw] > 0 and all(w[x] >= mm for x, mm in _SU3_WEIGHTS[irr].items()):
                n = min(w[x] // mm for x, mm in _SU3_WEIGHTS[irr].items())
                for x, mm in _SU3_WEIGHTS[irr].items():
                    w[x] -= n * mm
                res[irr] += n
                changed = True
                break
    if sum(w.values()) != 0:
        return None
    return res


def summarise(v, name=None):
    N = order(v)
    spec = spectrum(v)
    hid, _ = hidden_su3(v)
    lines = []
    tot = Counter()
    vs = ",".join(str(x) for x in v)
    lines.append("HETEROTIC T6/%s  (twist v = (%s), standard embedding V = v)" % (name or "Z%d" % N, vs))
    lines.append("gauge group: E8 x E6 x %s   [hidden part of the embedding SU(3)]" % hid)
    if name not in FACTORISABLE_OK:
        lines.append("warning: fixed-point counts assume a factorisable lattice (Lefschetz); for %s the "
                     "sectors with fixed planes may differ [inferred]" % (name or vs))
    lines.append("charges (Qa,Qb) = (6(p1-p2), 2(p1+p2-2p3)) of the shifted momentum; left-chiral superfields only")
    lines.append("")
    rows = []
    for k in range(N):
        lst = spec[k]
        c = Counter()
        for kind, Q, dyn, osc, m in lst:
            c[kind] += m
        tot += c
        loci = fixed_loci(v, k)
        locs = "-" if k == 0 else " x ".join(loci)
        lines.append("sector theta^%d   fixed loci per plane: %s   27: %d   27bar: %d   singlets: %d"
                     % (k, locs, c["27"], c["27bar"], c["1"]))
        if hid == "SU(3)":
            for kind in ("27", "27bar", "1"):
                sub = [(dyn, m) for kd, Q, dyn, osc, m in lst if kd == kind]
                mult = su3_multiplets(sub) if sub else None
                if mult:
                    lines.append("     %-6s as SU(3) multiplets: %s" % (kind, ", ".join("%d x %s" % (n, irr) for irr, n in sorted(mult.items()))))
        for kind, Q, dyn, osc, m in lst:
            lines.append("     %-6s (Qa,Qb) = (%s,%s)   osc: %-8s  x %d" % (kind, Q[0], Q[1], osc or "-", m))
            rows.append((k, locs, kind, Q, osc, m))
        lines.append("")
    lines.append("TOTAL  27: %d   27bar: %d   E6 singlets: %d   (h11, h21) = (%d, %d) for standard embedding"
                 % (tot["27"], tot["27bar"], tot["1"], tot["27"], tot["27bar"]))
    return "\n".join(lines), rows, tot, spec


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write("sector,fixed_loci,rep,Qa,Qb,oscillators,multiplicity\n")
        for k, locs, kind, Q, osc, m in rows:
            f.write("theta^%d,%s,%s,%s,%s,%s,%d\n" % (k, locs, kind, Q[0], Q[1], osc, m))


def write_tex(rows, path, title):
    with open(path, "w") as f:
        f.write("%% generated by lgor hetorbifold\n\\begin{tabular}{l|l|l|r|l|r}\n\\toprule\n"
                "sector & fixed points per plane & rep & $(Q_a,Q_b)$ & osc. & \\# \\\\\n\\midrule\n")
        last = None
        for k, locs, kind, Q, osc, m in rows:
            if last is not None and k != last:
                f.write("\\midrule\n")
            rep = {"27": "$\\mathbf{27}$", "27bar": "$\\overline{\\mathbf{27}}$", "1": "$\\mathbf 1$"}[kind]
            sec = ("$\\theta^{%d}$" % k if k else "untw.") if k != last else ""
            loc = locs.replace("T2", "$T^2$").replace(" x ", "$\\times$") if k != last else ""
            oscs = " ".join("$\\tilde\\alpha_{%s}%s$" % (o[1], "^*" if o.endswith("*") else "") for o in osc.split()) if osc else "--"
            f.write("%s & %s & %s & $(%s,%s)$ & %s & %d \\\\\n" % (sec, loc, rep, Q[0], Q[1], oscs, m))
            last = k
        f.write("\\bottomrule\n\\end{tabular}\n")
