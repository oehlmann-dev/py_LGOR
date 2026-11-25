"""`fixedpoints`: twisted sectors and fixed points of a toroidal LGO against the geometric orbifold (3.13).

For an LGO that is the Greene-Plesser mirror of a factorisable heterotic T^6/Z_N (standard embedding; blocks
A = (1)^3, B = (2)^2, C = (4)(1) as in `toroidal`), this compares the LGOR massless states with the geometric
orbifold state by state, at the level of charge classes:

 1. Orbifold sector of every LGOR state.  Block charge r_b = sum_{i in b}(q_i + 1/2 - w_i) (+1/2 for E6 singlets,
    w_i = field weight) satisfies r_b = k v_b mod 1 for one k (the twist v is given in block order) [verified on
    Z6-II, all 277 states].
 2. C block (x^6 + y^3, the G2 plane): Rep(S3) class of the k=4 factor (L2 = 2, -1, 0 <-> class e, 3-cycle,
    transposition; L4 = (-1)^l) against the orbifold count per G2 fixed-point orbit (origin / pair / triple),
    from the standard projection with the trace of theta^l on the orbit [verified on Z6-II].
 3. A4 origin/triple split of the C-block Z2-twisted sector: in each charge class, states with purely holomorphic
    C-block content in the (l_x, l_y) = (1,1) sector (mode xy) pair with those in (3,0) (mode x^3) into xy +- z x^3
    (A4 eigenvalue 3 = origin for +, -1 = triple for - and every unpaired state).  Rule [inferred]; the counts are
    compared with the orbifold per class [verified on Z6-II].
 4. B block (x^4 + y^4, square torus): label test for g = i^Q, Q = 4(q_x - q_y).  A target-space fixed-point label
    must be trivial on every state of the sectors where the B plane is untwisted, and split the 4 fixed-point
    states of every Z2-twisted multiplet 2 + 2.
 5. R-symmetry.  (a) Quartet pattern: a 90 degree rotation of the square plane about a point that fixes two of the
    four Z2 fixed points acts on each quartet as phase x diag(1,1,1,-1) and commutes with g: every Z2-twisted
    charge class must split into quartets {(r,e),(r+M/2,e),(r,-e),(r,-e)} in (Q_R, g).  (b) Fit: LGOR Q_R against
    rho = prod_b rho_b^{a_b} (rho_b = rotation of plane b by 2 pi/o_b; o = 3, 4, 6 for A, B, C) times
    U(1)^{n-1} gauge (block-charge differences), E6 triality, quantum Z_N and a constant; orbifold fermion
    R-charges R_b = -q_sh,b + N_b - Nbar_b (sign as in the projection phase).  C-block exponents are set to 0
    (theta = prod rho_b^{o_b v_b} is trivial on physical states).  Exact rational solve, no grid.

Input: the LGOR run of the Fermat model (PREFIX.states.csv, PREFIX.lgor.json, PREFIX.lgo) and the R-symmetry rule
(Q_R = a k0 - b q- mod M), read from a summary file where it exists (e.g. the run with the torus moduli on).
Untwisted gauge-neutral singlets (moduli, and the states lifted with them) are left out of the R comparison.

Limitations: three blocks; at most one Z2-twisted B block per sector; A/B/C blocks only; standard embedding.
Z6-II reference results (tests/test_fixedpoints.py): sectors 18/96/51/64/48; Rep(S3) per orbit exact; A4 theta^3
singlets 12/40 = orbifold; g trivial on all theta-even states and 2+2 on all quartets; R = rho_B^-1 rho_A x gauge
x (-1)^F on all kept states (only (a_B, a_A) = (3, 1) works).
"""
from fractions import Fraction as F
from collections import Counter, defaultdict
from itertools import product
import csv
import math
import cmath
import re
from pathlib import Path

from . import hetorb

REP_OF_TYPE = {'27 (16_-1/2 component)': '27', '27bar (16bar_+1/2 component)': '27bar', 'E6 singlet': '1'}
ORDER = {'A': 3, 'B': 4, 'C': 6}


# ----------------------------------------------------------------------------- input
def parse_blocks(s):
    return [[int(i) for i in b.split(',')] for b in s.split(';')]


def parse_twist(s):
    return [F(x) for x in s.split(',')]


def read_lgo_weights(path):
    korder, w = None, []
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\s*KOrder\s*=\s*(\d+)', line)
        if m:
            korder = int(m.group(1))
        m = re.match(r'\s*Field\s+"[^"]*"\s+(\d+)', line)
        if m:
            w.append(int(m.group(1)))
    return [F(x, korder) for x in w]


def read_r_rule(summary):
    txt = Path(summary).read_text()
    m = re.search(r'Q_R = (\d+) k0 - (\d+) q-\s+mod (\d+)\s+\(gauginos: (-?\d+)', txt)
    if not m:
        raise ValueError('no R-symmetry line "Q_R = a k0 - b q- mod M" in %s' % summary)
    return tuple(int(x) for x in m.groups())  # a, b, M, gaugino


def block_type(weights, block):
    exps = sorted(int(1 / weights[i]) for i in block)
    if exps == [3, 3, 3]:
        return 'A'
    if exps == [4, 4]:
        return 'B'
    if exps == [3, 6]:
        return 'C'
    raise ValueError('block %s with exponents %s is not of type A=(3,3,3), B=(4,4), C=(6,3)' % (block, exps))


def load_states(prefix, with_labels=True):
    prefix = str(prefix)
    rows = []
    for x in csv.DictReader(open(prefix + '.states.csv')):
        if x['type'] not in REP_OF_TYPE:
            continue
        rows.append(dict(rep=REP_OF_TYPE[x['type']], sector=x['sector'], state=x['state'], k0=int(x['k0']),
                         q_minus=F(x['q_minus']), q=[F(x['q_%d' % (i + 1)]) for i in range(_nfields(x))]))
    if with_labels and Path(prefix + '.lgor.json').exists():
        from . import results
        from .tdl import state_labels
        spec = results.load(prefix + '.lgor.json')
        lab = {}
        for b in spec.bunches:
            for sec in b.sectors:
                for st in sec.states():
                    if st.q_plus != F(-1, 2):
                        continue
                    ls = state_labels(sec, st)
                    if len(ls) == 1:
                        lab[(sec.label, st.to_string())] = [int(v) for v in next(iter(ls))]
        for r in rows:
            r['l'] = lab.get((r['sector'], r['state']))
    return rows


def _nfields(x):
    n = 0
    while ('q_%d' % (n + 1)) in x:
        n += 1
    return n


# ----------------------------------------------------------------------------- 1. sectors
class Setup:
    def __init__(self, prefix, blocks, twist, rrule=None):
        self.prefix = str(prefix)
        self.weights = read_lgo_weights(self.prefix + '.lgo')
        self.blocks = blocks
        self.v = twist
        if len(blocks) != 3 or len(twist) != 3:
            raise ValueError('fixedpoints needs exactly three blocks and three twist entries')
        if sum(twist) % 1 != 0:
            raise ValueError('twist must sum to an integer')
        self.types = [block_type(self.weights, b) for b in blocks]
        self.N = math.lcm(*[x.denominator for x in twist])
        self.rrule = rrule
        self.rows = load_states(self.prefix)
        for r in self.rows:
            r['rb'] = self.block_charges(r)
            r['k'] = self.sector(r)
            if rrule:
                a, b, M, _ = rrule
                r['R'] = (a * r['k0'] - b * r['q_minus'] - rrule[3]) % M   # chiral superfield R

    def block_charges(self, r):
        sh = F(1, 2) if r['rep'] == '1' else 0
        return [sum(r['q'][i] + F(1, 2) - self.weights[i] for i in b) + sh for b in self.blocks]

    def sector(self, r):
        ks = [k for k in range(self.N) if all((rb - k * vb) % 1 == 0 for rb, vb in zip(r['rb'], self.v))]
        if len(ks) != 1:
            raise ValueError('state %s %s: no unique orbifold sector (%s)' % (r['sector'], r['state'], ks))
        return ks[0]

    def twisted(self, b, k):
        """'untw', 'Z2', or the rotation fraction of plane b in sector k."""
        ph = (k * self.v[b]) % 1
        if ph == 0:
            return 'untw'
        if ph == F(1, 2):
            return 'Z2'
        return ph

    def cblock(self):
        return self.types.index('C') if 'C' in self.types else None

    def bblock(self):
        return self.types.index('B') if 'B' in self.types else None

    def gcharge(self, r):
        b = self.bblock()
        x, y = self.blocks[b]
        return 4 * (r['q'][x] - r['q'][y])


# ----------------------------------------------------------------------------- orbifold side
def orbifold_fermions(v, plane_trace=None):
    """Massless left-chiral fermions of T^6/Z_N (standard embedding) with plane R-charges and degeneracy.
    v in the user's (block) order.  plane_trace(b, k, l) -> trace of theta^l on the chosen fixed points of plane b
    in sector k (None = all fixed points, Lefschetz)."""
    N = math.lcm(*[x.denominator for x in v])
    V = list(v) + [F(0)] * 5
    V2 = sum(x * x for x in V)
    v2 = sum(x * x for x in v)
    out = []
    for k in range(N):
        om = [hetorb._frac(k * x) for x in v]
        dc = sum(o * (1 - o) for o in om) / 2
        rst = []
        cand = [F(2 * n + 1, 2) for n in range(-6, 6)]
        for s in product(cand, repeat=3):
            q = (F(-1, 2),) + s
            if int(sum(x + F(1, 2) for x in q)) % 2 != 1:
                continue
            qsh = [q[0]] + [q[i + 1] + k * v[i] for i in range(3)]
            if sum(x * x for x in qsh) / 2 - F(1, 2) + dc == 0:
                if k == 0 and q[1:] == (F(1, 2),) * 3:
                    continue
                rst.append(tuple(qsh))
        freqs = []
        for i, o in enumerate(om):
            freqs += [(F(1), i, +1), (F(1), i, -1)] if o == 0 else [(o, i, +1), (1 - o, i, -1)]
        lst = []
        for p in hetorb._e8_shifted([k * x for x in V]):
            psh = tuple(p[j] + k * V[j] for j in range(8))
            m = sum(x * x for x in psh) / 2 - 1 + dc
            if m > 0 or any(x != 0 for x in psh[3:]):
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
                lst.append((psh, c))

        def chi(l):
            if k == 0:
                return 1
            gg = math.gcd(math.gcd(k, l), N)
            c = 1
            for b, x in enumerate(v):
                if hetorb._frac(k * x) == 0:
                    continue
                t = plane_trace(b, k, l) if plane_trace else None
                c *= t if t is not None else hetorb.nfix_plane(hetorb._frac(gg * x))
            return c
        for psh, c in lst:
            nosc = sum(f[2] * v[f[1]] for f in c)
            for qsh in rst:
                ph = (sum(a * b for a, b in zip(psh, V)) - sum(a * b for a, b in zip(qsh[1:], v)) + nosc
                      - F(k, 2) * (V2 - v2))
                D = sum(chi(l) * cmath.exp(2j * math.pi * float(ph) * l) for l in range(N)) / N
                d = round(D.real)
                if abs(D - d) > 1e-6 or d < 0:
                    raise RuntimeError('degeneracy %s in sector %d' % (D, k))
                if d == 0:
                    continue
                s = psh[0] + psh[1] + psh[2]
                out.append(dict(k=k, s=s, psh=psh[:3], R=[-qsh[i + 1] + sum(f[2] for f in c if f[1] == i)
                                                         for i in range(3)], osc=c, d=d))
    # 27 vs 27bar: untwisted majority sign = 27 (geometric convention n27 = h11)
    plus = sum(x['d'] for x in out if x['k'] == 0 and x['s'] > 0)
    minus = sum(x['d'] for x in out if x['k'] == 0 and x['s'] < 0)
    for x in out:
        x['rep'] = '1' if x['s'] == 0 else ('27' if (x['s'] > 0) == (plus >= minus) else '27bar')
    return out


def _c_trace(cb, part):
    """theta^l trace on the C-plane (v_C = 1/6) fixed points: part = 'origin' or 'orbit'."""
    def tr(b, k, l):
        if b != cb:
            return None
        if part == 'origin':
            return 1
        gg = math.gcd(math.gcd(k, l), 6)
        return {2: 2, 3: 3}.get(gg, 0)
    return tr


# ----------------------------------------------------------------------------- 2./3. C block
L2_OF_L = [2, 0, -1, 0, 2]


def rep_s3(S):
    """LGOR Rep(S3) class counts of the C-block k=4 factor per sector, and orbifold per-orbit counts."""
    cb = S.cblock()
    if cb is None:
        return None
    i4 = [i for i in S.blocks[cb] if S.weights[i] == F(1, 6)][0]
    lg = Counter()
    for r in S.rows:
        if r.get('l') is None:
            continue
        L2 = L2_OF_L[r['l'][i4]]
        cls = {2: 'e (origin)', -1: '3-cycle (pair)', 0: 'transposition'}[L2]
        lg[(r['k'], r['rep'], cls)] += 1
    vC = S.v[cb]
    orb = Counter()
    if abs(vC) == F(1, 6):
        for part in ('origin', 'orbit'):
            for x in orbifold_fermions(S.v, _c_trace(cb, part)):
                tw = S.twisted(cb, x['k'])
                if tw == 'untw' or x['k'] == 0:
                    continue
                cls = 'e (origin)' if part == 'origin' else ('3-cycle (pair)' if tw != 'Z2' else 'transposition')
                if tw == 'Z2':
                    cls = 'transposition'
                orb[(x['k'], x['rep'], cls, part)] += x['d']
    return lg, orb


def _holo_in(state, fields):
    toks = re.findall(r'(phibar|phi|psibar|psi)(\d+)\[', state)
    return all(t in ('phi', 'psi') for t, i in toks if int(i) - 1 in fields)


def a4_split(S):
    """A4 origin/triple split of the C-block Z2-twisted sector, per class (rep, block-charge differences)."""
    cb = S.cblock()
    if cb is None:
        return None
    blk = S.blocks[cb]
    ix = [i for i in blk if S.weights[i] == F(1, 6)][0]
    iy = [i for i in blk if S.weights[i] == F(1, 3)][0]
    ks = [k for k in range(S.N) if S.twisted(cb, k) == 'Z2']
    groups = defaultdict(list)
    others = [i for i in range(len(S.weights)) if i not in blk]
    for r in S.rows:
        if r['k'] not in ks or r.get('l') is None:
            continue
        key = (r['k'], r['rep'], _diffs(r['rb']), tuple(r['q'][i] for i in others),
               r.get('R'), tuple(r['l'][i] for i in others))
        groups[key].append(r)
    lg = Counter()
    for key, v in groups.items():
        A = [r for r in v if (r['l'][ix], r['l'][iy]) == (1, 1) and _holo_in(r['state'], blk)]
        B = [r for r in v if (r['l'][ix], r['l'][iy]) == (3, 0) and _holo_in(r['state'], blk)]
        n = min(len(A), len(B))
        cls = (key[0], key[1], key[2])
        lg[cls + ('origin',)] += n
        lg[cls + ('triple',)] += len(v) - n
    orb = Counter()
    for part in ('origin', 'orbit'):
        for x in orbifold_fermions(S.v, _c_trace(cb, part)):
            if x['k'] in ks:
                orb[(x['k'], x['rep'], _diffs(x['psh']), 'origin' if part == 'origin' else 'triple')] += x['d']
    return lg, orb


def _diffs(rb):
    return tuple(rb[i] - rb[-1] for i in range(len(rb) - 1))


# ----------------------------------------------------------------------------- 4. B-block label test
def label_test(S, label=None):
    """label(r) -> +1/-1 (default g = i^Q, Q = 4(q_x - q_y) of the B block).  Returns (exceptions on sectors where
    the B plane is untwisted, quartet split counter on Z2-twisted sectors)."""
    bb = S.bblock()
    if bb is None:
        return None
    if label is None:
        def label(r):
            Q = S.gcharge(r)
            return 1 if (int(Q) // 2) % 2 == 0 else -1
    exc = 0
    split = Counter()
    for r in S.rows:
        tw = S.twisted(bb, r['k'])
        if tw == 'untw':
            exc += label(r) != 1
    for key, v in _zt_classes(S).items():
        c = Counter(label(r) for r in v)
        split[(c[1], c[-1])] += 1
    return exc, split


def _zt_classes(S):
    """Charge classes of the sectors where the B plane is Z2-twisted (fixed-point quartets)."""
    bb = S.bblock()
    bx = set(S.blocks[bb])
    others = [i for i in range(len(S.weights)) if i not in bx]
    cl = defaultdict(list)
    for r in S.rows:
        if S.twisted(bb, r['k']) != 'Z2':
            continue
        cl[(r['k'], r['rep'], _diffs(r['rb']), tuple(r['q'][i] for i in others))].append(r)
    return cl


# ----------------------------------------------------------------------------- 5. R-symmetry
def quartets(ms, M):
    """Partition a Counter of (R, g) into quartets {(r,e),(r+M/2,e),(r,-e),(r,-e)}; list of bases or None."""
    if sum(ms.values()) == 0:
        return []
    R0, e0 = min(k for k, v in ms.items() if v > 0)
    h = M // 2
    for rr, e in [(R0, e0), ((R0 - h) % M, e0), (R0, -e0)]:
        need = Counter({(rr, e): 1, ((rr + h) % M, e): 1, (rr, -e): 2})
        if all(ms[k] >= v for k, v in need.items()):
            p = quartets(ms - need, M)
            if p is not None:
                return [(rr, e)] + p
    return None


def r_quartets(S):
    M = S.rrule[2]
    res = {}
    for key, v in _zt_classes(S).items():
        ms = Counter((r['R'], 1 if (int(S.gcharge(r)) // 2) % 2 == 0 else -1) for r in v)
        res[key] = quartets(ms, M)
    return res


def _lgor_r_classes(S, Q):
    """class -> Counter of fermion R / M (quartet bases in Z2-twisted sectors)."""
    a, b, M, gaug = S.rrule
    bb = S.bblock()
    out = defaultdict(Counter)
    for key, bases in Q.items():
        if bases is None:
            raise ValueError('quartet pattern fails for class %s' % (key,))
        for (rr, e) in bases:
            out[(key[0], key[1], key[2])][F((rr + gaug) % M, M)] += 1
    for r in S.rows:
        if bb is not None and S.twisted(bb, r['k']) == 'Z2':
            continue
        if r['k'] == 0 and r['rep'] == '1' and all(d == 0 for d in _diffs(r['rb'])):
            continue  # moduli
        out[(r['k'], r['rep'], _diffs(r['rb']))][F((r['R'] + gaug) % M, M)] += 1
    return out


def _orb_r_classes(S, orb):
    bb = S.bblock()
    out = defaultdict(list)
    for x in orb:
        if x['k'] == 0 and x['rep'] == '1' and all(d == 0 for d in _diffs(x['psh'])):
            continue
        q4 = bb is not None and S.twisted(bb, x['k']) == 'Z2'
        out[(x['k'], x['rep'], _diffs(x['psh']))].append((x['R'], x['d'] // (4 if q4 else 1)))
    return out


def r_fit(S, orb=None):
    """Return (list of solutions, per-exponent report).  A solution is (a, zeta, kappa, alpha, c0) with
    a = rotation exponents per block, alpha = gauge coefficients of the block-charge differences."""
    if orb is None:
        orb = orbifold_fermions(S.v)
    Q = r_quartets(S)
    lg = _lgor_r_classes(S, Q)
    oc = _orb_r_classes(S, orb)
    keys = sorted(lg)
    missing = [k for k in keys if k not in oc]
    if missing:
        return [], {'unmatched classes': missing}
    tri = {'1': 0, '27': 1, '27bar': 2}
    ranges = [range(ORDER[t]) if t != 'C' else [0] for t in S.types]
    sols, report = [], {}
    for a in product(*ranges):
        def base(kk):
            c = Counter()
            for R, m in oc[kk]:
                c[sum(F(a[b], ORDER[S.types[b]]) * R[b] for b in range(3)) % 1] += m
            return c
        sh = {}
        ok = True
        for kk in keys:
            bc = base(kk)
            cands = [c for c in (F(n, S.rrule[2]) for n in range(S.rrule[2]))
                     if Counter({(x + c) % 1: m for x, m in bc.items()}) == lg[kk]]
            if len(cands) != 1:
                ok = False
                break
            sh[kk] = cands[0]
        if not ok:
            report[a] = 'class multisets do not match'
            continue
        found = _solve_shifts(sh, keys, tri, S.N)
        report[a] = '%d gauge solutions' % len(found)
        for f in found:
            sols.append((a,) + f)
    return sols, report


def _solve_shifts(sh, keys, tri, N):
    """sh[kk] = c0 + kappa k/N + alpha . diffs + zeta t/3 mod 1, solved exactly (two block-charge differences).
    Differences are scaled to integers D = L*diffs; alpha is returned for the unscaled differences, mod L."""
    L = math.lcm(*[x.denominator for kk in keys for x in kk[2]])
    D = {kk: (int(kk[2][0] * L), int(kk[2][1] * L)) for kk in keys}
    found = []
    pairs = [(x, y) for x in keys for y in keys if x < y]
    m = None
    for p1, p2 in product(pairs, pairs):
        mm = [[D[p1[0]][0] - D[p1[1]][0], D[p1[0]][1] - D[p1[1]][1]],
              [D[p2[0]][0] - D[p2[1]][0], D[p2[0]][1] - D[p2[1]][1]]]
        det = mm[0][0] * mm[1][1] - mm[0][1] * mm[1][0]
        if det != 0:
            m = mm
            break
    if m is None:
        return found
    inv = [[F(m[1][1], det), F(-m[0][1], det)], [F(-m[1][0], det), F(m[0][0], det)]]
    for ze in range(3):
        for ka in range(N):
            T = {kk: (sh[kk] - F(ze * tri[kk[1]], 3) - F(ka * kk[0], N)) % 1 for kk in keys}
            s = [(T[p1[0]] - T[p1[1]]) % 1, (T[p2[0]] - T[p2[1]]) % 1]
            seen = set()
            for n1 in range(abs(det)):
                for n2 in range(abs(det)):
                    r = [s[0] + n1, s[1] + n2]
                    al = ((inv[0][0] * r[0] + inv[0][1] * r[1]) % 1, (inv[1][0] * r[0] + inv[1][1] * r[1]) % 1)
                    if al in seen:
                        continue
                    seen.add(al)
                    c0 = set((T[kk] - al[0] * D[kk][0] - al[1] * D[kk][1]) % 1 for kk in keys)
                    if len(c0) == 1:
                        found.append((ze, ka, (al[0] * L, al[1] * L), c0.pop()))
    return found


# ----------------------------------------------------------------------------- report
def run(prefix, blocks, twist, rsummary=None):
    S = Setup(prefix, parse_blocks(blocks) if isinstance(blocks, str) else blocks,
              parse_twist(twist) if isinstance(twist, str) else twist,
              read_r_rule(rsummary) if rsummary else None)
    out = []
    w = out.append
    w('LGOR fixedpoints: %s' % S.prefix)
    w('blocks %s  types %s  twist (block order) %s  N = %d' % (S.blocks, ''.join(S.types),
                                                            tuple(str(x) for x in S.v), S.N))
    orb = orbifold_fermions(S.v)
    lgc = Counter((r['k'], r['rep']) for r in S.rows)
    oc = Counter()
    for x in orb:
        oc[(x['k'], x['rep'])] += x['d']
    w('\n1. Orbifold sectors (r_b = k v_b mod 1 for every block)  [verified per state]')
    w('   k   LGOR 27/27bar/1      orbifold 27/27bar/1')
    for k in range(S.N):
        a = tuple(lgc[(k, t)] for t in ('27', '27bar', '1'))
        b = tuple(oc[(k, t)] for t in ('27', '27bar', '1'))
        w('   %d   %-20s %-20s %s' % (k, a, b, '' if k == 0 else ('ok' if a == b else 'DIFFERENT')))
    w('   (untwisted singlets differ by the Fermat-point extra states; compare at finite size)')
    rs = rep_s3(S)
    if rs:
        lg, orbc = rs
        w('\n2. C block: Rep(S3) class (L2) of the k=4 factor vs orbifold G2 fixed-point orbit')
        for k in range(1, S.N):
            for rep in ('27', '27bar', '1'):
                row = {c: lg[(k, rep, c)] for c in ('e (origin)', '3-cycle (pair)', 'transposition')}
                o = {p: sum(v for (kk, rr, c, pp), v in orbc.items() if kk == k and rr == rep and pp == p)
                     for p in ('origin', 'orbit')}
                if any(row.values()) or any(o.values()):
                    w('   theta^%d %-6s LGOR %s | orbifold origin %d, orbit %d' % (k, rep, row, o['origin'], o['orbit']))
    a4 = a4_split(S)
    if a4:
        lg, orbc = a4
        w('\n3. A4 origin/triple split (C block Z2-twisted)  [rule inferred, counts compared]')
        cls = sorted(set(k[:3] for k in lg) | set(k[:3] for k in orbc), key=str)
        for c in cls:
            a = (lg[c + ('origin',)], lg[c + ('triple',)])
            b = (orbc[c + ('origin',)], orbc[c + ('triple',)])
            w('   theta^%d %-6s diffs %-18s LGOR %d/%d  orbifold %d/%d  %s' % (
                c[0], c[1], tuple(str(x) for x in c[2]), a[0], a[1], b[0], b[1], 'ok' if a == b else 'DIFFERENT'))
    lt = label_test(S)
    if lt:
        exc, split = lt
        w('\n4. B-block label g = i^Q, Q = 4(q_x - q_y): exceptions on B-untwisted sectors: %d; '
          'Z2-twisted classes split (g=+1, g=-1): %s' % (exc, dict(split)))
    if S.rrule:
        a_, b_, M, gaug = S.rrule
        w('\n5. R-symmetry Q_R = %d k0 - %d q- mod %d (gaugino %d, superfield = fermion - %d)' % (a_, b_, M, gaug, gaug))
        Q = r_quartets(S)
        bad = [k for k, v in Q.items() if v is None]
        w('   (a) quartet pattern {(r,e),(r+%d,e),(r,-e),(r,-e)}: %d classes, %d fail' % (M // 2, len(Q), len(bad)))
        sols, rep = r_fit(S, orb)
        w('   (b) fit to plane rotations (exponents per block, C fixed to 0):')
        for a, t in rep.items():
            w('       a = %s: %s' % (a, t))
        if sols:
            best = sorted(sols, key=lambda s: (s[1] != 0, s[2] != 0,
                                               max(s[3][0].denominator, s[3][1].denominator)))[0]
            a, ze, ka, al, c0 = best
            w('   RESULT: Q_R = rotation %s x gauge exp 2 pi i (%s d_1 + %s d_2) (d_b = r_b - r_3) x triality^%d x '
              'quantum Z_%d^%d, constant %s (c0 = 1/2 is (-1)^F)' % (
                  ' '.join('rho_%d^%d' % (b + 1, a[b]) for b in range(3) if a[b]), al[0], al[1], ze, S.N, ka, c0))
            ggR = sum(F(a[b], ORDER[S.types[b]]) * F(-1, 2) for b in range(3)) + c0
            w('   gaugino check: orbifold %s x %d = %s; LGOR left-handed gaugino = R(W)/2 = %d' % (
                ggR % 1, M, (ggR * M) % M, (-gaug) % M))
        else:
            w('   no rotation x gauge fit')
    return '\n'.join(out), S
