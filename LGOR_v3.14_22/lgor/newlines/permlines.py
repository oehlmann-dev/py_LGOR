"""Group-theoretical lines from permutations that do NOT normalise the orbifold group (``lgor permlines``).

Setting [literature: Bhardwaj-Tachikawa 2017 (gauging finite subgroups); Ostrik 2003 (module categories)].
The parent LG model W = sum x_f^{n_f} has the symmetry G = G_max x| P (diagonal phases, permutations
of fields with equal exponents).  The LGO X/H gauges the abelian group H (J and the .lgo generators).
The lines of X/H obtained from G form the group-theoretical category C(G, H): simple lines are
labelled by a double coset H g H and an irrep rho of the stabiliser K_g = H cap g H g^-1, with
quantum dimension  d = |H g H| / |H| * dim(rho) = |H : K_g| * dim(rho).
* g = (phase, sigma) with sigma H sigma^-1 = H: invertible lines (phase/permutation symmetries of X/H,
  composed with quantum symmetries rho in H^).  These are already in LGOR (symmetries, perms).
* sigma H sigma^-1 != H:  |H : K| > 1, the line is NON-INVERTIBLE.  This module treats these.

Action on local operators of X/H [inferred from the orbifold construction, checked here by the
normaliser control below]: for psi in the h-twisted sector,
    L(sigma, rho) psi = |H:K| rho(sigma h sigma^-1) P_H(sigma psi)   if sigma h sigma^-1 in H,
                      = 0                                            otherwise,
where sigma psi is the permuted state (sector, charges and modes permuted) and P_H the projection
onto H-invariant states.  P_H(sigma psi) = sigma psi if a state with the permuted labels exists in
the LGOR spectrum of X/H, else 0 (permuted labels absent = not H-invariant).  Control [verified per
model]: for every normalising sigma the map is a bijection of the whole spectrum.

Selection-rule content [inferred, argument in the report]: every correlator identity produced by
these lines is <psi_1..psi_n>_{X/H} = <sigma psi_1 .. sigma psi_n>, i.e. a RELATION between couplings,
or a zero that equals the phase rule of the subgroup sigma^-1 H sigma of G_max (an invertible
symmetry of X/H).  The module counts the relations between 27 multiplets that no invertible
permutation of X/H reproduces.
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from ..toroidal import _gens, _span

V, LIT, INF, MISS = "[verified]", "[literature]", "[inferred]", "[missing]"


def _perm_vec(v, s):
    out = [None] * len(v)
    for f, x in enumerate(v):
        out[s[f]] = x
    return tuple(out)


def _compose(a, b):
    """(a o b)[f] = a[b[f]]"""
    return tuple(a[b[f]] for f in range(len(a)))


def _inverse(s):
    out = [0] * len(s)
    for f, t in enumerate(s):
        out[t] = f
    return tuple(out)


class PermLines:
    def __init__(self, sym, cap_perms: int = 50000, cap_chars: int = 200000):
        from ..perms import candidate_count
        self.sym = sym
        m = sym.model
        self.model = m
        self.n = [m.fermat_exponent(i) for i in range(m.nfields)]
        if any(x is None for x in self.n):
            raise ValueError("permlines needs a Fermat superpotential")
        self.H = _span(_gens(m), self.n)
        nf = m.nfields
        if candidate_count(m) > cap_perms:
            raise ValueError(f"too many candidate permutations ({candidate_count(m)} > {cap_perms})")
        classes = defaultdict(list)
        for f, e in enumerate(self.n):
            classes[e].append(f)
        keys = sorted(classes)
        self.perms = []
        for choice in itertools.product(*[itertools.permutations(classes[e]) for e in keys]):
            s = [0] * nf
            for e, img in zip(keys, choice):
                for a, b in zip(classes[e], img):
                    s[a] = b
            self.perms.append(tuple(s))
        self.ident = tuple(range(nf))
        self.normalising = [s for s in self.perms if all(_perm_vec(h, s) in self.H for h in self.H)]
        self.broken = [s for s in self.perms if s not in set(self.normalising)]
        self.cap_chars = cap_chars
        # states
        st = sym.states
        self.states = st
        N = m.N
        self.N = N

        # all twist vectors (boson Bt, fermion Ft) of the sectors of X/H (k0 in Z_2K: J^(k0/2) twists)
        from ..sector import Sector
        self.twists = set()
        for k0 in range(2 * m.korder):
            for tw in itertools.product(*[range(o) for o in m.orders]):
                sec = Sector(m, k0, tuple(tw))
                sec.compute_twists()
                self.twists.add((0, tuple(sec.Bt), tuple(sec.Ft)))
        self.hv = [(0, tuple(s.sector.Bt), tuple(s.sector.Ft)) for s in st]
        self.phase_normalising = self.normalising
        self.broken = [p for p in self.perms if p not in set(self.normalising)]
        self.keys = [self._key(s) for s in st]
        self.count = Counter(self.keys)

    def _key(self, s):
        # k0 is not part of the key: a permutation may map the sector (k0, g) to (k0', g') with the same
        # field twists (Bt, Ft) when sigma g sigma^-1 = g' J^c (relabelling of the J component)
        return (0, tuple(s.sector.Bt), tuple(s.sector.Ft), s.state.q_plus, s.state.q_minus,
                tuple(Fraction(x) for x in s.q))

    def _pkey(self, key, sig):
        k0, Bt, Ft, qp, qm, q = key
        return (k0, _perm_vec(Bt, sig), _perm_vec(Ft, sig), qp, qm, _perm_vec(q, sig))

    def stabiliser(self, sig):
        """K = H cap sig H sig^-1 (phase vectors)."""
        inv = _inverse(sig)
        return {h for h in self.H if _perm_vec(h, inv) in self.H}

    def characters(self, K):
        """Distinct characters of the abelian group K (phase vectors mod n) as functions b -> exp(2 pi i b.k/n)."""
        Kl = sorted(K)
        size = 1
        for x in self.n:
            size *= x
        if size > self.cap_chars:
            return None
        seen = {}
        for b in itertools.product(*[range(x) for x in self.n]):
            sig = tuple(sum(Fraction(bi * ki, ni) for bi, ki, ni in zip(b, k, self.n)) % 1 for k in Kl)
            if sig not in seen:
                seen[sig] = b
            if len(seen) == len(Kl):
                break
        return list(seen.values())

    def action(self, sig):
        """Class-level action of L(sig, trivial rho): list per state of
        ('in', target_key) | ('out-sector',) | ('not-invariant',) | ('mult-mismatch', target_key)."""
        res = []
        for s, key, h in zip(self.states, self.keys, self.hv):
            if (h[0], _perm_vec(h[1], sig), _perm_vec(h[2], sig)) not in self.twists:
                res.append(("out-sector",))
                continue
            pk = self._pkey(key, sig)
            c = self.count.get(pk, 0)
            if c == 0:
                res.append(("not-invariant",))
            elif c != self.count[key]:
                res.append(("mult-mismatch", pk))
            else:
                res.append(("in", pk))
        return res

    def control(self) -> Tuple[bool, str]:
        bad = 0
        for s in self.normalising:
            act = self.action(s)
            if any(a[0] != "in" for a in act):
                bad += 1
        return bad == 0, (f"normaliser control: {len(self.normalising)} invertible permutations, every state mapped into the "
                          f"spectrum by all of them " + (V if bad == 0 else f"FAILED for {bad}"))

    def double_cosets(self):
        """Representatives of N sigma N for broken sigma (N = normalising permutations)."""
        Nn = self.normalising
        seen, reps = set(), []
        for s in self.broken:
            if s in seen:
                continue
            orb = {_compose(_compose(a, s), b) for a in Nn for b in Nn}
            seen |= orb
            reps.append((s, len(orb)))
        return reps

    def multiplets(self):
        from ..report import build_multiplets
        mults, _ = build_multiplets(self.sym)
        idx = {id(x): i for i, x in enumerate(self.states)}
        out = {}
        for rep in ("27", "27bar"):
            out[rep] = [[idx[id(c[1])] for c in e] for e in mults[rep]]
        return out

    def report(self) -> str:
        m = self.model
        L = ["PERMUTATION LINES (group-theoretical, non-invertible part)", "-" * 100]
        L.append(f"model {m.name}: exponents {self.n}; phase group |H| = {len(self.H)} (J and .lgo generators); "
                 f"{len(self.perms)} permutations of W, {len(self.normalising)} normalise H, {len(self.broken)} do not " + V)
        ok, txt = self.control()
        L.append("  " + txt)
        if not self.broken:
            L.append("  no non-invertible permutation lines (every permutation of W normalises H) " + V)
            return "\n".join(L) + "\n"
        mults = self.multiplets()
        # state index -> multiplet id
        mid = {}
        for rep, ms in mults.items():
            for j, comp in enumerate(ms):
                for i in comp:
                    mid[i] = (rep, j)
        key_to_states = defaultdict(list)
        for i, k in enumerate(self.keys):
            key_to_states[k].append(i)
        Nn = self.normalising
        L.append(f"  double cosets N sigma N of broken permutations: {len(self.double_cosets())}")
        L.append(f"  {'sigma (f -> sigma f)':<22} {'|K|':>5} {'d=|H:K|':>8} {'#rho':>5}  states: fixed / moved / annihilated (sector, projection)"
                 f"   27+27bar multiplets: kept / annihilated;  27-pairs related only by this line")
        self.rows = []
        for s, orbsize in self.double_cosets():
            K = self.stabiliser(s)
            d = len(self.H) // len(K)
            nrho = len(K)          # K abelian: |K| characters rho (they only add phases rho(sigma h sigma^-1))
            act = self.action(s)
            fixed = sum(1 for a, k in zip(act, self.keys) if a[0] == "in" and a[1] == k)
            moved = sum(1 for a, k in zip(act, self.keys) if a[0] == "in" and a[1] != k)
            out_s = sum(1 for a in act if a[0] == "out-sector")
            not_inv = sum(1 for a in act if a[0] == "not-invariant")
            mism = sum(1 for a in act if a[0] == "mult-mismatch")
            kept27 = ann27 = 0
            new_pairs = 0
            for rep in ("27", "27bar"):
                for j, comp in enumerate(mults[rep]):
                    a0 = act[comp[0]]
                    if a0[0] == "in":
                        kept27 += 1
                        # target multiplet(s)
                        tgt = {mid.get(i) for i in key_to_states[a0[1]]}
                        if tgt == {(rep, j)}:
                            continue
                        # is the move reproduced by an invertible (normalising) permutation?
                        src_key = self.keys[comp[0]]
                        if not any(self._pkey(src_key, nu) == a0[1] for nu in Nn):
                            new_pairs += 1
                    else:
                        ann27 += 1
            cyc = self._cycles(s)
            L.append(f"  {cyc:<22} {len(K):>5} {d:>8} {str(nrho):>5}  {fixed} / {moved} / {out_s + not_inv} ({out_s}, {not_inv})"
                     f"{'  mult-mismatch ' + str(mism) if mism else ''}   {kept27} / {ann27};  {new_pairs}")
            self.rows.append(dict(sigma=s, K=len(K), d=d, nrho=nrho, fixed=fixed, moved=moved, out_sector=out_s,
                                  not_invariant=not_inv, mismatch=mism, kept27=kept27, ann27=ann27, new_pairs=new_pairs,
                                  orbit=orbsize))
        L.append("")
        L.append("  Reading: d > 1 lines are non-invertible " + V + " (|H:K| > 1).  They act on a state either by")
        L.append("  mapping it to its permuted partner (relation between couplings) or by annihilating it.  The")
        L.append("  zeros they imply are those of the phase group sigma^-1 H sigma, a subgroup of G_max, which acts")
        L.append("  invertibly on X/H " + INF + ".  '27-pairs related only by this line' counts 27/27bar multiplets")
        L.append("  mapped to a different multiplet by sigma but by no invertible permutation of X/H: these are the")
        L.append("  genuinely non-invertible RELATIONS " + V + " (class level; multiplicity > 1 classes treated as a block " + MISS + ").")
        return "\n".join(L) + "\n"

    @staticmethod
    def _cycles(p):
        seen, cyc = set(), []
        for i in range(len(p)):
            if i in seen or p[i] == i:
                seen.add(i)
                continue
            c, j = [], i
            while j not in seen:
                seen.add(j)
                c.append(j + 1)
                j = p[j]
            cyc.append("(" + " ".join(map(str, c)) + ")")
        return "".join(cyc) or "id"
