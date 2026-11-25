"""Quantum anomalies of the 4D symmetries.

The 4D theory is N=1 with gauge group E6 x E8 x G_extra.  The left-handed Weyl
fermions are

* chiral multiplets:  every 27 / 27bar contributes 27 fermions, every singlet 1,
  every chiral 78 contributes 78;
* gauginos: E6 (78), E8 (248), and the extra gauginos (Cartans and roots).

Charges: the extra U(1)/Cartan charges of a fermion are those of its multiplet.
R-charges: chiral fermion R(psi) = R(Phi) - R(theta) = R(Phi) + 3, gaugino
R(lambda) = R(theta) = -3 (normalisation R(W) = -6, as in the rest of the program).

Continuous anomalies (must vanish, or be universal and cancelled by the
Green-Schwarz mechanism):
  U(1)_a - grav^2       Tr Q_a
  U(1)_a - E6^2         sum over 27 / 27bar of 2T(27) Q_a  (2T(27) = 6, 2T(78) = 24)
  U(1)_a - E8^2         0 (nothing is E8 charged except E8 gauginos, which are neutral)
  U(1)_a U(1)_b U(1)_c  Tr Q_a Q_b Q_c
Using the Cartan generators of a non-Abelian extra factor as the U(1)s, these
traces also contain its cubic anomaly and its mixed anomalies.

Discrete anomalies (Ibanez-Ross; Z_N^R as in Lee-Raby-Ratz-Ross-Schieren-
Schmidt-Sohnius-Vaudrevange, here with T normalised as 2T(fundamental of SU(N)) = 1):
  Z_N - G - G       A_G = sum_fermions 2T(r) q          mod N
  Z_N^R - G - G     A_G = sum_fermions 2T(r) R(fermion) mod N
must be universal, A_E6 = A_E8 mod N (Green-Schwarz).  Gravitational and cubic
discrete anomalies depend on heavy states and are not checked.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import List

from .symmetries import SymmetryAnalysis, _mod, fmt

TWO_T = {"27": 6, "27bar": 6, "78": 24, "1": 0}
DIM = {"27": 27, "27bar": 27, "78": 78, "1": 1}
TWO_T_E6_ADJ, TWO_T_E8_ADJ = 24, 60


class AnomalyAnalysis:
    def __init__(self, sym: SymmetryAnalysis):
        from .report import build_multiplets
        self.sym = sym
        self.model = sym.model
        mults, _ = build_multiplets(sym)
        # chiral fermions: (rep, StateInfo of the multiplet)
        self.chiral = [("27", e[0][1]) for e in mults["27"]] + [("27bar", e[0][1]) for e in mults["27bar"]] \
            + [("1", s) for s in mults["singlets"]] + [("78", s) for s in mults["78"]]
        # extra gauginos (E6 singlets at q+ = -3/2, q- = 0), without the E6 Cartan (trace)
        self.gauginos_extra = list(sym.extra_vectors)
        self.n_e6 = 1 if sym.untwisted is not None else 0
        self.results: "OrderedDict[str, list]" = OrderedDict()
        self.problems: List[str] = []
        self._continuous()
        self._discrete()

    # ------------------------------------------------------------------ continuous
    def _continuous(self):
        sym = self.sym
        names = sym.u1_names
        r = len(names)
        rows = []
        if r == 0:
            self.results["continuous"] = rows
            return
        Q = [(DIM[rep], [Fraction(x) for x in s.u1]) for rep, s in self.chiral]
        # extra gauginos: neutral Cartans + roots (come in +- pairs; included for completeness)
        Q += [(1, [Fraction(x) for x in s.u1]) for s in self.gauginos_extra]
        for a in range(r):
            g = sum(d * q[a] for d, q in Q)
            rows.append((f"{names[a]} - gravity^2", g))
        for a in range(r):
            e6 = sum(TWO_T[rep] * Fraction(s.u1[a]) for rep, s in self.chiral)
            rows.append((f"{names[a]} - E6^2", e6))
        for a, b, c in combinations_with_replacement(range(r), 3):
            v = sum(d * q[a] * q[b] * q[c] for d, q in Q)
            lab = " ".join(names[i] for i in (a, b, c))
            rows.append((f"{lab}", v))
        self.results["continuous"] = rows
        bad = [(n, v) for n, v in rows if v != 0]
        if bad:
            self.problems.append(f"{len(bad)} non-vanishing continuous anomaly coefficients")

    # ------------------------------------------------------------------ discrete
    def _discrete(self):
        sym = self.sym
        rows = []
        # non-R discrete symmetries: orbifold Z_n and remnants of broken U(1)s
        groups = [(f"Z{o} #{j + 1}", o, lambda s, j=j: s.discrete[j]) for j, o in enumerate(self.model.orders)]
        groups += [(f"{n} (remnant)", k, lambda s, j=j: s.remnant[j]) for j, (n, _, k) in enumerate(sym.remnants)]
        for name, N, charge in groups:
            vals = [charge(s) for _, s in self.chiral]
            if any(v is None for v in vals):
                rows.append((name, N, None, None, "charges undefined"))
                continue
            A6 = sum(TWO_T[rep] * Fraction(charge(s)) for rep, s in self.chiral)
            A8 = Fraction(0)
            ok = _mod(A6 - A8, N) == 0
            rows.append((name, N, A6, A8, "ok" if ok else "ANOMALOUS"))
            if not ok:
                self.problems.append(f"{name}: A_E6 = {fmt(A6)} differs from A_E8 = 0 mod {N}")
        # R-symmetry
        if sym.n_R is not None:
            M = sym.M
            R_lambda = Fraction(sym.R_THETA)            # -3
            A6 = TWO_T_E6_ADJ * R_lambda + sum(TWO_T[rep] * (Fraction(s.R_superfield) - sym.R_THETA)
                                               for rep, s in self.chiral)
            A8 = TWO_T_E8_ADJ * R_lambda
            ok = _mod(A6 - A8, M) == 0
            rows.append((f"Z{M}^R (R-symmetry)", M, A6, A8, "ok" if ok else "ANOMALOUS"))
            if not ok:
                self.problems.append(f"R-symmetry: A_E6 = {fmt(A6)} and A_E8 = {fmt(A8)} differ mod {M}")
        self.results["discrete"] = rows

    # ------------------------------------------------------------------ output
    @property
    def anomaly_free(self) -> bool:
        return not self.problems

    def text(self) -> str:
        out = [f"ANOMALIES of {self.model.name}",
               "  4D left-handed fermions: chiral multiplets (27: 27 states, 27bar: 27, singlet: 1, 78: 78) and gauginos.",
               "  Chiral fermion R = R(Phi) + 3, gaugino R = -3 (R(W) = -6)."]
        cont = self.results.get("continuous", [])
        if cont:
            nz = [(n, v) for n, v in cont if v != 0]
            out.append(f"  Continuous (extra U(1)s / Cartans of the extra gauge group): {len(cont)} coefficients, "
                       f"{'all vanish' if not nz else str(len(nz)) + ' non-zero'}")
            grav = [(n, v) for n, v in cont if "gravity" in n]
            e6 = [(n, v) for n, v in cont if "E6" in n]
            out.append("    U(1)-gravity^2 : " + ", ".join(f"{n.split(' -')[0]}={fmt(v)}" for n, v in grav))
            out.append("    U(1)-E6^2      : " + ", ".join(f"{n.split(' -')[0]}={fmt(v)}" for n, v in e6))
            out.append("    U(1)-E8^2      : all 0")
            cubic = [(n, v) for n, v in cont if "-" not in n]
            out.append(f"    cubic U(1)_a U(1)_b U(1)_c : {len(cubic)} coefficients, "
                       + ("all 0" if all(v == 0 for _, v in cubic) else
                          "non-zero: " + ", ".join(f"[{n}]={fmt(v)}" for n, v in cubic if v)))
        else:
            out.append("  Continuous: no extra gauge symmetry.")
        out.append("  Discrete (anomaly coefficients with 2T(27)=6, 2T(78)=24, 2T(248)=60; GS requires A_E6 = A_E8 mod N):")
        for name, N, A6, A8, status in self.results.get("discrete", []):
            if A6 is None:
                out.append(f"    {name:<24} {status}")
            else:
                out.append(f"    {name:<24} A_E6 = {fmt(A6):>6}   A_E8 = {fmt(A8):>6}   "
                           f"difference = {fmt(_mod(A6 - A8, N))} mod {N}   {status}")
        if not self.results.get("discrete"):
            out.append("    none")
        out.append("  (gravitational and cubic discrete anomalies depend on heavy states and are not imposed;")
        out.append("   discrete anomalies with non-Abelian extra factors are not checked)")
        out.append("  RESULT: " + ("all checked symmetries are anomaly free" if self.anomaly_free else
                                   "ANOMALIES: " + "; ".join(self.problems)))
        return "\n".join(out) + "\n"
