"""Interactive LGOR shell.

Start with   python3 -m lgor            (empty session)
       or    python3 -m lgor model.lgo  (loads and computes the model)

Type  help  for the list of commands,  help <command>  for details,
and   guide  for a short walkthrough.
"""

from __future__ import annotations

import cmd
import os
import shlex
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__

BANNER = f"""
 LGOR {__version__}  -  massless spectra of heterotic (2,2) Landau-Ginzburg orbifolds
 ------------------------------------------------------------------------------
 type  help  for all commands,  help <command>  for details,  guide  for a tour.
"""

GUIDE = """
A TOUR OF LGOR
==============
 1. load examples/Q1Fermat2.lgo      read a model (fields, R-charges, discrete
                                      symmetries, superpotential W)
    model                             show what was read
 2. compute                           compute all twisted sectors, the Qbar
                                      cohomology, E6 multiplets and symmetries
 3. summary                           4D spectrum, gauge group, charge generators
    symmetries                        U(1)s, R-symmetry, discrete symmetries
    anomalies                         check that all symmetries are anomaly free
 4. sectors                           all twisted sectors T(k0; k) with vacua
    sector 1                          one sector: vacuum, oscillator modes, states
 5. states 27                         list all 27-plets with their charges
    states 1 --R 6                    singlets with R(Phi) = 6
    state 27-001                      one multiplet: its three component states
                                      (oscillators acting on the twisted vacua) and
                                      all charges
 6. yukawa                            all symmetry-allowed trilinear couplings
 7. deform 5 x1 x2 x3 x4 x5           add a non-Fermat term to W: computes the
                                      deformed model, compares it with the original,
                                      proposes the 4D VEV that causes it and checks
                                      D- and F-flatness
    models / use NAME                 switch between the original and deformed model
    compare, vev                      show the comparison / VEV report again
    mirror                            Greene-Plesser mirror and mirror check
    tdl [--lattice]                   Verlinde lines: labels, loci, 27^3.1 selection rules
 8. save results/                     write summary, states, CSV, JSON, Yukawa,
                                      anomaly and deformation reports
 Output style of states:  style modes | lgor | latex
"""


class Entry:
    def __init__(self, model, spec=None, sym=None, parent=None, deformation=None):
        self.model, self.spec, self.sym = model, spec, sym
        self.parent = parent
        self.deformation = deformation
        self._labels = None

    @property
    def name(self):
        return self.model.name

    def labels(self):
        """label -> (kind, data) for all multiplets, consistent with the summary file."""
        if self._labels is None:
            from .report import build_multiplets
            mults, _ = build_multiplets(self.sym)
            lab = {}
            for rep in ("27", "27bar"):
                for i, e in enumerate(mults[rep], 1):
                    lab[f"{rep}-{i:03d}"] = (rep, e)
            for i, s in enumerate(mults["singlets"], 1):
                lab[f"1-{i:03d}"] = ("1", s)
            for i, s in enumerate(mults["78"], 1):
                lab[f"78-{i:03d}"] = ("78", s)
            vec = [s for s in self.sym.states if s.state.q_plus == Fraction(-3, 2)]
            vec.sort(key=lambda x: (x.state.q_minus != 0, x.weight_zero is False, tuple(x.u1)))
            for i, s in enumerate(vec, 1):
                lab[f"V-{i:03d}"] = ("V", s)
            self._labels = lab
        return self._labels


class LgorShell(cmd.Cmd):
    intro = BANNER
    doc_header = "commands (type help <command>)"

    def __init__(self, processes: Optional[int] = None):
        super().__init__()
        self.entries: Dict[str, Entry] = {}
        self.current: Optional[str] = None
        self.style = "modes"
        self.processes = processes or os.cpu_count() or 1
        self._update_prompt()

    # ------------------------------------------------------------------ helpers
    def _update_prompt(self):
        self.prompt = f"lgor[{self.current}]> " if self.current else "lgor> "

    @property
    def cur(self) -> Optional[Entry]:
        return self.entries.get(self.current) if self.current else None

    def _need_model(self) -> Optional[Entry]:
        if not self.cur:
            print("no model loaded: use  load FILE.lgo")
            return None
        return self.cur

    def _need_spec(self) -> Optional[Entry]:
        e = self._need_model()
        if e and e.spec is None:
            print("not computed yet: use  compute")
            return None
        return e

    def _progress(self, i, n):
        if not sys.stdout.isatty() and i < n:
            return
        print(f"\r  {i}/{n} bunches of twisted sectors", end="" if i < n else "\n", flush=True)

    def emptyline(self):
        pass

    def default(self, line):
        print(f"unknown command: {line.split()[0]}   (type  help)")

    def do_help(self, arg):
        """help [command]   list the commands, or explain one command in detail"""
        if arg:
            return super().do_help(arg)
        groups = [
            ("Start", ["guide", "load", "model", "compute"]),
            ("Spectrum and states", ["summary", "sectors", "sector", "states", "state", "style"]),
            ("Symmetries and couplings", ["symmetries", "anomalies", "yukawa", "tdl"]),
            ("Deformations and mirrors", ["deform", "models", "use", "compare", "vev", "mirror"]),
            ("Files", ["save"]),
            ("Other", ["help", "quit"]),
        ]
        print("\nLGOR commands (help <command> for details):")
        short = {
            "guide": "guide                 a short tour through the main features",
            "load": "load FILE.lgo         read a model file",
            "model": "model                 show fields, R-charges, discrete charges, W",
            "compute": "compute [-j N]        compute the massless spectrum and all symmetries",
            "summary": "summary               4D spectrum, gauge group, charge generators",
            "sectors": "sectors               table of all twisted sectors and their states",
            "sector": "sector K0 [T..]       one twisted sector: vacuum, oscillator modes, states",
            "states": "states KIND [opts]    list 27 / 27bar / 1 / V multiplets with charges",
            "state": "state LABEL           one multiplet in full: worldsheet states and charges",
            "style": "style modes|lgor|latex  how worldsheet states are printed",
            "symmetries": "symmetries            U(1)s, R-symmetry, discrete symmetries, remnants",
            "anomalies": "anomalies             check that all symmetries are anomaly free",
            "yukawa": "yukawa [type]         symmetry-allowed trilinear couplings",
            "tdl": "tdl [--lattice]       Verlinde lines (TDLs): labels, loci, 27^3.1 selection rules",
            "deform": "deform TERMS          add non-Fermat terms to W: compute, compare, VEV + flatness",
            "models": "models                list the models of this session",
            "use": "use NAME              switch to another model",
            "compare": "compare [A B]         state-by-state comparison of two models",
            "vev": "vev                   VEV proposal, D-/F-flatness, Higgs-mechanism check",
            "mirror": "mirror [NAME]         Greene-Plesser mirror + state-by-state mirror check",
            "save": "save [DIR]            write all result files",
            "help": "help [command]        this list, or details on one command",
            "quit": "quit                  leave (also exit, Ctrl-D)",
        }
        for title, cmds in groups:
            print(f"\n  {title}")
            for c in cmds:
                print(f"    {short.get(c, c)}")
        print()

    # ------------------------------------------------------------------ start
    def do_guide(self, arg):
        """guide   a short tour through the main features"""
        print(GUIDE)

    def do_load(self, arg):
        """load FILE.lgo   read a model file

        The .lgo format:
            Name = "Q1Fermat2"
            KOrder = 5               alpha_i = q_i / KOrder
            Order = 5 5              orders of the discrete Z_n symmetries (may be empty)
            Field "1" 1 1 0          name, q_i, one discrete charge per Z_n (summing to 0)
            SPotTerm 1 5 0 0 0 0     coefficient, power of every field
        Optional (ignored by the old C++ program):
            U1 "name" c1 ... cn      your own U(1) generator  c.q
            RCorrection n1 ... nn    your own correction n in Q_R = 3k0 - 2q- + n.q
            DiscreteCorrection j n1 ... nn
        After loading, use  compute."""
        from .model import Model
        path = arg.strip().strip('"')
        if not path:
            print("usage: load FILE.lgo")
            return
        try:
            m = Model.from_file(path)
        except Exception as e:
            print(f"cannot read {path}: {e}")
            return
        problems = m.check()
        if problems:
            print("the model is inconsistent:\n  " + "\n  ".join(problems))
            return
        name = m.name
        while name in self.entries:
            name += "'"
        m.name = name
        self.entries[name] = Entry(m)
        self.current = name
        self._update_prompt()
        print(m.describe())
        print("\nnext:  compute")

    def do_model(self, arg):
        """model   show the current model: fields, R-charges, discrete charges, superpotential"""
        e = self._need_model()
        if not e:
            return
        from .report import _poly_W
        print(e.model.describe())
        print("  W = " + _poly_W(e.model))
        if e.parent:
            print(f"  (deformation of {e.parent})")

    def do_compute(self, arg):
        """compute [-j N]   compute the massless spectrum of the current model

        Goes through all twisted sectors T(k0; k1..kN), constructs the massless
        neutral oscillator states, computes the Qbar cohomology exactly, assembles
        E6 multiplets and analyses the symmetries.  -j N uses N processes
        (default: all cores)."""
        e = self._need_model()
        if not e:
            return
        procs = self.processes
        a = arg.split()
        if "-j" in a:
            procs = int(a[a.index("-j") + 1])
        from .spectrum import compute_model
        from .symmetries import SymmetryAnalysis
        t = time.time()
        e.spec = compute_model(e.model, processes=procs, progress=self._progress)
        e.sym = SymmetryAnalysis(e.spec)
        e._labels = None
        print(f"done in {time.time() - t:.1f}s\n")
        self.do_summary("")

    # ------------------------------------------------------------------ spectrum
    def do_summary(self, arg):
        """summary   4D spectrum, extra gauge group and charge generators of the current model"""
        e = self._need_spec()
        if not e:
            return
        from .report import summary_text
        print(summary_text(e.spec, e.sym).split("=" * 100)[0].rstrip())
        print("\nnext:  states 27 | states 1 | state 27-001 | anomalies | yukawa | deform ...")

    def do_symmetries(self, arg):
        """symmetries   all symmetry generators: Cartan charges q^i, extra U(1)s / gauge group,
        R-symmetry Q_R = 3k0 - 2q- + n.q mod 6K, discrete symmetries and remnants of broken U(1)s"""
        e = self._need_spec()
        if e:
            print(e.sym.describe())

    def do_anomalies(self, arg):
        """anomalies   check that all symmetries are anomaly free

        Continuous: U(1)-gravity^2, U(1)-E6^2, U(1)-E8^2 and all cubic U(1)_a U(1)_b U(1)_c
        anomalies of the extra U(1)s (Cartans of non-Abelian factors included).
        Discrete: Z_N - E6^2 and Z_N^R - E6^2 compared with E8 (Green-Schwarz universality)."""
        e = self._need_spec()
        if e:
            from .anomalies import AnomalyAnalysis
            print(AnomalyAnalysis(e.sym).text())

    def do_tdl(self, arg):
        """tdl [--lattice] [--no-couplings]   Verlinde-line (TDL) analysis of a Fermat model

        A. su(2)_k labels of every massless state; B. line eigenvalues; C. which marginal
        deformations keep each non-invertible line; D. the flat-coordinate 27^3.1 couplings
        at the Fermat point compared with the weak and strict su(2) fusion rules;
        --lattice: E. can any abelian symmetry explain the vanishing couplings (slow)."""
        e = self._need_spec()
        if e:
            from .tdl import tdl_text
            try:
                print(tdl_text(e.spec, couplings="--no-couplings" not in arg, lattice="--lattice" in arg))
            except ValueError as err:
                print(err)

    def do_defects(self, arg):
        """defects [1,2,3,4,5] [--no-lgor]   general defect lines of a Fermat model

        1 survival of Verlinde lines with dressing; 2 self-duality (Tambara-Yamagami) defects from
        half gauging (exact torus test + LGOR massless check); 3 group-theoretical lines (Rep(G),
        C(G,K)); 4 alpha-induction, fixed points, sum Z^2; 5 F-symbols and anomalies.
        Every result is tagged [verified], [literature], [inferred] or [missing]."""
        e = self._need_spec()
        if e:
            from .defects import DefectAnalysis
            nums = [int(x) for x in arg.replace("--no-lgor", "").replace(",", " ").split() if x.isdigit()]
            try:
                an = DefectAnalysis(e.spec.model, spec=e.spec, lgor_check="--no-lgor" not in arg)
                print(an.text(tuple(nums) if nums else (1, 2, 3, 4, 5)))
            except ValueError as err:
                print(err)

    def do_sectors(self, arg):
        """sectors   table of all twisted sectors T(k0; k): vacuum energy and charges, number of
        massless states per q- (gauginos | chiral), gravitino/E8 sectors"""
        e = self._need_spec()
        if not e:
            return
        print(f"  {'sector':<22}{'type':<5}{'E_vac':>7}{'q-':>7}{'q+':>7}   q-: #(q+=-3/2) | #(q+=-1/2)")
        from .symmetries import fmt
        for b in e.spec.bunches:
            for sec in b.sectors:
                items = [f"{fmt(q)}: {r[0]}|{r[1]}" for q, r in zip(sec.q_minus_values(), sec.counts()) if r[0] or r[1]]
                if not items and not sec.gravitini:
                    continue
                v = sec.vacuum
                print(f"  {sec.label:<22}{'R' if sec.ramond else 'NS':<5}{fmt(v['energy']):>7}{fmt(v['q_minus']):>7}"
                      f"{fmt(v['q_plus']):>7}   " + "  ".join(items) + ("  + gravitino/E8" if sec.gravitini else ""))
        print("\nnext:  sector K0 [twists]")

    def do_sector(self, arg):
        """sector K0 [T1 T2 ...]   details of one twisted sector T(K0; T1, T2, ...)

        Shows the vacuum (energy, q-, q+, discrete charges), the mode numbers of all
        oscillators phi, phibar, psi, psibar of every field, the dimensions of the
        cohomology for every (q-, q+), and the states with their multiplet labels."""
        e = self._need_spec()
        if not e:
            return
        try:
            nums = [int(x) for x in arg.split()]
        except ValueError:
            print("usage: sector K0 [T1 T2 ...]")
            return
        if not nums:
            print("usage: sector K0 [T1 T2 ...]")
            return
        k0, tw = nums[0], tuple(nums[1:]) or tuple([0] * e.model.ndiscrete)
        try:
            sec = e.spec.sector(k0, tw)
        except (KeyError, IndexError):
            print(f"no sector T({k0}; {tw})")
            return
        from .symmetries import fmt
        v = sec.vacuum
        print(f"Sector {sec.label}  ({'Ramond' if sec.ramond else 'Neveu-Schwarz'})")
        print(f"  vacuum {sec.vacuum_ket()}: E = {fmt(v['energy'])}, q- = {fmt(v['q_minus'])}, "
              f"q+ = {fmt(v['q_plus'])}, discrete charges {v['discrete_charges']}")
        print("  oscillator mode numbers (a mode with number m costs energy -m):")
        print(f"    {'field':<8}{'phi':>8}{'phibar':>8}{'psi':>8}{'psibar':>8}")
        for row in sec.oscillator_table():
            print(f"    {row['field']:<8}" + "".join(f"{fmt(row[k]):>8}" for k in ("phi", "phibar", "psi", "psibar")))
        print("  cohomology dimensions (q+ = -3/2, -1/2, 1/2, 3/2):")
        for q, row in zip(sec.q_minus_values(), sec.counts()):
            if any(row):
                print(f"    q- = {fmt(q):>5}: {row}")
        if sec.gravitini:
            print("  + vacuum at E = -1: gravitino, E8 and SO(10) gauginos")
        inv = {id(d[1][0][1].state.terms) if k in ("27", "27bar") else id(d[1].state.terms): l
               for l, d in e.labels().items() for k in [d[0]]}
        comp = {}
        for l, (k, d) in e.labels().items():
            if k in ("27", "27bar"):
                for lab, s in d:
                    comp[id(s.state.terms)] = f"{l} ({lab})"
        print("  states (q+ < 0):")
        for s in [x for x in e.sym.states if x.sector is sec and x.state.q_plus < 0]:
            lab = inv.get(id(s.state.terms)) or comp.get(id(s.state.terms)) or "-"
            print(f"    {lab:<22} q-={fmt(s.state.q_minus):>4} q+={fmt(s.state.q_plus):>4}  {s.state.to_string(self.style)}")

    def do_states(self, arg):
        """states KIND [--R r] [--sector K0 [T..]] [--max N]   list multiplets with their charges

        KIND: 27, 27bar, 1 (E6 singlets), V (vector multiplets / gauginos), 78, all
        --R r        only those with R-charge r (superfield R for chiral, gaugino R for V)
        --sector ..  only those whose (first) component lives in sector T(K0; T..)
        --max N      show at most N (default 60; 0 = all)
        Columns: label, U(1) charges, R, discrete and remnant charges, sector, state.
        Use  state LABEL  for the full information on one of them."""
        e = self._need_spec()
        if not e:
            return
        try:
            a = shlex.split(arg)
        except ValueError:
            a = arg.split()
        kind = a[0] if a else "all"
        rfilter = None
        sfilter = None
        mx = 60
        i = 1
        while i < len(a):
            if a[i] == "--R":
                rfilter = Fraction(a[i + 1]) % e.sym.M
                i += 2
            elif a[i] == "--max":
                mx = int(a[i + 1])
                i += 2
            elif a[i] == "--sector":
                j = i + 1
                nums = []
                while j < len(a) and not a[j].startswith("--"):
                    nums.append(int(a[j]))
                    j += 1
                sfilter = (nums[0], tuple(nums[1:]) or tuple([0] * e.model.ndiscrete))
                i = j
            else:
                print(f"unknown option {a[i]}")
                return
        kinds = {"27": ["27"], "27bar": ["27bar"], "1": ["1"], "singlets": ["1"], "V": ["V"], "v": ["V"],
                 "vectors": ["V"], "78": ["78"], "all": ["V", "27", "27bar", "78", "1"]}
        if kind not in kinds:
            print("KIND must be one of 27, 27bar, 1, V, 78, all")
            return
        from .report import _charges_cols
        width = max([len(str(x)) for s in e.sym.states for x in s.u1] + [1])
        shown = 0
        total = 0
        for lab, (k, d) in e.labels().items():
            if k not in kinds[kind]:
                continue
            s = d[0][1] if k in ("27", "27bar") else d
            R = s.R if k == "V" else s.R_superfield
            if rfilter is not None and R != rfilter:
                continue
            if sfilter and (s.sector.k0, s.sector.twist) != sfilter:
                continue
            total += 1
            if mx and shown >= mx:
                continue
            shown += 1
            print(f"  {lab:<10} {_charges_cols(s, e.sym, width, superfield=(k != 'V')):<40} {s.sector.label:<16} "
                  f"{s.state.to_string(self.style)}")
        if total > shown:
            print(f"  ... {total - shown} more (use --max 0 to show all)")
        print(f"  {total} multiplet(s)")

    def do_state(self, arg):
        """state LABEL   full information on one multiplet, e.g.  state 27-001,  state 1-017,  state V-003

        For a 27 / 27bar: all three components (1, 16, 10 of SO(10)), each as an
        oscillator polynomial acting on its twisted vacuum, with the mode numbers used.
        For every state: q-, q+, the Cartan charges q^i, the U(1) charges, the R-charge
        and the discrete / remnant charges."""
        e = self._need_spec()
        if not e:
            return
        lab = arg.strip()
        if lab not in e.labels():
            print(f"unknown label {lab}  (labels look like 27-001, 27bar-001, 1-001, V-001; see  states)")
            return
        k, d = e.labels()[lab]
        comps = d if k in ("27", "27bar") else [(k if k != "V" else "gaugino", d)]
        from .report import _charge_str
        from .symmetries import fmt
        print(f"{lab}:  " + {"27": "27-plet of E6", "27bar": "27bar-plet of E6", "1": "E6 singlet (chiral)",
                             "78": "chiral 78 of E6", "V": "vector multiplet (gaugino state)"}[k])
        for clab, s in comps:
            sec = s.sector
            print(f"  component {clab}:  sector {sec.label}, q- = {fmt(s.state.q_minus)}, q+ = {fmt(s.state.q_plus)}")
            print(f"    state   {s.state.to_string(self.style)}")
            used = set()
            n = e.model.nfields
            for _, m in s.state.terms:
                for kind in range(4):
                    for i in range(n):
                        if m[kind * n + i]:
                            used.add((kind, i))
            if used:
                names = ("phi", "phibar", "psi", "psibar")
                print("    modes   " + ", ".join(f"{names[kk]}{e.model.fields[i].name}: m = {fmt(sec.mode_number(kk, i))}"
                                              for kk, i in sorted(used)))
            for line in _charge_str(s, e.sym):
                print("    " + line)

    def do_style(self, arg):
        """style modes|lgor|latex   how states are printed:
        modes  phi1[-1/10]^2 psi3[-3/5] |1>      (field, mode number)
        lgor   B1^2 F3                             (names of the old C++ program)
        latex  (\\phi^{1}_{-1/10})^{2}\\,\\psi^{3}_{-3/5} \\left|1\\right\\rangle"""
        if arg.strip() in ("modes", "lgor", "latex"):
            self.style = arg.strip()
            print(f"style: {self.style}")
        else:
            print("usage: style modes|lgor|latex")

    # ------------------------------------------------------------------ couplings
    def do_yukawa(self, arg):
        """yukawa [27|27bar|mixed|singlets] [--max N]   trilinear superpotential couplings allowed by
        E6, the extra U(1)s, the discrete symmetries and the R-symmetry (R(W) = -6 mod M),
        separated by type: 27.27.27, 27bar^3, 27.27bar.1 and singlets only.  Without an
        argument: the counts and the first couplings of every type."""
        e = self._need_spec()
        if not e:
            return
        from .yukawa import YukawaAnalysis
        a = arg.split()
        mx = 10
        if "--max" in a:
            mx = int(a[a.index("--max") + 1]) or None
        y = YukawaAnalysis(e.sym)
        text = y.text(max_list=mx)
        sel = {"27": "27.27.27", "27bar": "27bar.27bar.27bar", "mixed": "27.27bar.1", "singlets": "1.1.1"}
        if a and a[0] in sel:
            blocks = text.split("=" * 100)
            for i, b in enumerate(blocks):
                if b.strip().startswith(sel[a[0]]):
                    print("=" * 100 + b + "=" * 100 + blocks[i + 1])
            return
        print(text)

    # ------------------------------------------------------------------ deformations
    def do_deform(self, arg):
        """deform TERMS [as NAME]   add non-Fermat terms to W and analyse the deformed model

        TERMS: e.g.   5 x1 x2 x3 x4 x5        or   x1^2 x2 x3 x4 + 2 x2^3 x3 x5
        (fields x<name> as in the model file, integer coefficients).  The terms must
        have R-charge 1 and be invariant under the discrete symmetries.
        The deformed model is computed completely and becomes the current model.  Then
          - compare: states lost / surviving / new and broken symmetries,
          - vev:     the Fermat singlets whose VEV is this deformation
                     (worldsheet state sum_i psi_i d_i(dW)|1>), with exact D-flatness,
                     F-flatness (trilinear and to all orders) and the 4D Higgs-mechanism
                     prediction (broken U(1)s, discrete remnant, R-symmetry) compared
                     with the worldsheet result."""
        e = self._need_spec()
        if not e:
            return
        text, name = arg, None
        if " as " in arg:
            text, name = arg.rsplit(" as ", 1)
            name = name.strip()
        from .deform import Deformation
        t = time.time()
        try:
            d = Deformation(e.spec, e.sym, text, name=name or f"{e.name}+def{sum(1 for x in self.entries.values() if x.parent == e.name) + 1}",
                            processes=self.processes, progress=self._progress)
        except ValueError as err:
            print(err)
            return
        new = Entry(d.model, d.spec, d.sym, parent=e.name, deformation=d)
        self.entries[d.model.name] = new
        self.current = d.model.name
        self._update_prompt()
        print(f"computed in {time.time() - t:.1f}s; current model is now {d.model.name} (use  use {e.name}  to go back)\n")
        print(d.compare_text.split("COMPARISON BY CLASS")[0].rstrip())
        print()
        print(d.vev.text(d.stats))
        print("next:  compare  (state by state) | vev | summary | states 1 | save DIR")

    def do_models(self, arg):
        """models   list all models of this session (* = current)"""
        for n, e in self.entries.items():
            st = "computed" if e.spec else "not computed"
            print(f" {'*' if n == self.current else ' '} {n:<30} {st}" + (f"   deformation of {e.parent}" if e.parent else ""))

    def do_use(self, arg):
        """use NAME   make another model of this session the current one (see  models)"""
        if arg.strip() in self.entries:
            self.current = arg.strip()
            self._update_prompt()
        else:
            print("unknown model; see  models")

    def do_compare(self, arg):
        """compare [A B]   compare two models state by state (default: current model and the model
        it was deformed from): lost / surviving / new states per class of preserved quantum numbers"""
        names = arg.split()
        if len(names) == 2:
            A, B = (self.entries.get(n) for n in names)
        else:
            B = self.cur
            A = self.entries.get(B.parent) if B and B.parent else None
        if not A or not B or A.spec is None or B.spec is None:
            print("usage: compare A B   (both computed)  - or use it on a deformed model")
            return
        if B.deformation and A.name == B.parent:
            print(B.deformation.compare_text)
        else:
            from .compare import compare
            print(compare(A.spec, A.sym, B.spec, B.sym, A.name, B.name))

    def do_vev(self, arg):
        """vev   the VEV proposal, D-/F-flatness and Higgs-mechanism check of the current deformed model"""
        e = self._need_spec()
        if not e:
            return
        if not e.deformation:
            print("the current model is not a deformation; use  deform ...")
            return
        print(e.deformation.vev.text(e.deformation.stats))

    def do_mirror(self, arg):
        """mirror [NAME]   Greene-Plesser mirror of the current model and the mirror check

        Builds the mirror orbifold (any Fermat W; for a common prime exponent the F_n
        construction, otherwise the Berglund-Huebsch-Krawitz dual group), computes it, adds it to the session as NAME (default
        MODEL_mirror) and checks: 27 <-> 27bar, singlets, gauge group, all states
        state-by-state under (q+, q-, q^i) -> (q+, -q-, -q^i), the complete discrete
        symmetry group modulo gauge transformations on both sides, and anomalies."""
        e = self._need_spec()
        if not e:
            return
        from .mirror import mirror_model, mirror_check
        from .symmetries import SymmetryAnalysis
        try:
            mm = mirror_model(e.model, arg.strip() or None)
        except ValueError as err:
            print(err)
            return
        t = time.time()
        from .spectrum import compute_model
        spec = compute_model(mm, processes=self.processes, progress=self._progress)
        sym = SymmetryAnalysis(spec)
        self.entries[mm.name] = Entry(mm, spec, sym)
        print(f"mirror {mm.name} computed in {time.time() - t:.1f}s  (discrete group "
              + ", ".join(f"Z{o}" for o in mm.orders) + f"); use  use {mm.name}  to inspect it\n")
        ok, txt = mirror_check(e.spec, e.sym, spec, sym)
        print(txt)

    # ------------------------------------------------------------------ files
    def do_save(self, arg):
        """save [DIR]   write all results of the current model into DIR (default: next to the input):
        NAME.summary.txt, NAME.states.txt, NAME.states.csv, NAME.lgor.json, NAME.yukawa.txt,
        NAME.anomalies.txt, NAME.lgo (the model) and, for a deformed model,
        NAME.deformation.txt (comparison + VEV proposal)"""
        e = self._need_spec()
        if not e:
            return
        d = Path(arg.strip() or (Path(e.model.filename).parent if e.model.filename else "."))
        d.mkdir(parents=True, exist_ok=True)
        stem = d / "".join(c if c.isalnum() or c in "-_+" else "_" for c in e.name)
        from . import report, results
        from .anomalies import AnomalyAnalysis
        from .yukawa import write_yukawa
        files = report.write_all(e.spec, e.sym, stem)
        files.append(results.save(e.spec, stem.with_name(stem.name + ".lgor.json"), e.sym))
        files.append(write_yukawa(e.sym, stem.with_name(stem.name + ".yukawa.txt")))
        p = stem.with_name(stem.name + ".anomalies.txt")
        p.write_text(AnomalyAnalysis(e.sym).text())
        files.append(p)
        p = stem.with_name(stem.name + ".lgo")
        p.write_text(e.model.to_string())
        files.append(p)
        if e.deformation:
            p = stem.with_name(stem.name + ".deformation.txt")
            p.write_text(e.deformation.text())
            files.append(p)
        print("written:\n  " + "\n  ".join(str(f) for f in files))

    # ------------------------------------------------------------------ exit
    def do_quit(self, arg):
        """quit   leave LGOR (also: exit, Ctrl-D)"""
        print("bye")
        return True

    do_exit = do_quit
    do_EOF = do_quit


def main(path: Optional[str] = None, processes: Optional[int] = None, commands: Optional[List[str]] = None):
    sh = LgorShell(processes)
    if path:
        sh.onecmd(f"load {path}")
        if sh.cur:
            sh.onecmd("compute")
    if commands:
        for c in commands:
            print(f"{sh.prompt}{c}")
            if sh.onecmd(c):
                break
        return
    try:
        sh.cmdloop()
    except KeyboardInterrupt:
        print("\nbye")
