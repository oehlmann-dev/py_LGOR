"""Command line interface.

    python -m lgor [MODEL.lgo]                             interactive shell (help, guide inside)
    python -m lgor deform MODEL.lgo "5 x1 x2 x3 x4 x5"     deformed model + comparison + VEV proposal
    python -m lgor mirror MODEL.lgo                        Greene-Plesser mirror + mirror check (any Fermat W)
    python -m lgor tdl MODEL.lgo [--lattice]               Verlinde lines: labels, loci, 27^3.1 selection rules
    python -m lgor defects MODEL.lgo [--sections 1,2]      general defect lines (dressing, TY self-duality, Rep(G),
                                                           alpha-induction, F-symbols/anomalies), tagged by status
    python -m lgor blocklines PATH ... [--oneline]         consistent lines of every block (A, Kazama-Suzuki, D, E7)
                                                           and their fusion types; reads .lgo, saved .lgor.json or
                                                           run folders, recomputes nothing

    python -m lgor run MODEL.lgo [MORE.lgo | FOLDER ...]   compute spectra; writes MODEL.summary.txt,
                                                           MODEL.states.txt, MODEL.states.csv, MODEL.lgor.json
    python -m lgor show RESULT.json|MODEL.lgo [--sector K0 [TWIST ...]] [--states]
    python -m lgor compare FOLDER                          group models with identical spectra
    python -m lgor check-lrt MODEL.lgo OLD.lrt             compare with an old C++ result file
    python -m lgor diff A.lgo B.lgo                        states lost / kept / new between two models
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

from . import results
from .model import Model
from .spectrum import compute_model


def _result_path(lgo: Path, outdir: Path = None) -> Path:
    d = outdir or lgo.parent
    return d / (lgo.stem + ".lgor.json")


def _expand(paths: List[str]) -> List[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(p.glob("*.lgo")))
        else:
            out.append(p)
    return out


def _progress(name):
    t0 = time.time()

    def f(i, n):
        if n > 1:
            print(f"\r  {name}: {i}/{n} bunches  ({time.time() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
            if i == n:
                print(file=sys.stderr)
    return f


def _load_any(path: Path):
    if path.suffix == ".json":
        return results.load(path)
    model = Model.from_file(path)
    cached = _result_path(path)
    if cached.exists():
        return results.load(cached)
    return compute_model(model, processes=os.cpu_count() or 1)


def cmd_run(args):
    files = _expand(args.paths)
    if not files:
        print("no .lgo files found")
        return 1
    outdir = Path(args.out) if args.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    status = 0
    for lgo in files:
        out = _result_path(lgo, outdir)
        if out.exists() and not args.force:
            print(f"{lgo.name}: results exist ({out.name}), skipping (use --force to recompute)")
            continue
        try:
            model = Model.from_file(lgo)
            problems = model.check()
            if problems:
                print(f"{lgo.name}: inconsistent model: " + "; ".join(problems))
                status = 1
                continue
            from .mirror import generator_overlap
            warn = generator_overlap(model)
            if warn:
                print(f"{lgo.name}: WARNING {warn}")
            t = time.time()
            spec = compute_model(model, processes=args.jobs, check=not args.no_check,
                                 progress=_progress(lgo.name))
            dt = time.time() - t
        except Exception as e:      # keep going through a folder
            print(f"{lgo.name}: ERROR {e}")
            status = 1
            continue
        from .symmetries import SymmetryAnalysis
        from . import report
        sym = SymmetryAnalysis(spec)
        print(report.summary_text(spec, sym).split("=" * 100)[0].rstrip())
        print(f"({dt:.1f}s)")
        if not args.no_save:
            results.save(spec, out, sym)
            files = report.write_all(spec, sym, out.with_name(lgo.stem))
            if not args.no_yukawa:
                from .yukawa import write_yukawa
                files.append(write_yukawa(sym, out.with_name(lgo.stem + ".yukawa.txt")))
            from .anomalies import AnomalyAnalysis
            an = out.with_name(lgo.stem + ".anomalies.txt")
            an.write_text(AnomalyAnalysis(sym).text())
            files.append(an)
            print("saved " + ", ".join(p.name for p in [out] + files))
        print()
    return status


def cmd_show(args):
    spec = _load_any(Path(args.path))
    style = "latex" if args.latex else ("lgor" if args.lgor_names else "modes")
    if args.sector is None:
        print(spec.summary())
        return 0
    k0, twist = args.sector[0], tuple(args.sector[1:])
    sec = spec.sector(k0, twist)
    v = sec.vacuum
    print(f"Sector {sec.label}  ({'Ramond' if sec.ramond else 'NS'})")
    print(f"  vacuum: E = {v['energy']}, q- = {v['q_minus']}, q+ = {v['q_plus']}, discrete charges {v['discrete_charges']}")
    print("  oscillator modes (mode number = -energy):")
    for row in sec.oscillator_table():
        print("   ", row["field"], " ".join(f"{k}:{row[k]}" for k in ("phi", "phibar", "psi", "psibar")))
    print("  cohomology dimensions (rows q-, columns q+ = -3/2..3/2):")
    for q, row in zip(sec.q_minus_values(), sec.counts()):
        print(f"    q- = {str(q):>5}: {row}")
    if sec.gravitini:
        print(f"  gravitino / E8 x SO(10) gaugino states with q+ = {[str(q) for q in sec.gravitini]}")
    for w in sec.warnings:
        print("  warning:", w)
    if args.states:
        for st in sec.states(q_minus=args.q_minus, q_plus=args.q_plus):
            print(f"  [q-={st.q_minus}, q+={st.q_plus}]  {st.to_string(style)}")
    return 0


def cmd_compare(args):
    files = _expand(args.paths)
    groups = {}
    for p in files:
        res = _result_path(p) if p.suffix == ".lgo" else p
        if not res.exists():
            print(f"{p.name}: no results yet (run `python -m lgor run` first)")
            continue
        spec = results.load(res)
        key = (spec.total.key(), tuple(sorted((s.key(), len(t)) for s, t in spec.grouped())))
        groups.setdefault(key, (spec, []))[1].append(p.name)
    print(f"{len(groups)} inequivalent spectra among {sum(len(v[1]) for v in groups.values())} models")
    for i, (key, (spec, names)) in enumerate(groups.items()):
        print(f"\n=== spectrum {i:04d}: {len(names)} model(s)")
        print(spec.total.table())
        for n in names:
            print("   ", n)
    return 0


def cmd_check_lrt(args):
    from .legacy import read_lrt
    from .linalg import rank
    model = Model.from_file(args.model)
    spec = compute_model(model, processes=args.jobs)
    cpp = read_lrt(args.lrt, model.nfields)
    py = {}
    for b in spec.bunches:
        for sec in b.sectors:
            for a, cx in enumerate(sec.complexes):
                for q in range(4):
                    if cx.cohomology[q]:
                        py[(b.twist, sec.k0, a, q)] = cx.cohomology[q]
    bad = 0
    for k in sorted(set(py) | set(cpp)):
        P, C = py.get(k, []), cpp.get(k, [])
        if len(P) != len(C):
            print(f"twist {k[0]} sector {k[1]} (q- idx {k[2]}, q+ idx {k[3]}): python {len(P)} states, C++ {len(C)}")
            bad += 1
            continue
        monos = sorted({m for p in P + C for _, m in p})
        idx = {m: i for i, m in enumerate(monos)}

        def vecs(polys):
            rows = []
            for p in polys:
                v = [0] * len(idx)
                for c, m in p:
                    v[idx[m]] += c
                rows.append(v)
            return rows
        if rank(vecs(P) + vecs(C), len(idx)) != len(P):
            print(f"twist {k[0]} sector {k[1]} (q- idx {k[2]}, q+ idx {k[3]}): same dimension, different states")
            bad += 1
    print(f"{bad} differences between the Python results and {args.lrt}")
    return 1 if bad else 0


def cmd_diff(args):
    from .compare import compare
    from .symmetries import SymmetryAnalysis
    specs = []
    for p in (args.a, args.b):
        p = Path(p)
        specs.append(_load_any(p) if p.suffix == ".json" else compute_model(Model.from_file(p), processes=args.jobs))
    syms = [SymmetryAnalysis(s) for s in specs]
    na, nb = Path(args.a).name.split(".")[0], Path(args.b).name.split(".")[0]
    text = compare(specs[0], syms[0], specs[1], syms[1], na, nb)
    out = Path(args.out) if args.out else Path(args.b).parent / f"{na}_vs_{nb}.diff.txt"
    out.write_text(text)
    print(text.split("COMPARISON BY CLASS")[0].rstrip())
    print(f"\nsaved {out}")
    return 0


def cmd_shell(args):
    from .shell import main as shell_main
    shell_main(args.model, args.jobs)
    return 0


def cmd_deform(args):
    from .deform import Deformation
    from .symmetries import SymmetryAnalysis
    from . import report, results
    from .anomalies import AnomalyAnalysis
    from .yukawa import write_yukawa
    model = Model.from_file(args.model)
    spec = compute_model(model, processes=args.jobs)
    sym = SymmetryAnalysis(spec)
    d = Deformation(spec, sym, args.terms, name=args.name, processes=args.jobs)
    out = Path(args.out) if args.out else Path(args.model).parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "".join(c if c.isalnum() or c in "-_+" else "_" for c in d.model.name)
    (stem.with_name(stem.name + ".lgo")).write_text(d.model.to_string())
    report.write_all(d.spec, d.sym, stem)
    results.save(d.spec, stem.with_name(stem.name + ".lgor.json"), d.sym)
    write_yukawa(d.sym, stem.with_name(stem.name + ".yukawa.txt"))
    stem.with_name(stem.name + ".anomalies.txt").write_text(AnomalyAnalysis(d.sym).text())
    rep = stem.with_name(stem.name + ".deformation.txt")
    rep.write_text(d.text())
    print(d.compare_text.split("COMPARISON BY CLASS")[0].rstrip())
    print()
    print(d.vev.text(d.stats))
    print(f"files written to {out}/  ({stem.name}.*)")
    return 0


def cmd_mirror(args):
    from .mirror import mirror_model, mirror_check
    from .symmetries import SymmetryAnalysis
    from . import report, results
    from .anomalies import AnomalyAnalysis
    from .yukawa import write_yukawa
    model = Model.from_file(args.model)
    if not model.is_fermat:
        print(f"{Path(args.model).name}: W is not Fermat; `mirror` needs the Fermat/Gepner description (only `run`, `deform` and `diff` handle non-Fermat W in 2.4)")
        return 1
    mm = mirror_model(model)
    specA = compute_model(model, processes=args.jobs)
    specB = compute_model(mm, processes=args.jobs)
    symA, symB = SymmetryAnalysis(specA), SymmetryAnalysis(specB)
    out = Path(args.out) if args.out else Path(args.model).parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "".join(c if c.isalnum() or c in "-_+" else "_" for c in mm.name)
    stem.with_name(stem.name + ".lgo").write_text(mm.to_string())
    report.write_all(specB, symB, stem)
    results.save(specB, stem.with_name(stem.name + ".lgor.json"), symB)
    write_yukawa(symB, stem.with_name(stem.name + ".yukawa.txt"))
    stem.with_name(stem.name + ".anomalies.txt").write_text(AnomalyAnalysis(symB).text())
    ok, txt = mirror_check(specA, symA, specB, symB)
    stem.with_name(stem.name + ".mirrorcheck.txt").write_text(txt)
    print(txt)
    print(f"files written to {out}/  ({stem.name}.*)")
    return 0 if ok else 1


def cmd_tdl(args):
    from .tdl import tdl_text
    status = 0
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        if not model.is_fermat:
            print(f"{Path(lgo).name}: W is not Fermat; `tdl` needs the Fermat/Gepner description (only `run`, `deform` and `diff` handle non-Fermat W in 2.4)")
            status = 1
            continue
        cached = _result_path(lgo)
        spec = results.load(cached) if cached.exists() and not args.force else \
            compute_model(model, processes=args.jobs)
        try:
            txt = tdl_text(spec, couplings=not args.no_couplings, lattice=args.lattice)
        except ValueError as e:
            print(f"{lgo.name}: {e}")
            status = 1
            continue
        out = Path(args.out) / (lgo.stem + ".tdl.txt") if args.out else lgo.with_name(lgo.stem + ".tdl.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(txt)
        print(f"written to {out}")
    return status


def cmd_lines(args):
    from .tdl import survival_text
    status = 0
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        if not model.is_fermat:
            print(f"{Path(lgo).name}: W is not Fermat; `lines` needs the Fermat/Gepner description (only `run`, `deform` and `diff` handle non-Fermat W in 2.4)")
            status = 1
            continue
        cached = _result_path(lgo)
        spec = results.load(cached) if cached.exists() and not args.force else \
            compute_model(model, processes=args.jobs)
        try:
            txt = survival_text(spec)
        except ValueError as e:
            print(f"{lgo.name}: {e}")
            status = 1
            continue
        out = Path(args.out) / (lgo.stem + ".lines.txt") if args.out else lgo.with_name(lgo.stem + ".lines.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(txt)
        print(f"written to {out}")
    return status


def _expand_models(paths: List[str]) -> List[Path]:
    """.lgo / .lgor.json files, run folders (one model each) and folders of either, one level deep."""
    out = []

    def model_files(d: Path):
        js = sorted(p for p in d.glob("*.lgor.json") if "_mirror" not in p.name)
        lg = sorted(p for p in d.glob("*.lgo") if "_mirror" not in p.name)
        stems = {p.name[:-len(".lgor.json")] for p in js} | {p.stem for p in lg}
        if len(stems) == 1:
            return [js[0] if js else lg[0]]
        byname = {p.name[:-len(".lgor.json")]: p for p in js}
        return [byname.get(p.stem, p) for p in lg] + [p for s_, p in sorted(byname.items())
                                                      if s_ not in {q.stem for q in lg}]
    for p in map(Path, paths):
        if p.is_dir():
            files = model_files(p)
            if not files:
                for sub in sorted(q for q in p.iterdir() if q.is_dir()):
                    files += model_files(sub)
            out += files
        else:
            out.append(p)
    return out


def cmd_blocklines(args):
    from .blocklines import ModelLines, load_model
    status = 0
    for path in _expand_models(args.paths):
        try:
            ml = ModelLines(load_model(path))
        except Exception as e:                       # report and go on with the next model
            print(f"{path}: ERROR {type(e).__name__}: {e}")
            status = 1
            continue
        if not (ml.checks_ok and ml.recognised and ml.c3 == 3):
            status = 1
        if args.oneline:
            print(ml.oneline(), flush=True)
        else:
            print(ml.text())
        if args.no_write:
            continue
        stem = path.name[:-len(".lgor.json")] if path.name.endswith(".lgor.json") else path.stem
        out = Path(args.out) / (stem + ".blocklines.txt") if args.out else path.with_name(stem + ".blocklines.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ml.text())
        if not args.oneline:
            print(f"written to {out}")
    return status


def cmd_defects(args):
    from .defects import DefectAnalysis
    status = 0
    secs = tuple(int(x) for x in args.sections.split(",")) if args.sections else (1, 2, 3, 4, 5)
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        if not model.is_fermat:
            print(f"{Path(lgo).name}: W is not Fermat; `defects` needs the Fermat/Gepner description (only `run`, `deform` and `diff` handle non-Fermat W in 2.4)")
            status = 1
            continue
        cached = _result_path(lgo)
        spec = results.load(cached) if cached.exists() and not args.force else None
        try:
            an = DefectAnalysis(model, spec=spec, max_subgroups=args.max_subgroups, max_perms=args.max_perms,
                                lgor_check=not args.no_lgor, processes=args.jobs)
            txt = an.text(secs)
        except ValueError as e:
            print(f"{lgo.name}: {e}")
            status = 1
            continue
        out = Path(args.out) / (lgo.stem + ".defects.txt") if args.out else lgo.with_name(lgo.stem + ".defects.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(txt)
        print(f"written to {out}")
    return status


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m lgor", description="Massless spectra of Landau-Ginzburg orbifolds")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compute spectra of .lgo files or folders of them")
    r.add_argument("paths", nargs="+")
    r.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes")
    r.add_argument("-o", "--out", help="folder for the results (default: next to the .lgo file)")
    r.add_argument("-f", "--force", action="store_true", help="recompute even if results exist")
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--no-check", action="store_true", help="skip the Qbar^2 = 0 check (a bit faster)")
    r.add_argument("--no-yukawa", action="store_true", help="do not write the Yukawa coupling file")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("show", help="print the summary, or one sector and its states")
    s.add_argument("path", help=".lgo file or .lgor.json result")
    s.add_argument("--sector", type=int, nargs="+", metavar="K", help="k0 followed by the discrete twists")
    s.add_argument("--states", action="store_true", help="print the cohomology states")
    s.add_argument("--q-minus", help="only states with this q- (e.g. -1/2)")
    s.add_argument("--q-plus", help="only states with this q+ (e.g. -1/2)")
    s.add_argument("--latex", action="store_true")
    s.add_argument("--lgor-names", action="store_true", help="print B, bB, F, bF as in the old program")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("compare", help="group models with identical spectra (old 'i' command)")
    c.add_argument("paths", nargs="+")
    c.set_defaults(func=cmd_compare)

    k = sub.add_parser("check-lrt", help="compare with a result file of the old C++ program")
    k.add_argument("model")
    k.add_argument("lrt")
    k.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    k.set_defaults(func=cmd_check_lrt)

    sh = sub.add_parser("shell", help="interactive shell (also: python -m lgor [MODEL.lgo])")
    sh.add_argument("model", nargs="?")
    sh.add_argument("-j", "--jobs", type=int, default=None)
    sh.set_defaults(func=cmd_shell)

    df = sub.add_parser("deform", help="deform a model by extra W terms; compare and propose the VEV")
    df.add_argument("model")
    df.add_argument("terms", help='e.g. "5 x1 x2 x3 x4 x5"')
    df.add_argument("--name")
    df.add_argument("-o", "--out")
    df.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    df.set_defaults(func=cmd_deform)

    d = sub.add_parser("diff", help="compare two models (e.g. Fermat point and a deformation)")
    d.add_argument("a", help="reference model (.lgo or .lgor.json)")
    d.add_argument("b", help="deformed model (.lgo or .lgor.json)")
    d.add_argument("-o", "--out", help="output file (default: A_vs_B.diff.txt next to B)")
    d.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    d.set_defaults(func=cmd_diff)

    mi = sub.add_parser("mirror", help="Greene-Plesser mirror of a Fermat model + mirror check")
    mi.add_argument("model")
    mi.add_argument("-o", "--out", help="folder for the mirror .lgo, results and check (default: next to MODEL)")
    mi.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    mi.set_defaults(func=cmd_mirror)

    t = sub.add_parser("tdl", help="Verlinde-line (TDL) analysis of Fermat models: labels, loci, selection rules")
    t.add_argument("paths", nargs="+")
    t.add_argument("-o", "--out", help="folder for MODEL.tdl.txt (default: next to the .lgo file)")
    t.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    t.add_argument("-f", "--force", action="store_true", help="recompute the spectrum even if results exist")
    t.add_argument("--no-couplings", action="store_true", help="skip the 27^3.1 four-point classification")
    t.add_argument("--lattice", action="store_true", help="also test whether any abelian symmetry explains the zeros (slow)")
    t.set_defaults(func=cmd_tdl)

    ln = sub.add_parser("lines", help="which deformations (complex structure, Kähler, blow-ups) keep which lines and products of lines")
    ln.add_argument("paths", nargs="+")
    ln.add_argument("-o", "--out", help="folder for MODEL.lines.txt (default: next to the .lgo file)")
    ln.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    ln.add_argument("-f", "--force", action="store_true", help="recompute the spectrum even if results exist")
    ln.set_defaults(func=cmd_lines)

    de = sub.add_parser("defects", help="general defect lines of Fermat models: survival with dressing, self-duality (TY) "
                                        "defects, group-theoretical lines, alpha-induction, F-symbols and anomalies")
    de.add_argument("paths", nargs="+")
    de.add_argument("-o", "--out", help="folder for MODEL.defects.txt (default: next to the .lgo file)")
    de.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    de.add_argument("-f", "--force", action="store_true", help="recompute the spectrum even if results exist")
    de.add_argument("--sections", help="comma-separated subset of 1,2,3,4,5 (default: all)")
    de.add_argument("--no-lgor", action="store_true", help="skip the LGOR massless cross-check of self-dual gaugings")
    de.add_argument("--max-subgroups", type=int, default=5000, help="cap on the subgroups scanned for self-dual gaugings")
    de.add_argument("--max-perms", type=int, default=120, help="cap on permutations of identical factors tried")
    de.set_defaults(func=cmd_defects)

    bl = sub.add_parser("blocklines", help="consistent lines of every block (minimal, Kazama-Suzuki, D, E7) and their "
                                           "fusion types; reads .lgo, .lgor.json or run folders without recomputing")
    bl.add_argument("paths", nargs="+")
    bl.add_argument("-o", "--out", help="folder for MODEL.blocklines.txt (default: next to the input)")
    bl.add_argument("--oneline", action="store_true", help="print one summary line per model")
    bl.add_argument("--no-write", action="store_true", help="do not write MODEL.blocklines.txt")
    bl.set_defaults(func=cmd_blocklines)

    argv = list(sys.argv[1:] if argv is None else argv)
    # no sub-command: start the interactive shell (optionally with a model file)
    rest = list(argv)
    jobs = None
    if rest[:1] == ["-j"] and len(rest) > 1:
        jobs = int(rest[1])
        rest = rest[2:]
    if not rest or (rest[0] not in sub.choices and not rest[0].startswith("-")):
        if "-j" in rest:
            i = rest.index("-j")
            jobs = int(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
        from .shell import main as shell_main
        shell_main(rest[0] if rest else None, jobs)
        return 0
    # allow "--q-plus -1/2" (argparse would read -1/2 as an option)
    for i in range(len(argv) - 1):
        if argv[i] in ("--q-plus", "--q-minus") and argv[i + 1][:1] == "-":
            argv[i] = argv[i] + "=" + argv[i + 1]
            argv[i + 1] = None
    argv = [a for a in argv if a is not None]
    args = p.parse_args(argv)
    return args.func(args)
