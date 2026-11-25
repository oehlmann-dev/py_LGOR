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
    python -m lgor enumerative MODEL.lgo --factor F --r R [--numerics] [--h11 N]
                                                           lines vs enumerative data: brane charges, Kahler
                                                           couplings at the Gepner point, survival, 1-parameter
                                                           periods/GV/monodromies

    python -m lgor toroidal MODEL.lgo [--family]           toroidal orbifold phase, remnant symmetries, lines (2.7)
    python -m lgor orbifold CBA 121 | --all                LGO whose mirror is T^6/Gamma (2.7)
    python -m lgor family MODEL.lgo --torus|--terms "..."  moduli alone and together: spectrum, symmetries, lines
    python -m lgor lineonly MODEL.lgo --factor F --r R     line-only zero candidates (phase + u1_forced filtered)
    python -m lgor perms MODEL.lgo [--terms "..."]         permutation symmetries
    python -m lgor batch FOLDER -o OUT [--timeout H]       resumable per-model pipeline with JSONL log
    python -m lgor estimate MODEL.lgo                      run-time estimate

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
            print(f"\r  {name}: {i}/{n} sectors  ({time.time() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
            if i == n:
                print(file=sys.stderr)
    return f


def _ckpt_args(args, lgo: Path, model_name: str = None, outdir: Path = None):
    """(checkpoint dir, log file) from --checkpoint/--log; 'auto' = NAME.ckpt / NAME.progress.log next to the results."""
    ck = getattr(args, "checkpoint", None)
    lg = getattr(args, "log", None)
    stem = model_name or lgo.stem
    d = outdir or lgo.parent
    if ck == "auto":
        ck = d / (stem + ".ckpt")
    if ck and not lg:
        lg = d / (stem + ".progress.log")
    return (Path(ck) if ck else None), (Path(lg) if lg else None)


def _finish_ckpt(args, ck):
    if ck and ck.exists() and not getattr(args, "keep_checkpoint", False):
        import shutil
        shutil.rmtree(ck, ignore_errors=True)


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
            ck, lg = _ckpt_args(args, lgo, outdir=outdir)
            spec = compute_model(model, processes=args.jobs, check=not args.no_check,
                                 progress=_progress(lgo.name), checkpoint=ck, log=lg)
            dt = time.time() - t
            _finish_ckpt(args, ck)
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
    lgo = Path(args.model)
    out = Path(args.out) if args.out else lgo.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "".join(c if c.isalnum() or c in "-_+" else "_" for c in mm.name)
    jsA, jsB = _result_path(lgo), stem.with_name(stem.name + ".lgor.json")
    if args.from_json and jsA.exists():
        print(f"model: using saved spectrum {jsA.name}")
        specA = results.load(jsA)
    else:
        ckA, lgA = _ckpt_args(args, lgo)
        specA = compute_model(model, processes=args.jobs, progress=_progress(lgo.stem), checkpoint=ckA, log=lgA)
        _finish_ckpt(args, ckA)
    if args.from_json and jsB.exists():
        print(f"mirror: using saved spectrum {jsB.name}")
        specB = results.load(jsB)
    else:
        ckB, lgB = _ckpt_args(args, lgo, model_name=stem.name, outdir=out)
        specB = compute_model(mm, processes=args.jobs, progress=_progress(stem.name), checkpoint=ckB, log=lgB)
        _finish_ckpt(args, ckB)
    symA, symB = SymmetryAnalysis(specA), SymmetryAnalysis(specB)
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


def cmd_enumerative(args):
    from .enumerative import enumerative_text
    status = 0
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        if not model.is_fermat:
            print(f"{Path(lgo).name}: W is not Fermat; `enumerative` needs the Fermat/Gepner description")
            status = 1
            continue
        try:
            txt = enumerative_text(model, args.factor - 1, args.r, numerics=args.numerics, h11=args.h11)
        except ValueError as e:
            print(f"{lgo.name}: {e}")
            status = 1
            continue
        tag = f".enumerative_f{args.factor}_r{args.r}.txt"
        out = Path(args.out) / (lgo.stem + tag) if args.out else lgo.with_name(lgo.stem + tag)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt + "\n")
        print(txt)
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


# ----------------------------------------------------------------------------- 2.7
def _write_out(args, lgo: Path, suffix: str, txt: str):
    out = Path(args.out) / (lgo.stem + suffix) if getattr(args, "out", None) else lgo.with_name(lgo.stem + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print(txt)
    print(f"written to {out}")


def cmd_toroidal(args):
    import json
    from .toroidal import report, summary_dict
    rows = []
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        txt = report(model, max_factors=args.max_factors, family=args.family, processes=args.jobs)
        _write_out(args, lgo, ".toroidal.txt", txt)
        if args.json:
            rows.append(summary_dict(model, args.max_factors))
    if args.json:
        with open(args.json, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"summary appended to {args.json}")
    return 0


def cmd_orbifold(args):
    from .orbifold import write, parse_gens, LIST29
    out = args.out or "."
    if args.all:
        for blocks, gens in LIST29:
            print(write(blocks, gens, out))
        return 0
    if not args.blocks or not args.gens:
        print("give BLOCKS and GENS (e.g. CBA 121) or --all")
        return 1
    print(write(args.blocks, parse_gens(" ".join(args.gens)), out, args.name))
    return 0


def cmd_family(args):
    from .family import family_text
    status = 0
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        if args.torus:
            from .toroidal import detect
            ph, note = detect(model)
            if ph is None:
                print(f"{lgo.name}: no toroidal phase: {note}")
                status = 1
                continue
            m, terms = ph.model, ph.deform_terms()
        else:
            m, terms = model, [t.strip() for t in (args.terms or "").split(";") if t.strip()]
        if not terms:
            print("give --terms 'T1; T2' or --torus")
            return 1
        txt = family_text(m, terms, singles=not args.no_singles, processes=args.jobs,
                          max_factors=args.max_factors, spectra=not args.lines_only)
        _write_out(args, lgo, ".family.txt", txt)
    return status


def cmd_lineonly(args):
    from .lineonly import lineonly_text
    from .deform import parse_terms
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        mods = []
        if args.torus:
            from .toroidal import detect
            ph, note = detect(model)
            if ph is None or ph.side != "model":
                print(f"{lgo.name}: no polynomial torus moduli ({note})")
                return 1
            mods = [mm for _, mm in ph.moduli()]
        elif args.moduli:
            mods = [parse_terms(model, t)[0].powers for t in args.moduli.split(";") if t.strip()]
        txt = lineonly_text(model, args.factor - 1, args.r, args.points, mods, args.insert, limit=args.limit)
        _write_out(args, lgo, f".lineonly_f{args.factor}_r{args.r}_n{args.points}.txt", txt)
    return 0


def cmd_perms(args):
    from .perms import perms_text
    from .deform import parse_terms
    for lgo in _expand(args.paths):
        model = Model.from_file(lgo)
        mons = [parse_terms(model, t)[0].powers for t in (args.terms or "").split(";") if t.strip()]
        print(perms_text(model, mons, "deformed" if mons else ""))
    return 0


def cmd_batch(args):
    from .batch import run_batch
    steps = args.steps.split(",") if args.steps else None
    rows = run_batch(_expand(args.paths), Path(args.out), steps or ["run", "mirror", "tdl", "lines", "toroidal"],
                     jobs=args.jobs, timeout=args.timeout * 3600 if args.timeout else None)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows)} steps run, {len(bad)} not ok; log in {Path(args.out) / 'batch.jsonl'}")
    return 1 if bad else 0


def cmd_estimate(args):
    from .batch import estimate_text
    for lgo in _expand(args.paths):
        print(estimate_text(Model.from_file(lgo), args.samples, args.jobs, mirror=not args.no_mirror))
    return 0






def _parse_terms(specs, model):
    """'FERMI COEFF e1 ... en' -> (name, Fraction, powers)."""
    from fractions import Fraction
    out = []
    for sp in specs or []:
        tok = sp.replace(",", " ").split()
        if len(tok) != 2 + model.nphi:
            raise SystemExit(f"--deform '{sp}': need FERMI COEFF and {model.nphi} exponents")
        out.append((tok[0], Fraction(tok[1]), tuple(int(x) for x in tok[2:])))
    return out
















def cmd_hetorbifold(args):
    from . import hetorb
    for tw in args.twists:
        v, name = hetorb.parse_twist(tw)
        text, rows, tot, spec = hetorb.summarise(v, name)
        print(text)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            base = os.path.join(args.out, "hetorb_" + name.replace("/", "_").replace(",", "_"))
            open(base + ".txt", "w").write(text + "\n")
            hetorb.write_csv(rows, base + ".states.csv")
            hetorb.write_tex(rows, base + ".states.tex", name)
            print("wrote %s.{txt,states.csv,states.tex}" % base)
        print()


def cmd_fixedpoints(args):
    from . import fixedpoints
    prefix = args.prefix
    for ext in (".states.csv", ".lgor.json", ".lgo"):
        if prefix.endswith(ext):
            prefix = prefix[: -len(ext)]
    text, _ = fixedpoints.run(prefix, args.blocks, args.twist, rsummary=args.rsummary)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print("wrote", args.out)


def cmd_grouplines(args):
    from . import grouplines as gl
    out = []
    for b in args.block or []:
        ns = tuple(int(x) for x in b.split(","))
        rep, txt = gl.analyse_block(ns, nmax=args.nmax)
        out.append(txt)
    if args.models:
        txt, _ = gl.analyse_models(args.models, nmax=args.nmax)
        out.append(txt)
    text = "\n".join(out)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print("wrote", args.out)


def cmd_gtlines(args):
    import json
    from . import gtlines as gt
    chats = tuple(int(x) for x in args.chat.split(","))
    out, allrows, cache = [], [], {}
    for b in args.block or []:
        rep, txt = gt.analyse_block(tuple(int(x) for x in b.split(",")), nmax=args.nmax, max_states=args.max_states)
        out.append(txt)
    for m in args.models:
        mirror = None if args.no_mirror else (args.mirror or "auto")
        txt, rows, cache = gt.analyse_model(m, mirror=mirror, nmax=args.nmax, chats=chats, kmax=args.kmax,
                                            max_states=args.max_states, cache=cache,
                                            only=[tuple(int(x) for x in o.split(",")) for o in args.only] if args.only else None)
        out.append(txt)
        allrows += rows
    if args.states:
        if not (len(args.models) == 1 and args.mirror):
            raise SystemExit("--states needs one model and --mirror (the LGOR runs of both sides)")
        bf = [int(x) - 1 for x in args.block_fields.split(",")]
        txt, _ = gt.compare_states(args.models[0], args.states[0], args.mirror, args.states[1], bf, nmax=min(args.nmax, 4))
        out.append(txt)
    text = "\n".join(out)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print("wrote", args.out)
    if args.json:
        json.dump([{k: (str(v) if not isinstance(v, (bool, int, str, type(None), list)) else v) for k, v in r.items()}
                   for r in allrows], open(args.json, "w"), indent=1, default=str)
        print("wrote", args.json)


def cmd_gtclosure(args):
    from .gtlines.closure import closure_report
    out = []
    for b in args.blocks:
        ns = tuple(int(x) for x in b.split(","))
        if getattr(args, "complete", False):
            from .gtlines.heisenberg import completed_closure
            txt, _ = completed_closure(ns, nmax=args.nmax, heis=not args.no_heis)
        else:
            txt, _ = closure_report(ns, nmax=args.nmax)
        out.append(txt)
    txt = "\n\n".join(out)
    print(txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt + "\n")
    return 0


def cmd_gtsurvey(args):
    from .gtlines.survey import run_survey
    names = []
    for x in args.names:
        if not x.endswith(".lgo") and os.path.isfile(x):        # a list file (any extension)
            names += [l.split()[0] for l in open(x) if l.strip() and not l.startswith("#")]
        else:
            names.append(x[:-4] if x.endswith(".lgo") else x)
    print(run_survey(args.folder, names, args.out, nmax=args.nmax, log=lambda m: print(m, flush=True)))


def cmd_newlines(args):
    from .newlines import runner
    parts = args.only.split(",") if args.only else ("discfull", "permlines", "fplines", "mfdefects")
    for path in args.paths:
        txt = runner.run(path, parts=parts, mirror=not args.no_mirror, max_exponent=args.max_exponent)
        print(txt)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            stem = os.path.basename(path).replace(".lgor.json", "").replace(".lgo", "")
            fn = os.path.join(args.out, stem + ".newlines.txt")
            open(fn, "w").write(txt + "\n")
            print("wrote", fn)


def cmd_mirrorlines(args):
    from . import mirrorlines as ml
    if args.states:
        bf = [int(x) - 1 for x in args.block_fields.split(",")]
        txt, _ = ml.compare_lgor_states(args.original, args.states[0], args.mirror, args.states[1], bf,
                                        nmax=min(args.nmax, 4))
    else:
        txt, _ = ml.compare_pair(args.original, args.mirror, nmax=args.nmax)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt + "\n")
        print("wrote", args.out)


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
    r.add_argument("--checkpoint", nargs="?", const="auto", metavar="DIR",
                   help="save every finished sector and resume from saved ones (default DIR: NAME.ckpt)")
    r.add_argument("--log", metavar="FILE", help="progress log, one line per sector (default with --checkpoint: NAME.progress.log)")
    r.add_argument("--keep-checkpoint", action="store_true", help="keep the checkpoint folder after success")
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
    mi.add_argument("--checkpoint", nargs="?", const="auto", metavar="DIR",
                   help="save every finished sector and resume from saved ones (default DIR: NAME.ckpt)")
    mi.add_argument("--log", metavar="FILE", help="progress log, one line per sector (default with --checkpoint: NAME.progress.log)")
    mi.add_argument("--keep-checkpoint", action="store_true", help="keep the checkpoint folder after success")
    mi.add_argument("--from-json", action="store_true", help="reuse saved spectra (MODEL.lgor.json, MIRROR.lgor.json) instead of recomputing")
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

    en = sub.add_parser("enumerative", help="lines vs enumerative data: brane charges, Kahler couplings at the "
                        "Gepner point, survival along Kahler moduli, one-parameter periods/GV/monodromies")
    en.add_argument("paths", nargs="+", help=".lgo files or folders")
    en.add_argument("--factor", type=int, required=True, help="factor (1-based) whose Verlinde line is used")
    en.add_argument("--r", type=int, required=True, help="line label r (0 < r < k)")
    en.add_argument("--numerics", action="store_true", help="one-parameter numerics (five fields, h11 = 1; needs mpmath)")
    en.add_argument("--h11", type=int, help="h11 of the model (only for the bookkeeping notes)")
    en.add_argument("-o", "--out", help="folder for MODEL.enumerative_fF_rR.txt (default: next to the input)")
    en.set_defaults(func=cmd_enumerative)

    to = sub.add_parser("toroidal", help="toroidal orbifold phase (model or mirror): Gamma, torus moduli, remnant "
                        "symmetries at the Gepner point and finite size, surviving lines")
    to.add_argument("paths", nargs="+")
    to.add_argument("--family", action="store_true", help="also compute spectra/gauge/discrete along T1, T2, T3, T1+T2+T3")
    to.add_argument("--max-factors", type=int, default=2, help="products of at most this many lines (default 2)")
    to.add_argument("--json", help="append a one-line JSON summary per model to this file")
    to.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    to.add_argument("-o", "--out", help="folder for MODEL.toroidal.txt")
    to.set_defaults(func=cmd_toroidal)

    ho = sub.add_parser("hetorbifold", help="geometric heterotic T6/Z_N orbifold, standard embedding: massless "
                        "spectrum by twisted sector (27, 27bar, singlets, SU(3) multiplets, U(1)^2 charges)")
    ho.add_argument("twists", nargs="+", help="Z3, Z4, Z6-I, Z6-II, Z7, ... or a twist like 1/6,1/3,-1/2")
    ho.add_argument("-o", "--out", help="folder for hetorb_NAME.{txt,states.csv,states.tex}")
    ho.set_defaults(func=cmd_hetorbifold)
    fx = sub.add_parser("fixedpoints", help="3.13: toroidal LGO vs geometric T6/Z_N state by state: orbifold sector of "
                        "every state, Rep(S3)/A4 fixed points of the C (G2) block, B-block label test, R-symmetry "
                        "quartets and fit to plane rotations x gauge")
    fx.add_argument("prefix", help="LGOR run prefix of the Fermat model (PREFIX.states.csv, .lgor.json, .lgo)")
    fx.add_argument("--blocks", required=True, help="field indices per block, 0-based, e.g. '0,3;1,2;4,5,6'")
    fx.add_argument("--twist", required=True, help="orbifold twist in BLOCK order, e.g. '1/6,-1/2,1/3'")
    fx.add_argument("--rsummary", help="summary file with the line 'Q_R = a k0 - b q- mod M' (e.g. the run with "
                    "the torus moduli on); enables the R-symmetry checks")
    fx.add_argument("-o", "--out")
    fx.set_defaults(func=cmd_fixedpoints)

    gq = sub.add_parser("grouplines", help="(3.7 name, c_hat = 1 only; use gtlines) group-theoretical (space-group / class-algebra) lines of c=3 Fermat blocks "
                        "from 2D twisted-twined indices, TFT-certified, with their action under the gauged group")
    gq.add_argument("models", nargs="*", help=".lgo files (Fermat)")
    gq.add_argument("--block", action="append", help="analyse a single block, e.g. --block 6,3")
    gq.add_argument("--nmax", type=int, default=5, help="largest point number in the TFT certification (3..5)")
    gq.add_argument("-o", "--out")
    gq.set_defaults(func=cmd_grouplines)

    gt = sub.add_parser("gtlines", help="group-theoretical (space-group / class-algebra) lines of ANY LGO: blocks of "
                        "integer c_hat, 2D fixed-point tables, TFT certification in the B and A pictures, existence as "
                        "B-/A-type in the model and its BHK mirror, mirror agreement")
    gt.add_argument("models", nargs="*", help=".lgo files")
    gt.add_argument("--mirror", help="mirror .lgo (default: build the BHK mirror automatically)")
    gt.add_argument("--no-mirror", action="store_true")
    gt.add_argument("--chat", default="1,2,3", help="block central charges c_hat to scan (default 1,2,3)")
    gt.add_argument("--nmax", type=int, default=5, help="largest point number in the TFT certification (3..5)")
    gt.add_argument("--kmax", type=int, default=60, help="skip blocks with |J_S| > kmax")
    gt.add_argument("--max-states", type=int, default=150, help="certify only blocks with at most this many chiral states")
    gt.add_argument("--block", action="append", help="analyse a single Fermat block, e.g. --block 6,3")
    gt.add_argument("--only", action="append", help="restrict a model run to blocks with these exponents, e.g. --only 6,3")
    gt.add_argument("--states", nargs=2, metavar=("CSV_ORIG", "CSV_MIRROR"),
                    help="LGOR .states.csv of model and mirror: line action on the Fermat-phase 27/27bar")
    gt.add_argument("--block-fields", default="1,4", help="fields of the c_hat = 1 block for --states, 1-based")
    gt.add_argument("-o", "--out")
    gt.add_argument("--json")
    gt.set_defaults(func=cmd_gtlines)

    gc = sub.add_parser("gtclosure", help="3.12: fusion closure of the certified gtlines of Fermat blocks; flags "
                        "products that need simple lines outside the certified list")
    gc.add_argument("blocks", nargs="+", help="Fermat block exponents, e.g. 6,3 4,4")
    gc.add_argument("--nmax", type=int, default=4)
    gc.add_argument("--complete", action="store_true", help="3.14: add the full phase x permutation group, the "
                    "Verlinde lines of the factors and the Heisenberg orbit lines; exact MILP decompositions (scipy)")
    gc.add_argument("--no-heis", action="store_true", help="with --complete: leave out the Heisenberg lines")
    gc.add_argument("-o", "--out")
    gc.set_defaults(func=cmd_gtclosure)

    gs = sub.add_parser("gtsurvey", help="gtlines + Fermat-phase spectrum mirror check + Kähler survival of the group-"
                        "theoretical lines for many models (needs NAME.states.csv and NAME_mirror.states.csv)")
    gs.add_argument("folder", help="folder with NAME.lgo, NAME.states.csv, NAME_mirror.lgo, NAME_mirror.states.csv")
    gs.add_argument("names", nargs="+", help="model names, or a .txt file with one name per line")
    gs.add_argument("-o", "--out", required=True)
    gs.add_argument("--nmax", type=int, default=5)
    gs.set_defaults(func=cmd_gtsurvey)

    for nm, only, hlp in (("newlines", None, "3.9: full discrete group + mirror embedding, permutation lines, fixed-point lines, MF defects"),
                          ("discfull", "discfull", "3.9: full abelian discrete group (prime powers) and mirror embedding check"),
                          ("permlines", "permlines", "3.9: non-invertible lines from permutations not normalising the orbifold group"),
                          ("fplines", "fplines", "3.9: fixed-point resolution lines of the simple-current extension"),
                          ("mfdefects", "mfdefects", "3.9: rank-one matrix-factorisation defects and deformation survival")):
        q = sub.add_parser(nm, help=hlp)
        q.add_argument("paths", nargs="+", help=".lgo or .lgor.json (spectrum must exist; mirror json used if present)")
        q.add_argument("--no-mirror", action="store_true")
        q.add_argument("--max-exponent", type=int, default=12, help="mfdefects: skip factors with larger exponent")
        q.add_argument("-o", "--out", default=None)
        q.set_defaults(func=cmd_newlines, only=only)

    from .mfcat.cli import add_parser as _mfcat_add
    _mfcat_add(sub)

    mlp = sub.add_parser("mirrorlines", help="build the grouplines lines on both sides of a mirror pair (B-type on the "
                         "Jacobian ring, A-type on twisted sectors via FJRW/Chiodo) and check that they agree")
    mlp.add_argument("original")
    mlp.add_argument("mirror")
    mlp.add_argument("--nmax", type=int, default=5)
    mlp.add_argument("--states", nargs=2, metavar=("CSV_ORIG", "CSV_MIRROR"),
                     help="LGOR .states.csv of both runs: compare the line action on the full Fermat-phase 27/27bar spectrum")
    mlp.add_argument("--block-fields", default="1,4", help="fields of the (6,3) block, 1-based (with --states)")
    mlp.add_argument("-o", "--out")
    mlp.set_defaults(func=cmd_mirrorlines)

    ob = sub.add_parser("orbifold", help="write the LGO whose GP mirror is T^6/Gamma (blocks A/B/C + Gamma generators)")
    ob.add_argument("blocks", nargs="?", help="three letters: A (Z3 torus), B (Z4 torus), C (Z6 torus)")
    ob.add_argument("gens", nargs="*", help="Gamma generators in rotation units, e.g. 121 or 022 202")
    ob.add_argument("--all", action="store_true", help="write the 29 orbifolds of the 2026-10-30 study")
    ob.add_argument("--name", help="model name (default T_BLOCKS_G|Gamma|_gens)")
    ob.add_argument("-o", "--out", help="output folder")
    ob.set_defaults(func=cmd_orbifold)

    fa = sub.add_parser("family", help="deform by each modulus alone and all together: spectrum, gauge group, "
                        "discrete symmetries, surviving lines")
    fa.add_argument("paths", nargs="+")
    fa.add_argument("--terms", help="moduli in deform syntax separated by ';'")
    fa.add_argument("--torus", action="store_true", help="use the torus size moduli of `toroidal`")
    fa.add_argument("--no-singles", action="store_true", help="only the point with all moduli on")
    fa.add_argument("--lines-only", action="store_true", help="skip the spectrum computations")
    fa.add_argument("--max-factors", type=int, default=2)
    fa.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    fa.add_argument("-o", "--out")
    fa.set_defaults(func=cmd_family)

    lo = sub.add_parser("lineonly", help="couplings forbidden by one line only (after Fermat phases and u1_forced)")
    lo.add_argument("paths", nargs="+")
    lo.add_argument("--factor", type=int, required=True, help="factor (1-based)")
    lo.add_argument("--r", type=int, required=True, help="line label, 0 < r < k")
    lo.add_argument("--points", type=int, default=4, help="number of marginal 27 entries (default 4)")
    lo.add_argument("--torus", action="store_true", help="insert the torus size moduli")
    lo.add_argument("--moduli", help="moduli to insert, deform syntax separated by ';'")
    lo.add_argument("--insert", type=int, default=2, help="max insertions of each modulus (default 2)")
    lo.add_argument("--limit", type=int, help="stop after this many genuine candidates")
    lo.add_argument("-o", "--out")
    lo.set_defaults(func=cmd_lineonly)

    pe = sub.add_parser("perms", help="permutation symmetries normalising the orbifold group (R-type flagged)")
    pe.add_argument("paths", nargs="+")
    pe.add_argument("--terms", help="deformation terms to preserve, ';'-separated")
    pe.set_defaults(func=cmd_perms)

    ba = sub.add_parser("batch", help="run steps per model with timeouts and checkpoints; JSONL log; resumable")
    ba.add_argument("paths", nargs="+")
    ba.add_argument("-o", "--out", required=True, help="output folder (one subfolder per model)")
    ba.add_argument("--steps", help="comma list, default run,mirror,tdl,lines,toroidal (also defects, blocklines)")
    ba.add_argument("--timeout", type=float, help="hours per step")
    ba.add_argument("-j", "--jobs", type=int, default=1, help="processes for run/mirror/lines")
    ba.set_defaults(func=cmd_batch)

    es = sub.add_parser("estimate", help="time sample sectors and extrapolate the run time (model and mirror)")
    es.add_argument("paths", nargs="+")
    es.add_argument("--samples", type=int, default=8)
    es.add_argument("--no-mirror", action="store_true")
    es.add_argument("-j", "--jobs", type=int, default=1, help="processes the real run would use")
    es.set_defaults(func=cmd_estimate)

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
