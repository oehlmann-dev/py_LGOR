"""gtsurvey: for a list of LGOs with LGOR runs of the model and its mirror (NAME.lgo, NAME.states.csv,
NAME_mirror.lgo, NAME_mirror.states.csv):
  1. gtlines on both sides (TFT certification in both pictures, B/A existence, B(X)=A(Y), A(X)=B(Y));
  2. Fermat-phase spectrum check for every c_hat = 1 block: eigenvalues of each line on the 27 / 27bar of X in the
     B (A) reading equal those on the 27bar / 27 of Y in the A (B) reading;
  3. Kähler survival: for every line that exists in a model (as B- or A-type), its eigenvalue on every 27bar
     (Kähler modulus, incl. twisted = blow-up modes).  Kept = eigenvalue equal to the quantum dimension d
     (transparent; Cordova-Rizi criterion for exactly marginal deformations).  27bars are split into 'bulk'
     (from a sector that twists every field: volume-type) and 'fixed-locus' (some field untwisted: blow-up /
     exceptional-divisor type) [inferred classification; the GLSM decides].
       CALIBRATION (2026-11-13): on the Verlinde line L_(2,0) the eigenvalue counts agree with the exact label
       method (lgor.tdl) only for the B reading of the model X and the A reading of the mirror Y (the two tables
       compared by the spectrum mirror check); the other two readings are reported as UNCALIBRATED, and the
       bulk / fixed-locus split is not used for verdicts (only FULL / PARTIAL / NONE).
       FULL          kept by every Kähler modulus: survives the whole Kähler moduli space, incl. smooth large volume
       SMOOTH-LOCUS  kept by every fixed-locus modulus and by some bulk ones: survives on a sub-locus where all
                     blow-ups are on, i.e. smooth, at special values of the broken bulk moduli [inferred]
       BULK          kept by all bulk moduli, broken by some blow-up modes: any volume, but singular
       PARTIAL/NONE  kept by some / none.
"""
import csv
import os
from collections import Counter

from ..model import Model
from .blocks import find_blocks
from .pipeline import analyse_model
from .states import lgor_line_table


def _kept(cnt, kind, d):
    c = cnt[kind]
    tot = sum(v for k, v in c.items() if not isinstance(k, tuple))
    unp = sum(v for k, v in c.items() if isinstance(k, tuple))
    return c.get(round(float(d), 6), 0), tot, unp


def survey_model(folder, name, nmax=5, cache=None):
    lx, ly = os.path.join(folder, name + ".lgo"), os.path.join(folder, name + "_mirror.lgo")
    cx, cy = os.path.join(folder, name + ".states.csv"), os.path.join(folder, name + "_mirror.states.csv")
    X, Y = Model.from_file(lx), Model.from_file(ly)
    txt, rows, cache = analyse_model(X, mirror=Y, nmax=nmax, cache=cache)
    out = [txt, "", "FERMAT-PHASE SPECTRUM CHECK AND KÄHLER SURVIVAL (c_hat = 1 blocks)"]
    res = []
    have_csv = os.path.exists(cx) and os.path.exists(cy)
    if not have_csv:
        out.append("  no states.csv for model and mirror: skipped")
    blocks, _ = find_blocks(X, chats=(1,))
    for b in blocks:
        if not b.fermat or b.key not in cache:
            continue
        rep = cache[b.key][0]
        cands = [(c["SG"], c["labels"], c["tsec"], c["tcls"], c["certB"]) for c in rep["candidates"]
                 if c.get("status", "").startswith("certified")]
        if not cands or not have_csv:
            continue
        T = {}
        for side, lgo, cs in (("X", lx, cx), ("Y", ly, cy)):
            for pic, swap in (("B", False), ("A", True)):
                T[side + pic] = lgor_line_table(lgo, cs, b.fields, krawitz_swap=swap, cands=cands)[1]
        fl = tuple(f + 1 for f in b.fields)
        out.append("  block fields %s %s" % (list(fl), b.ns))
        for nm, cnt in T["XB"].items():
            g, si, d = cnt["meta"]
            row = next((r for r in rows if r.get("fields") == fl and r.get("group") == g and r.get("solution") == si), None)
            if row is None:
                continue
            ok1 = T["XB"][nm]["27"] == T["YA"][nm]["27bar"] and T["XB"][nm]["27bar"] == T["YA"][nm]["27"]
            ok2 = T["XA"][nm]["27"] == T["YB"][nm]["27bar"] and T["XA"][nm]["27bar"] == T["YB"][nm]["27"]
            r = {"model": name, "fields": fl, "block": b.ns, "group": g, "solution": si, "d": d,
                 "kind": "; ".join(k for dd, k in row["lines"] if dd == d),
                 "B_X": row["B_X"], "A_X": row["A_X"], "B_Y": row["B_Y"], "A_Y": row["A_Y"],
                 "tft_mirror_agree": row["mirror_agree"], "spectrum_mirror_agree": ok1 and ok2}
            for side in ("X", "Y"):
                verdicts = []
                for pic in ("B", "A"):
                    if not row[pic + "_" + side]:
                        continue
                    cnt2 = T[side + pic][nm]
                    k, tot, unp = _kept(cnt2, "27bar", d)
                    k1, tot1, _ = _kept(cnt2, "27bar_T1", d)
                    kf, totf = k - k1, tot - tot1           # fixed-locus (blow-up type) moduli
                    if not tot or unp:
                        pass
                    if (side == "X" and X.orders) or (side == "Y" and Y.orders):
                        split_ok = False            # extra orbifold: all-twisted sectors can be fixed-point blow-ups
                    else:
                        split_ok = True
                    if not tot or unp:
                        v = "UNDECIDED" if unp else "NONE"
                    elif k == tot:
                        v = "FULL"
                    elif split_ok and kf == totf and k1 >= 1:
                        v = "SMOOTH-LOCUS"
                    elif split_ok and k1 == tot1 and tot1:
                        v = "BULK"
                    else:
                        v = "PARTIAL" if k else "NONE"
                    if v in ("SMOOTH-LOCUS", "BULK"):
                        v = "PARTIAL"           # bulk / fixed-locus split failed calibration (2026-11-13): not used
                    if (side, pic) not in (("X", "B"), ("Y", "A")):
                        v = "UNCALIBRATED"      # this reading disagrees with the Verlinde-label check (L_(2,0) test)
                    verdicts.append("%s-type: Kähler kept %d/%d (bulk %d/%d, fixed-locus %d/%d)%s -> %s" % (
                        pic, k, tot, k1, tot1, kf, totf, (", %d unpaired" % unp) if unp else "", v))
                    r["kahler_" + side + pic] = v
                r["kahler_" + side] = " | ".join(verdicts) if verdicts else "line does not exist"
            res.append(r)
            out.append("    %-22s sol %d d=%d %-45s B/A in X %s/%s, in Y %s/%s; TFT mirror %s, spectrum mirror %s" % (
                g, si, d, r["kind"][:45], _yn(r["B_X"]), _yn(r["A_X"]), _yn(r["B_Y"]), _yn(r["A_Y"]),
                _ok(r["tft_mirror_agree"]), _ok(r["spectrum_mirror_agree"])))
            out.append("        X: %s" % r["kahler_X"])
            out.append("        Y: %s" % r["kahler_Y"])
    # Verlinde (su(2)_k product) lines: survival on all Kähler moduli (lgor.tdl.line_survival, as in `lines`)
    from .. import results
    from ..tdl import line_survival
    verl = {}
    for side, lgo in (("X", lx), ("Y", ly)):
        js = lgo[:-4] + ".lgor.json"
        try:
            ls = line_survival(results.load(js))
            verl[side] = {"full": ls["full"]["27bar"], "n27bar": ls["n27bar"], "breaks_all": ls["kahler_breaks_all"]}
            out.append("  Verlinde lines (%s): factors whose lines are kept by ALL %d Kähler moduli: %s; Kähler moduli "
                       "breaking every non-invertible line: %d" % (side, ls["n27bar"], ls["full"]["27bar"] or "none",
                                                                  ls["kahler_breaks_all"]))
        except Exception as e:
            verl[side] = {"error": str(e)}
            out.append("  Verlinde lines (%s): not evaluated (%s)" % (side, e))
    mc = os.path.join(folder, name + "_mirror.mirrorcheck.txt")
    mres = "no mirrorcheck file"
    if os.path.exists(mc):
        ln = [l.strip() for l in open(mc) if "RESULT" in l]
        mres = ln[-1] if ln else "mirrorcheck without RESULT line"
    out.insert(1, "LGOR spectrum mirror check: " + mres)
    res_meta = {"verlinde": verl, "mirrorcheck": mres}
    return "\n".join(out), rows, res, cache, res_meta


def _yn(x):
    return "?" if x is None else ("y" if x else "n")


def _ok(x):
    return "ok" if x else "FAIL"


FIELDS = ["model", "fields", "block", "group", "solution", "d", "kind", "B_X", "A_X", "B_Y", "A_Y",
          "tft_mirror_agree", "spectrum_mirror_agree", "kahler_XB", "kahler_XA", "kahler_YB", "kahler_YA",
          "kahler_X", "kahler_Y"]


def run_survey(folder, names, outdir, nmax=5, log=print):
    os.makedirs(outdir, exist_ok=True)
    cache = {}
    allres, status, verl_full = [], [], []
    csvp = os.path.join(outdir, "gtsurvey.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, nm in enumerate(names):
            try:
                txt, rows, res, cache, meta = survey_model(folder, nm, nmax=nmax, cache=cache)
            except Exception as e:                     # keep going, record the failure
                status.append((nm, "ERROR %s: %s" % (type(e).__name__, e)))
                log("[%d/%d] %s ERROR %s" % (i + 1, len(names), nm, e))
                continue
            open(os.path.join(outdir, nm + ".gtsurvey.txt"), "w").write(txt + "\n")
            for r in res:
                w.writerow(r)
            fh.flush()
            allres += res
            nsol = sum(1 for r in rows if "solution" in r)
            bad = sum(1 for r in rows if "solution" in r and not r.get("mirror_agree"))
            vf = ["%s:%s" % (sd, meta["verlinde"][sd].get("full")) for sd in ("X", "Y")
                  if meta["verlinde"].get(sd, {}).get("full")]
            if vf:
                verl_full.append((nm, vf))
            mirror_ok = "identical symmetry content" in meta["mirrorcheck"] or "are mirror pairs" in meta["mirrorcheck"]
            status.append((nm, ("spectra mirror %s; " % ("ok" if mirror_ok else "NOT OK (" + meta["mirrorcheck"] + ")")) + "%d lines, TFT mirror disagreements %d, spectrum checks %d (fail %d), FULL Kähler %d" % (
                nsol, bad, len(res), sum(1 for r in res if not r["spectrum_mirror_agree"]),
                sum(1 for r in res if "FULL" in (r.get("kahler_XB", ""), r.get("kahler_XA", ""),
                                                  r.get("kahler_YB", ""), r.get("kahler_YA", ""))))))
            log("[%d/%d] %s: %s" % (i + 1, len(names), nm, status[-1][1]))
    summ = ["gtsurvey over %d models (folder %s)" % (len(names), folder)]
    summ += ["  %-28s %s" % s for s in status]
    full = [r for r in allres if "FULL" in (r.get("kahler_XB"), r.get("kahler_XA"), r.get("kahler_YB"), r.get("kahler_YA"))]
    vs = lambda r: (r.get("kahler_XB"), r.get("kahler_XA"), r.get("kahler_YB"), r.get("kahler_YA"))
    smooth = [r for r in allres if "SMOOTH-LOCUS" in vs(r) and r not in full]
    bulk = [r for r in allres if "BULK" in vs(r) and r not in full]
    summ += ["", "TOTAL: %d block lines checked on spectra; spectrum mirror failures %d; TFT mirror failures %d" % (
        len(allres), sum(1 for r in allres if not r["spectrum_mirror_agree"]), sum(1 for r in allres if not r["tft_mirror_agree"])),
             "  lines kept by ALL Kähler moduli of a side (FULL): %d" % len(full),
             "  lines on a smooth sub-locus (SMOOTH-LOCUS): %d" % len(smooth),
             "  lines kept by all bulk moduli only (BULK, singular): %d" % len(bulk)]
    summ.append("  Verlinde lines kept by ALL Kähler moduli (factor lists): %s" % (verl_full or "none"))
    for r in smooth[:50]:
        summ.append("    SMOOTH-LOCUS: %s fields %s %s sol %d (%s) X: %s; Y: %s" % (
            r["model"], list(r["fields"]), r["group"], r["solution"], r["kind"], r["kahler_X"], r["kahler_Y"]))
    for r in full[:50]:
        summ.append("    FULL: %s fields %s %s sol %d (%s) X: %s; Y: %s" % (r["model"], list(r["fields"]), r["group"],
                                                                         r["solution"], r["kind"], r["kahler_X"], r["kahler_Y"]))
    open(os.path.join(outdir, "SUMMARY.txt"), "w").write("\n".join(summ) + "\n")
    return "\n".join(summ)
