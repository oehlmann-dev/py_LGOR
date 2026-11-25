"""gtlines pipeline: blocks -> 2D fixed-point tables -> candidate groups -> TFT certification in both pictures ->
existence of each line as B-type / A-type in the model and in its BHK mirror -> mirror agreement."""
import itertools
from fractions import Fraction as Fr

from .fixedpoints import fixed_point_table, orbit_counts, table_signs, block_order, chat as _chat
from .groups import SpaceGroup, candidate_actions
from .tft import (label_states, certify, fjrw_corr, LIT_COUNT, line_on_monomials, verlinde_match, act_on_solution,
                  _same, _basis_str_safe)
from .blocks import Block, find_blocks, gauged_generators, gauge_group


def _block_of(x):
    if isinstance(x, Block):
        return x
    ns = tuple(sorted(x, reverse=True))
    qs = tuple(Fr(1, n) for n in ns)
    return Block(list(range(len(ns))), qs, ns, sum(1 - 2 * q for q in qs), block_order(qs))


def analyse_block(block, nmax=5, a_picture=True, max_states=150, max_tuples=3_000_000):
    """Everything that depends on the block alone.  block: Block or a tuple of Fermat exponents.
    Returns a dict (picklable apart from sympy matrices) and a text report."""
    b = _block_of(block)
    src = b.ns if b.fermat else b.qs
    K, tab, ok = fixed_point_table(src)
    rep = {"block": b.key, "K": K, "chat": b.chat, "table": tab, "integer": ok, "flipped_rows": table_signs(src),
           "candidates": []}
    out = ["block %s  (K = %d, c_hat = %s)" % (b.ns if b.fermat else [str(q) for q in b.qs], K, b.chat)]
    out.append("  2D fixed-point table chi(k,l) = #Fix(theta^k) fixed by theta^l  [twisted-twined Witten indices]%s"
               % ("; rows %s sign-flipped [inferred]" % rep["flipped_rows"] if rep["flipped_rows"] else ""))
    for k in range(K):
        oc = orbit_counts(tab[k], K)
        out.append("    k=%-2d %s   orbits %s" % (k, " ".join("%2d" % x for x in tab[k]),
                   " + ".join("%dx%d" % (c, s) for s, c in sorted(oc.items())) if oc else "not a permutation character"))
    if not ok:
        out.append("  table not integral: no candidates")
        return rep, "\n".join(out)
    done = set()
    for kstar in range(1, K):
        oc = orbit_counts(tab[kstar], K)
        if not oc or max(oc) == 1 or tuple(tab[kstar]) in done:
            continue
        done.add(tuple(tab[kstar]))
        for A, sigma in candidate_actions(tab[kstar], K):
            SG = SpaceGroup(A, sigma)
            irr = SG.irreps()
            cand = {"kstar": kstar, "group": SG.name(), "SG": SG, "dims": [d for d, _ in irr]}
            out.append("  candidate from theta^%d fixed points: A = %s, theta of order %d  ->  G = %s, irrep dims %s"
                       % (kstar, A.name(), SG.m, SG.name(), cand["dims"]))
            rep["candidates"].append(cand)
            if not b.fermat:
                cand["status"] = "uncertified (non-Fermat block) [missing]"
                out.append("    not certified: non-Fermat block, no TFT correlators implemented [missing]")
                continue
            if b.nstates > max_states:
                cand["status"] = "uncertified (size)"
                out.append("    not certified: %d chiral states > max_states = %d" % (b.nstates, max_states))
                continue
            labels, tsec, tcls = label_states(b.ns, K, tab, SG)
            if labels is None:
                cand["status"] = "unlabelled"
                out.append("    labels: %s" % tsec)
                continue
            resB = certify(b.ns, SG, labels, tsec, tcls, nmax=nmax, max_tuples=max_tuples)
            cand.update(labels=labels, tsec=tsec, tcls=tcls, certB=resB)
            out.append("    B picture (Saito): charge-allowed couplings %s, class-forbidden %s, of which not phase-zero %s,"
                       " allowed by all invertible lines of G %s; conditions %d%s%s"
                       % (resB["charge_allowed"], resB["forbidden"], resB["nontrivial_forbidden"], resB["line_only"],
                          resB["conditions"], "; truncated at n = %s" % resB["truncated"] if resB["truncated"] else "",
                          "; skipped terms %d" % resB["skipped_terms"] if resB["skipped_terms"] else ""))
            solB = [s for s in resB["solutions"] if not s.get("family")]
            if resB["refuted_by"] or not resB["solutions"]:
                cand["status"] = "refuted"
                out.append("    REFUTED by the block TFT" + (": e.g. %s" % (resB["refuted_by"][0],) if resB["refuted_by"] else
                                                          " (no consistent orbit basis)"))
                continue
            cand["status"] = "certified" if not resB["truncated"] else "certified (truncated search)"
            dup = next((o for o in rep["candidates"][:-1] if o.get("status", "").startswith("certified")
                        and o["dims"] == cand["dims"] and o["tsec"] == tsec
                        and len(o["solB"]) == len(solB) and all(any(_same(x["mats"], y["mats"]) for y in o["solB"]) for x in solB)), None)
            cand["solB"] = solB
            if dup is not None:
                cand["status"] = "duplicate"
                cand["duplicate_of"] = dup["group"]
                out.append("    same orbit bases and irrep dimensions as the candidate %s: same lines, not repeated" % dup["group"])
                continue
            if a_picture:
                LIT_COUNT["n5_single"] = 0
                resA = certify(b.ns, SG, labels, tsec, tcls, nmax=nmax, corr_fn=fjrw_corr, max_tuples=max_tuples)
                cand["certA"], cand["lit"] = resA, LIT_COUNT["n5_single"]
                solA = [s for s in resA["solutions"] if not s.get("family")]
                match = [next((j for j, t in enumerate(solA) if _same(s["mats"], t["mats"])), None) for s in solB]
                cand["matchAB"] = match
                same = None not in match and len(solA) == len(solB)
                out.append("    A picture (FJRW of narrow twisted sectors): %d solutions; same as B under the Krawitz map: %s"
                           "%s" % (len(solA), "yes" if same else "NO %s" % match,
                                   "  [%d five-point terms from r-spin = A_{r-1}, literature]" % cand["lit"] if cand["lit"] else ""))
            fam = [s for s in resB["solutions"] if s.get("family")]
            if fam:
                out.append("    + positive-dimensional family of orbit bases (not listed)")
            cand["lines"] = []
            for si, sol in enumerate(solB):
                out.append("    solution %d: %s" % (si, "; ".join("sector %s: %s" % (k, _basis_str_safe(tsec[k], m))
                                                                for k, m in sol["mats"].items()) or "no unknown basis"))
                ld = []
                for d, ch in irr:
                    if d == 1:
                        continue
                    diag, eig = line_on_monomials(b.ns, SG, labels, tsec, tcls, sol["mats"], ch)
                    vm = verlinde_match(b.ns, eig) if diag else None
                    kind = ("Verlinde L_r, r = %s" % (vm,)) if vm is not None else (
                        "diagonal, not Verlinde" if diag else "mixes monomials of equal charge (not Verlinde)")
                    ld.append((d, kind))
                    out.append("      line d=%d: %s" % (d, kind))
                cand["lines"].append(ld)
    return rep, "\n".join(out)


def _b_exists(b, tsec, mats, sols, gens):
    """B-type: the line commutes with the gauged group (conjugation by every generator fixes the solution)."""
    for _, ph in gens:
        img = act_on_solution(b.ns, tsec, mats, [ph[f] for f in b.fields])
        if not _same(img, mats):
            return False
    return True


def _a_exists(b, nf, tsec, mats, Gset):
    """A-type: every block-supported twist change h = theta' - theta used by the line lies in the gauged group."""
    for key, sts in tsec.items():
        M = mats[key]
        for i in range(M.rows):
            cols = [j for j in range(M.cols) if abs(complex(M[i, j])) > 1e-9]
            for j1, j2 in itertools.combinations(cols, 2):
                h = [Fr(0)] * nf
                for t, f in enumerate(b.fields):
                    h[f] = Fr(sts[j2][t] - sts[j1][t], b.ns[t]) % 1
                if tuple(h) not in Gset:
                    return False
    return True


def _load(m):
    from ..model import Model
    return m if hasattr(m, "fields") else Model.from_file(m)


def analyse_model(model, mirror="auto", nmax=5, chats=(1, 2, 3), kmax=60, max_states=150, max_tuples=3_000_000,
                  cache=None, only=None):
    """Find all group-theoretical lines of an LGO and test them on both mirror sides.
    model, mirror: Model or .lgo path; mirror = 'auto' builds the BHK mirror (lgor.mirror.mirror_model, Fermat W),
    None skips the mirror.  Returns (text, rows, cache) where rows are dicts per (block, candidate, solution)."""
    from ..mirror import mirror_model
    X = _load(model)
    Y = None
    note = ""
    if mirror == "auto":
        try:
            Y = mirror_model(X)
            note = "BHK mirror built automatically"
        except Exception as e:                       # non-Fermat W etc.
            note = "no mirror: %s" % e
    elif mirror is not None:
        Y = _load(mirror)
        note = "mirror from %s" % getattr(Y, "filename", Y.name)
    cache = {} if cache is None else cache
    gX = gauged_generators(X)
    GX = gauge_group(X)
    sides = [("X", X, gX, GX)]
    if Y is not None:
        gY = gauged_generators(Y)
        GY = gauge_group(Y)
        sides.append(("Y", Y, gY, GY))
    out = ["=" * 100, "gtlines: %s   (%s)" % (X.name, note),
           "  |G_X| = %d%s" % (len(GX), (", |G_Y| = %d" % len(GY)) if Y is not None else "")]
    blocks, skipped = find_blocks(X, chats=chats, kmax=kmax)
    if only:
        only = {tuple(sorted(o, reverse=True)) for o in only}
        blocks = [b for b in blocks if b.fermat and tuple(b.ns) in only]
    out.append("  %d blocks with c_hat in %s%s" % (len(blocks), list(chats),
                                                  "; skipped (size): %s" % [(s.label(), r) for s, r in skipped] if skipped else ""))
    rows = []
    for b in blocks:
        if b.key not in cache:
            cache[b.key] = analyse_block(b, nmax=nmax, max_states=max_states, max_tuples=max_tuples)
        rep, txt = cache[b.key]
        out.append("-" * 100)
        out.append("block on %s" % b.label())
        if not rep.get("_printed"):
            out.append(txt)
            rep["_printed"] = True
        else:
            out.append("  (same block type as above: %s)" % (b.ns,))
        nf = X.nfields
        for cand in rep["candidates"]:
            st = cand.get("status", "")
            if st == "duplicate":
                continue
            if not st.startswith("certified"):
                rows.append({"model": X.name, "fields": tuple(f + 1 for f in b.fields), "block": b.ns or b.qs,
                             "group": cand["group"], "status": st})
                continue
            solB = [s for s in cand["certB"]["solutions"] if not s.get("family")]
            solA = [s for s in cand.get("certA", {"solutions": []})["solutions"] if not s.get("family")]
            match = cand.get("matchAB", [None] * len(solB))
            for i, s in enumerate(solB):
                row = {"model": X.name, "fields": tuple(f + 1 for f in b.fields), "block": b.ns, "group": cand["group"],
                       "status": st, "solution": i, "lines": cand["lines"][i]}
                for side, M, gens, Gset in sides:
                    row["B_" + side] = _b_exists(b, cand["tsec"], s["mats"], solB, gens)
                    j = match[i]
                    row["A_" + side] = _a_exists(b, nf, cand["tsec"], solA[j]["mats"], Gset) if j is not None else None
                if Y is not None:
                    row["mirror_agree"] = (row["B_X"] == row["A_Y"]) and (row["B_Y"] == row["A_X"])
                rows.append(row)
            out.append("  %s:" % cand["group"])
            for r in rows:
                if r.get("fields") == tuple(f + 1 for f in b.fields) and r.get("group") == cand["group"] and "solution" in r:
                    out.append("    solution %d: B-type in X %s, A-type in X %s%s" % (
                        r["solution"], _yn(r["B_X"]), _yn(r["A_X"]),
                        ("; B-type in Y %s, A-type in Y %s -> mirror %s" % (_yn(r["B_Y"]), _yn(r["A_Y"]),
                                                                           "AGREE" if r["mirror_agree"] else "DISAGREE"))
                        if Y is not None else ""))
    out.append("=" * 100)
    out.append(summary(rows, Y is not None))
    return "\n".join(out), rows, cache


def _yn(x):
    return "?" if x is None else ("yes" if x else "no")


def summary(rows, with_mirror=True):
    sol = [r for r in rows if "solution" in r]
    lines = ["SUMMARY: %d certified block lines (solutions)" % len(sol)]
    nonv = [r for r in sol if any("not Verlinde" in k for _, k in r["lines"])]
    lines.append("  non-Verlinde (mix monomials or diagonal non-Verlinde): %d" % len(nonv))
    lines.append("  existing in the model: B-type %d, A-type %d, either %d" % (
        sum(bool(r["B_X"]) for r in sol), sum(bool(r["A_X"]) for r in sol), sum(bool(r["B_X"] or r["A_X"]) for r in sol)))
    if with_mirror:
        bad = [r for r in sol if not r["mirror_agree"]]
        lines.append("  mirror agreement B(X)=A(Y), A(X)=B(Y): %s" % ("ALL AGREE" if not bad else "%d DISAGREE" % len(bad)))
    from collections import defaultdict
    per = defaultdict(list)
    for r in sol:
        per[(r["fields"], r["group"])].append(r)
    for (f, g), rs in per.items():
        lines.append("  fields %s, G = %s: %d lines; B-type in X %s; A-type in X %s%s" % (
            list(f), g, len(rs), [r["solution"] for r in rs if r["B_X"]], [r["solution"] for r in rs if r["A_X"]],
            ("; B-type in Y %s; A-type in Y %s" % ([r["solution"] for r in rs if r["B_Y"]],
                                                 [r["solution"] for r in rs if r["A_Y"]])) if with_mirror else ""))
    unc = [r for r in rows if "solution" not in r]
    if unc:
        from collections import Counter
        lines.append("  candidates without certified lines: %s" % dict(Counter(r["status"] for r in unc)))
    return "\n".join(lines)


def compare_states(lgoX, csvX, lgoY, csvY, block_fields, nmax=4):
    """Fermat-phase LGOR spectrum check for one c_hat = 1 block: eigenvalue multisets of every block line on the
    27 / 27bar of X (B picture) and Y (A picture via the Krawitz swap); mirror symmetry needs X-27 = Y-27bar."""
    from .states import lgor_line_table
    tx, mx = lgor_line_table(lgoX, csvX, block_fields, nmax=nmax)
    ty, my = lgor_line_table(lgoY, csvY, block_fields, nmax=nmax, krawitz_swap=True)
    lines, allok = [tx, ty], True
    for k in mx:
        ok = k in my and mx[k]["27"] == my[k]["27bar"] and mx[k]["27bar"] == my[k]["27"]
        allok &= ok
        lines.append("  %-26s X-27 = Y-27bar and X-27bar = Y-27: %s" % (k, "yes" if ok else "NO"))
    lines.append("ALL AGREE" if allok else "MISMATCH")
    return "\n".join(lines), allok
