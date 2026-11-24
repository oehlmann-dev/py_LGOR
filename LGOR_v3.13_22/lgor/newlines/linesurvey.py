"""LGOR 3.10 `linesurvey`: one-line summary of every NON-INVERTIBLE line type of a model and of the moduli
that keep it (27 = complex structure, 27bar = Kähler incl. twisted blow-up modes).

Line types:
  verlinde   core su(2)_k lines L_r of Fermat factors (d > 1) and their products        [verified labels]
  perm       group-theoretical lines C(G,H) from permutations not normalising H (d=|H:K|)  [verified]
  fp         fixed-point-resolved lines (a,+-), d_a/2, of simple-current fixed sets        [literature]
  mf         rank-one B-type defects P_S with survival pattern of no Verlinde line         [verified pattern]
  (gt block lines are read from gtsurvey.csv by the aggregator)
Survival: a line is kept by a modulus iff the modulus is transparent (eigenvalue d).  'allK' = kept by every
Kähler modulus (survives large volume / all Kähler phases), 'someK' = kept by a non-empty proper subset."""
from __future__ import annotations

import json, re, signal
from collections import Counter


def _verlinde(spec):
    from ..tdl import line_survival
    s = line_survival(spec)
    out = []
    nK = s["n27bar"]
    for f in s["factors"]:
        if f["d"] <= 1.0000001:
            continue
        out.append(dict(k=f["k"], r=f["r"], d=round(f["d"], 4), K=f["27bar"], Ks=f["27bar_sign"], cs=f["27"]))
    best_prod = max((x["27bar"] for x in s["supports"]), default=0)
    return dict(lines=out, nK=nK, ncs=s["n27"], allK_factors=s["full"]["27bar"], allcs_factors=s["full"]["27"],
                best_product_K=best_prod, kahler_breaks_all=s["kahler_breaks_all"])


def _perm(sym):
    from .permlines import PermLines
    pl = PermLines(sym)
    if not pl.broken:
        return []
    mults = pl.multiplets()
    rows = []
    for s, orb in pl.double_cosets():
        K = pl.stabiliser(s)
        d = len(pl.H) // len(K)
        act = pl.action(s)
        res = {}
        for rep in ("27", "27bar"):
            fixed = 0
            for comp in mults[rep]:
                a0 = act[comp[0]]
                if a0[0] == "in" and a0[1] == pl.keys[comp[0]]:
                    fixed += 1
            res[rep] = (fixed, len(mults[rep]))
        rows.append(dict(sigma=pl._cycles(s), d=d, K=res["27bar"][0], nK=res["27bar"][1], cs=res["27"][0], ncs=res["27"][1]))
    return rows


def _fp(sym):
    from .fplines import fixed_sets, su2_dim
    try:
        r = fixed_sets(sym.model)
    except Exception as e:
        return dict(error=str(e))
    sets = r[0] if isinstance(r, tuple) else r
    return dict(nsets=len(sets), sets=[list(S) for S in sets][:10])


def _mf(model):
    from . import mfdefects
    txt = mfdefects.report(model, max_exponent=12)
    cands = [int(x) for x in re.findall(r"\(candidates\): (\d+)", txt)]
    return dict(candidates=sum(cands))


def survey(path: str, timeout: int = 600) -> dict:
    from .runner import json_for, _load
    from .. import results
    js = json_for(path)
    spec = results.load(js)
    sym = _load(js)
    out = dict(model=spec.model.name)
    def handler(sig, frm):
        raise TimeoutError()
    for key, fn in (("verlinde", lambda: _verlinde(spec)), ("perm", lambda: _perm(sym)),
                    ("fp", lambda: _fp(sym)), ("mf", lambda: _mf(spec.model))):
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        import time as _t; t0 = _t.time()
        try:
            out[key] = fn()
        except TimeoutError:
            out[key] = dict(error="timeout")
        except Exception as e:
            out[key] = dict(error=f"{type(e).__name__}: {e}"[:200])
        finally:
            signal.alarm(0)
            out.setdefault("time", {})[key] = round(_t.time() - t0, 1)
    return out
