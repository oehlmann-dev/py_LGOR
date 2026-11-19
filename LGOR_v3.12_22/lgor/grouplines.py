"""grouplines (LGOR 3.7 name): alias of the gtlines package (LGOR 3.8)."""
from .gtlines import fixedpoints as _f, groups as _g, tft as _t, blocks as _b, states as _s, pipeline as _p

for _m in (_f, _g, _t, _b, _s, _p):
    globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})


def model_blocks(model):
    """c_hat = 1 Fermat blocks (3.7 interface): ([(fields, exponents)], exponents)."""
    exps = [model.fermat_exponent(i) for i in range(model.nfields)]
    bl, _ = _b.find_blocks(model, chats=(1,))
    return [(b.fields, b.ns) for b in bl if b.fermat], exps


def analyse_models(paths, nmax=5):
    text, rows, cache = [], [], {}
    for p in paths:
        t, r, cache = _p.analyse_model(p, mirror=None, nmax=nmax, chats=(1,), cache=cache)
        text.append(t)
        rows += r
    return "\n".join(text), rows
