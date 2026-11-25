"""Compare the Python port with a C++ .lrt result file, state space by state space.

For every (twist, sector, q-, q+) the two codes must give cohomology spaces of the
same dimension, and the states must span the same space (checked by exact rank).

usage: python compare_cpp.py model.lgo model.lrt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgor import Model, compute_model
from lgor.legacy import read_lrt
from lgor.linalg import rank


def span_rank(polys, index):
    rows = []
    for p in polys:
        v = [0] * len(index)
        for c, m in p:
            v[index[m]] += c
        rows.append(v)
    return rank(rows, len(index))


def main(lgo, lrt):
    model = Model.from_file(lgo)
    spec = compute_model(model, processes=4)
    cpp = read_lrt(lrt, model.nfields)
    py = {}
    for b in spec.bunches:
        for sec in b.sectors:
            for a, cx in enumerate(sec.complexes):
                for q in range(4):
                    if cx.cohomology[q]:
                        py[(b.twist, sec.k0, a, q)] = cx.cohomology[q]
    keys = sorted(set(py) | set(cpp))
    dim_mismatch = span_mismatch = 0
    nstates = 0
    for k in keys:
        P, C = py.get(k, []), cpp.get(k, [])
        nstates += len(P)
        if len(P) != len(C):
            dim_mismatch += 1
            print("dimension mismatch", k, "python", len(P), "c++", len(C))
            continue
        monos = sorted({m for p in P + C for _, m in p})
        index = {m: i for i, m in enumerate(monos)}
        rp, rc, rb = span_rank(P, index), span_rank(C, index), span_rank(P + C, index)
        if not (rp == rc == rb == len(P)):
            span_mismatch += 1
            print("span mismatch", k, rp, rc, rb, len(P))
    print(f"{model.name}: {len(keys)} state spaces, {nstates} states; "
          f"{dim_mismatch} dimension mismatches, {span_mismatch} span mismatches")
    print("python warnings:", spec.warnings[:5])
    return dim_mismatch + span_mismatch


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2]) else 0)
