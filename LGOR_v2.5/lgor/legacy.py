"""Readers for the result files written by the original C++ LGOR.

* ``.lrt``: full results, every cohomology state written as
  ``Term <coeff>  B BB F BF  B BB F BF ...`` (four powers per field).
* ``.lrs``: short summary (counts of 1, 27, 27bar, 78 per q+).

Note: the C++ batch mode wrote the ``.lrs`` file *before* computing, filled with
zeros, so a crashed or interrupted run leaves an all-zero ``.lrs`` behind.  The
``.lrt`` header summary is also all zeros for models without discrete
symmetries.  Use :func:`read_lrt` and recompute the counts from the states.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .spectrum import SpectrumSum

LrtKey = Tuple[Tuple[int, ...], int, int, int]   # (twist, k0, q- index, q+ index)


def read_lrt(path, nfields: int) -> Dict[LrtKey, List[List[Tuple[int, Tuple[int, ...]]]]]:
    """Return {(twist, k0, q- index, q+ index): [state, ...]}.

    Each state is a list of (coeff, monomial), with the monomial in the encoding of
    :mod:`lgor.monomial` (B..., BB..., F..., BF...).
    """
    out: Dict[LrtKey, list] = defaultdict(list)
    twist: Tuple[int, ...] = ()
    k0 = ql = qr = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        word = line.split()[0]
        nums = [int(x) for x in re.findall(r"-?\d+", line)]
        if word == "Bunch":
            twist = tuple(nums)
        elif word == "Sector":
            k0 = nums[0]
        elif word == "Cspace":
            ql, qr = nums[0], nums[1]
        elif word == "State":
            out[(twist, k0, ql, qr)].append([])
        elif word == "Term":
            coeff, powers = nums[0], nums[1:]
            if len(powers) != 4 * nfields:
                raise ValueError(f"term with {len(powers)} powers, expected {4 * nfields}: {raw!r}")
            mono = [0] * (4 * nfields)
            for i in range(nfields):
                b, bb, f, bf = powers[4 * i:4 * i + 4]
                mono[i], mono[nfields + i], mono[2 * nfields + i], mono[3 * nfields + i] = b, bb, f, bf
            out[(twist, k0, ql, qr)][-1].append((coeff, tuple(mono)))
    return dict(out)


def read_lrs(path) -> Tuple[SpectrumSum, List[Tuple[int, SpectrumSum]]]:
    """Return (model total, [(multiplicity, bunch spectrum), ...]) from an .lrs file."""
    total = None
    bunches: List[Tuple[int, SpectrumSum]] = []
    current = None
    mult = 0
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Bunch Multiplicity"):
            mult = int(re.findall(r"\d+", line)[0])
            current = SpectrumSum()
            bunches.append((mult, current))
            continue
        m = re.match(r"(-?\d)/2\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            i = (int(m.group(1)) + 3) // 2
            target = current
            if target is None:
                if total is None:
                    total = SpectrumSum()
                target = total
            target.singlets[i], target.n27[i], target.n27bar[i], target.n78[i] = map(int, m.groups()[1:])
    return total or SpectrumSum(), bunches
