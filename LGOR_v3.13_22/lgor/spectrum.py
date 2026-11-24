"""Bunches of sectors, E6 multiplet assembly, and the model-level summary.

A bunch is the set of 2*KOrder sectors T_{k0; k} (k0 = 0..2K-1) with the same
discrete twist k.  Together they contain complete E6 multiplets: a 27 shows up as
1_{q-=-2} in sector k0, 16_{-1/2} in k0+1 and 10_{1} in k0+2, and so on.  This is
the port of Bunch::SummarizeSpectrum / CheckE6Consistency and of SpectrumSum /
ModelSum.
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import Model
from .sector import Sector

QPLUS = ("-3/2", "-1/2", "1/2", "3/2")


@dataclass
class SpectrumSum:
    """Counts of E6 singlets, 27, 27bar and 78 for each value of q+ (-3/2..3/2)."""
    singlets: List[int] = field(default_factory=lambda: [0] * 4)
    n27: List[int] = field(default_factory=lambda: [0] * 4)
    n27bar: List[int] = field(default_factory=lambda: [0] * 4)
    n78: List[int] = field(default_factory=lambda: [0] * 4)

    def key(self) -> Tuple:
        return (tuple(self.singlets), tuple(self.n27), tuple(self.n27bar), tuple(self.n78))

    def add(self, other: "SpectrumSum"):
        for a, b in ((self.singlets, other.singlets), (self.n27, other.n27),
                     (self.n27bar, other.n27bar), (self.n78, other.n78)):
            for i in range(4):
                a[i] += b[i]

    def empty(self) -> bool:
        return not any(self.singlets + self.n27 + self.n27bar + self.n78)

    def is_cpt_self_conjugate(self) -> bool:
        return all(self.singlets[i] == self.singlets[3 - i] and self.n27[i] == self.n27bar[3 - i]
                   and self.n78[i] == self.n78[3 - i] for i in range(4))

    # physics read-off (q+ = -3/2: gauginos / vector multiplets, q+ = -1/2: chiral multiplets)
    @property
    def extra_vectors(self) -> int:
        return self.singlets[0]

    @property
    def chiral_singlets(self) -> int:
        return self.singlets[1]

    def table(self) -> str:
        lines = ["q+      1      27     27bar  78"]
        for i in range(4):
            lines.append(f"{QPLUS[i]:<6}  {self.singlets[i]:<6} {self.n27[i]:<6} {self.n27bar[i]:<6} {self.n78[i]}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {"singlets": self.singlets, "27": self.n27, "27bar": self.n27bar, "78": self.n78}

    @classmethod
    def from_dict(cls, d) -> "SpectrumSum":
        return cls(list(d["singlets"]), list(d["27"]), list(d["27bar"]), list(d["78"]))


class Bunch:
    def __init__(self, model: Model, twist: Sequence[int]):
        self.model = model
        self.twist = tuple(twist)
        self.sectors: List[Sector] = []
        self.sum = SpectrumSum()
        self.warnings: List[str] = []

    def compute(self, check: bool = True) -> "Bunch":
        self.sectors = [Sector(self.model, k0, self.twist).compute(check) for k0 in range(2 * self.model.korder)]
        self.assemble()
        return self

    def spectrum_matrix(self, qplus: int) -> List[List[int]]:
        """9 x 2K table: row r <-> q- = (r-4)/2, column = sector k0 (as in the C++ code)."""
        K2 = 2 * self.model.korder
        S = [[0] * K2 for _ in range(9)]
        for j, sec in enumerate(self.sectors):
            for k, cx in enumerate(sec.complexes):
                row = 2 * k + 1 if sec.ramond else 2 * k
                S[row][j] = len(cx.cohomology[qplus])
        return S

    def assemble(self):
        K2 = 2 * self.model.korder
        s = SpectrumSum()
        self.warnings = []
        for i in range(4):
            S = self.spectrum_matrix(i)
            for j in range(K2):
                j1, j2 = (j + 1) % K2, (j + 2) % K2
                if S[0][j] == S[3][j1] == S[6][j2]:
                    s.n27[i] += S[0][j]
                else:
                    self.warnings.append(f"27-plet inconsistency (q+ {QPLUS[i]}, sector {j})")
                # the q- = 0 row also contains the singlets, hence <= (as in the C++ code)
                if S[1][j] <= S[4][j1] and S[1][j] == S[7][j2]:
                    s.n78[i] += S[1][j]
                else:
                    self.warnings.append(f"78-plet inconsistency (q+ {QPLUS[i]}, sector {j})")
                if S[2][j] == S[5][j1] == S[8][j2]:
                    s.n27bar[i] += S[2][j]
                else:
                    self.warnings.append(f"27bar-plet inconsistency (q+ {QPLUS[i]}, sector {j})")
                s.singlets[i] += S[4][j]
            s.singlets[i] -= s.n78[i]
        self.sum = s
        return s

    @property
    def gravitini(self) -> List[Tuple[str, Fraction]]:
        return [(sec.label, q) for sec in self.sectors for q in sec.gravitini]


def all_twists(model: Model) -> List[Tuple[int, ...]]:
    """All discrete twists, first entry running fastest (same order as the C++ ScanAll)."""
    if not model.orders:
        return [()]
    ranges = [range(o) for o in reversed(model.orders)]
    return [tuple(reversed(t)) for t in itertools.product(*ranges)]


def _compute_bunch(args):
    model, twist, check = args
    return Bunch(model, twist).compute(check)


@dataclass
class ModelSpectrum:
    model: Model
    bunches: List[Bunch]

    @property
    def total(self) -> SpectrumSum:
        t = SpectrumSum()
        for b in self.bunches:
            t.add(b.sum)
        return t

    def grouped(self) -> List[Tuple[SpectrumSum, List[Tuple[int, ...]]]]:
        groups: Dict[Tuple, Tuple[SpectrumSum, List]] = {}
        for b in self.bunches:
            k = b.sum.key()
            if k not in groups:
                groups[k] = (b.sum, [])
            groups[k][1].append(b.twist)
        return list(groups.values())

    def bunch(self, twist: Sequence[int] = ()) -> Bunch:
        twist = tuple(twist)
        for b in self.bunches:
            if b.twist == twist:
                return b
        raise KeyError(f"no bunch with twist {twist}")

    def sector(self, k0: int, twist: Sequence[int] = ()) -> Sector:
        return self.bunch(twist).sectors[k0]

    @property
    def warnings(self) -> List[str]:
        out = []
        import math
        g = self.model.korder
        for f in self.model.fields:
            g = math.gcd(g, f.kcharge)
        if g > 1:   # v2.4: e.g. a D_even block written with KOrder 2(n-1) instead of n-1
            out.append(f"KOrder {self.model.korder} and all R-charge numerators share the factor {g}: this is the "
                       f"Gepner model with KOrder {self.model.korder // g} orbifolded by an extra Z_{g} of the gauge lattice")
        for b in self.bunches:
            for w in b.warnings:
                out.append(f"bunch {b.twist}: {w}")
            for sec in b.sectors:
                for w in sec.warnings:
                    out.append(f"{sec.label}: {w}")
        return out

    def summary(self) -> str:
        t = self.total
        lines = [f"Spectrum of {self.model.name}", t.table(), "",
                 f"27: {t.n27[1]}   27bar: {t.n27bar[1]}   chiral singlets: {t.chiral_singlets}   "
                 f"extra vectors: {t.extra_vectors}   E6 gauginos (78): {t.n78[0]}"]
        groups = self.grouped()
        if len(self.bunches) > 1:
            lines.append("")
            lines.append(f"{len(self.bunches)} bunches, {len(groups)} different bunch spectra:")
            for s, twists in groups:
                lines.append(f"  multiplicity {len(twists)}  (e.g. twist {twists[0]})")
                lines.extend("    " + l for l in s.table().splitlines())
        w = self.warnings
        if w:
            lines.append("")
            lines.append(f"{len(w)} warnings:")
            lines.extend("  " + x for x in w[:20])
            if len(w) > 20:
                lines.append(f"  ... and {len(w) - 20} more")
        return "\n".join(lines)


def _compute_sector(args):
    model, twist, k0, check = args
    return twist, k0, Sector(model, k0, twist).compute(check)


def _unit_file(ckdir: Path, twist, k0: int) -> Path:
    return ckdir / (("t" + "_".join(map(str, twist)) if twist else "t") + f"__k{k0}.pkl")


def _open_checkpoint(ckdir: Path, model: Model):
    """Checkpoint folder: meta.json holds a hash of the model; a mismatch means another model's checkpoint."""
    import hashlib
    import json
    ckdir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(model.to_string().encode()).hexdigest()
    meta = ckdir / "meta.json"
    if meta.exists():
        old = json.loads(meta.read_text()).get("model_sha256")
        if old != h:
            raise ValueError(f"checkpoint {ckdir} belongs to a different model (hash {old[:12]} != {h[:12]}); "
                             "remove it or choose another --checkpoint folder")
    else:
        meta.write_text(json.dumps({"model": model.name, "model_sha256": h, "korder": model.korder,
                                    "orders": list(model.orders)}, indent=1))


def compute_model(model: Model, twists: Optional[Iterable[Sequence[int]]] = None,
                  processes: int = 1, check: bool = True, progress=None,
                  checkpoint: Optional[os.PathLike] = None, log: Optional[os.PathLike] = None) -> ModelSpectrum:
    """Compute the full massless spectrum (all bunches of twisted sectors).

    The work unit is one twisted sector (twist, k0), so -j N parallelises also models with a single
    bunch (J-only Gepner models).  checkpoint=DIR saves every finished sector as DIR/t<twist>__k<k0>.pkl
    and reuses saved sectors on the next call (resume after a crash or restart).  log=FILE appends one
    line per finished sector (time, count, sector, seconds) and is flushed at once, so long runs can be
    monitored with  tail -f FILE."""
    import pickle
    problems = model.check()
    if problems:
        raise ValueError("inconsistent model: " + "; ".join(problems))
    twists = list(twists) if twists is not None else all_twists(model)
    units = [(t, k0) for t in twists for k0 in range(2 * model.korder)]
    n = len(units)
    done = {}
    ckdir = Path(checkpoint) if checkpoint else None
    if ckdir:
        _open_checkpoint(ckdir, model)
        for t, k0 in units:
            f = _unit_file(ckdir, t, k0)
            if f.exists():
                try:
                    with open(f, "rb") as fh:
                        done[(t, k0)] = pickle.load(fh)
                except Exception:          # truncated file from a crash: recompute that sector
                    f.unlink()
    logfh = open(log, "a", buffering=1) if log else None
    t_start = time.time()
    if logfh:
        logfh.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {model.name}: {n} sectors, {len(done)} from checkpoint, "
                    f"{processes or 1} process(es)\n")

    def finished(t, k0, sec, dt):
        done[(t, k0)] = sec
        if ckdir:
            f = _unit_file(ckdir, t, k0)
            tmp = f.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(sec, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, f)
        if logfh:
            logfh.write(f"{time.strftime('%H:%M:%S')} {len(done)}/{n} twist={t} k0={k0} {dt:.1f}s "
                        f"(total {time.time() - t_start:.0f}s)\n")
        if progress:
            progress(len(done), n)

    todo = [(model, t, k0, check) for t, k0 in units if (t, k0) not in done]
    if progress and done:
        progress(len(done), n)
    ran_parallel = False
    if processes and processes > 1 and len(todo) > 1:
        import multiprocessing as mp
        # "fork" works from scripts without a __main__ guard and from notebooks;
        # it exists on Linux and macOS.  Elsewhere fall back to "spawn", and to a
        # serial run if worker processes cannot be started at all.
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        try:
            with mp.get_context(method).Pool(min(processes, len(todo))) as pool:
                last = time.time()
                for t, k0, sec in pool.imap_unordered(_compute_sector, todo):
                    now = time.time()
                    finished(t, k0, sec, now - last)
                    last = now
            ran_parallel = True
        except (RuntimeError, OSError):
            ran_parallel = False
    if not ran_parallel:
        for job in todo:
            if (job[1], job[2]) in done:
                continue
            t0 = time.time()
            t, k0, sec = _compute_sector(job)
            finished(t, k0, sec, time.time() - t0)
    if logfh:
        logfh.write(f"# done in {time.time() - t_start:.0f}s\n")
        logfh.close()
    bunches: List[Bunch] = []
    for t in twists:
        b = Bunch(model, t)
        b.sectors = [done[(t, k0)] for k0 in range(2 * model.korder)]
        b.assemble()
        bunches.append(b)
    return ModelSpectrum(model, bunches)
