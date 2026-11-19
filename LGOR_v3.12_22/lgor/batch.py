"""Batch runs and runtime estimates (new in 2.7).

batch: for every model, run a list of steps (run, mirror, tdl, lines, defects, toroidal, ...) as separate
`python -m lgor` processes, each with its own timeout, inside OUT/MODEL/.  `run` and `mirror` use
checkpoints, so a re-started batch resumes interrupted spectra sector by sector; finished steps (marker
file STEP.done) are skipped.  One JSON line per step goes to OUT/batch.jsonl (model, step, status, seconds),
and OUT/MODEL/DONE or OUT/MODEL/TIMEOUT marks the model.

estimate: time a few sample sectors of a model (and of its mirror) and extrapolate the full run time.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from .model import Model

DEFAULT_STEPS = ["run", "mirror", "tdl", "lines", "toroidal"]

STEP_ARGS = {
    "run": lambda lgo, d, j: ["run", str(lgo), "-o", str(d), "-j", str(j), "--checkpoint", str(d / (lgo.stem + ".ckpt"))],
    "mirror": lambda lgo, d, j: ["mirror", str(lgo), "-o", str(d), "-j", str(j), "--from-json",
                                 "--checkpoint", str(d / (lgo.stem + "_mirror.ckpt"))],
    "tdl": lambda lgo, d, j: ["tdl", str(lgo), "-o", str(d)],
    "lines": lambda lgo, d, j: ["lines", str(lgo), "-o", str(d), "-j", str(j)],
    "defects": lambda lgo, d, j: ["defects", str(lgo), "-o", str(d), "-j", str(j)],
    "toroidal": lambda lgo, d, j: ["toroidal", str(lgo), "-o", str(d)],
    "blocklines": lambda lgo, d, j: ["blocklines", str(lgo), "-o", str(d)],
}


def run_batch(lgos: Sequence[Path], out: Path, steps: Sequence[str] = DEFAULT_STEPS, jobs: int = 1,
              timeout: Optional[float] = None, python: str = sys.executable) -> List[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "batch.jsonl"
    rows = []
    env = dict(os.environ)
    pkg = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for lgo in map(Path, lgos):
        d = out / lgo.stem
        d.mkdir(exist_ok=True)
        local = d / lgo.name
        if not local.exists():
            shutil.copy(lgo, local)
        status_model = "DONE"
        for step in steps:
            if step not in STEP_ARGS:
                raise ValueError(f"unknown step {step}; known: {', '.join(STEP_ARGS)}")
            marker = d / f"{step}.done"
            if marker.exists():
                continue
            cmd = [python, "-m", "lgor"] + STEP_ARGS[step](local, d, jobs)
            t0 = time.time()
            with open(d / f"{step}.log", "a") as fh:
                fh.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                fh.flush()
                try:
                    rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
                    status = "ok" if rc == 0 else f"exit {rc}"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            dt = time.time() - t0
            row = {"model": lgo.stem, "step": step, "status": status, "seconds": round(dt, 1),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            rows.append(row)
            with open(log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{lgo.stem:30s} {step:10s} {status:8s} {dt:8.1f}s", flush=True)
            if status == "ok":
                marker.write_text(row["time"] + "\n")
            elif status == "timeout":
                status_model = "TIMEOUT"
                break
            else:
                status_model = "FAILED"
                if step in ("run", "mirror"):
                    break
        (d / status_model).write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for other in {"DONE", "TIMEOUT", "FAILED"} - {status_model}:
            if (d / other).exists():
                (d / other).unlink()
    return rows


# ----------------------------------------------------------------------------- estimate
def estimate(model: Model, samples: int = 8, processes: int = 1, seed: int = 1, check: bool = True) -> dict:
    """Time `samples` sectors drawn evenly over k0 (random twists) and extrapolate to all sectors."""
    from .spectrum import all_twists, _compute_sector
    twists = all_twists(model)
    units = [(t, k0) for t in twists for k0 in range(2 * model.korder)]
    rng = random.Random(seed)
    by_k0 = {}
    for u in units:
        by_k0.setdefault(u[1], []).append(u)
    k0s = sorted(by_k0)
    pick = []
    for i in range(min(samples, len(units))):
        k0 = k0s[(i * len(k0s)) // max(1, min(samples, len(units)))]
        pick.append(rng.choice(by_k0[k0]))
    times = []
    for t, k0 in pick:
        t0 = time.time()
        _compute_sector((model, t, k0, check))
        times.append(time.time() - t0)
    mean = sum(times) / len(times)
    total = mean * len(units)
    return {"sectors": len(units), "twists": len(twists), "sampled": len(pick), "mean_s": mean,
            "max_s": max(times), "serial_s": total, "parallel_s": total / max(1, processes)}


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def estimate_text(model: Model, samples: int = 8, processes: int = 1, mirror: bool = True) -> str:
    out = [f"RUNTIME ESTIMATE for {model.name} ({samples} sample sectors; spread is large, treat as order of magnitude)"]
    todo = [("model", model)]
    if mirror and model.is_fermat:
        from .mirror import mirror_model
        try:
            todo.append(("mirror", mirror_model(model)))
        except Exception as e:                  # pragma: no cover
            out.append(f"  mirror: not built ({e})")
    for lab, m in todo:
        e = estimate(m, samples, processes)
        out.append(f"  {lab:7s} {e['sectors']} sectors in {e['twists']} twist bunches; mean {e['mean_s']:.2f}s, "
                   f"max {e['max_s']:.2f}s per sampled sector -> serial ~{_hms(e['serial_s'])}, "
                   f"with -j {processes} ~{_hms(e['parallel_s'])}")
    return "\n".join(out) + "\n"
