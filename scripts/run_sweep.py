"""Lanzador del barrido factorial de P2.

Dos variables independientes:
  base_R           calidad del controlador base (margen disponible para el residuo)
  residual_penalty precio del residuo en la recompensa

Se escribe en Python y no en PowerShell porque la ruta del proyecto contiene
espacios y Start-Process los divide en argumentos separados.

Uso:
  python scripts/run_sweep.py --variant a3 --parallel 4
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT.parent / ".venv" / "Scripts" / "python.exe")
SCRIPT = ROOT / "scripts" / "train_residual.py"
LOGDIR = ROOT / "results" / "training" / "_sweep_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

BASE_R = [0.01, 1.0, 10.0, 100.0]
W_RES  = [0.0, 0.05]
SEEDS  = [1000, 2000, 3000]
STEPS  = 300_000


def cell_dir(variant: str, base_R: float, w_res: float, seed: int) -> Path:
    parent = ROOT / "results" / "training" / variant
    if base_R != 0.01 or w_res != 0.05:
        parent = parent / f"baseR_{base_R:g}_wres_{w_res:g}"
    return parent / f"seed_{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="a3")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(r, w, s) for r, w, s in itertools.product(BASE_R, W_RES, SEEDS)
            if not (cell_dir(args.variant, r, w, s) / "model.zip").exists()]
    print(f"Celdas pendientes: {len(jobs)} de {len(BASE_R)*len(W_RES)*len(SEEDS)}", flush=True)
    for r, w, s in jobs:
        print(f"  base_R={r:g}  w_res={w:g}  seed={s}", flush=True)
    if args.dry_run or not jobs:
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{ROOT}"
    env["PYTHONIOENCODING"] = "utf-8"

    for i in range(0, len(jobs), args.parallel):
        batch = jobs[i:i + args.parallel]
        print(f"\n[{datetime.now():%H:%M:%S}] Lote {i//args.parallel + 1}: {batch}", flush=True)
        procs = []
        for r, w, s in batch:
            tag = f"{args.variant}_R{r:g}_w{w:g}_s{s}"
            log = (LOGDIR / f"{tag}.log").open("w", encoding="utf-8")
            p = subprocess.Popen(
                [PYTHON, str(SCRIPT), "--variant", args.variant, "--seed", str(s),
                 "--timesteps", str(args.timesteps),
                 "--base-r", str(r), "--residual-penalty", str(w)],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            procs.append((p, log, tag))
        for p, log, tag in procs:
            rc = p.wait()
            log.close()
            print(f"[{datetime.now():%H:%M:%S}] {tag} -> rc={rc}", flush=True)

    print("\nBarrido completo.", flush=True)


if __name__ == "__main__":
    main()
