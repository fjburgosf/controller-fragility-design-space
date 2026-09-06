"""Lanza el entrenamiento formal: 4 variantes x 5 semillas en lotes de 4 paralelos."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SCRIPT = ROOT / "scripts" / "train_residual.py"
LOGDIR = ROOT / "results" / "logs" / "formal"
LOGDIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["a1", "a2", "a3", "p2"]
SEEDS    = [1000, 2000, 3000, 4000, 5000]
STEPS    = 300_000

env = {
    "PYTHONPATH": f".;src;{ROOT / 'src'}",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}
import os
env = {**os.environ, **env}

for seed in SEEDS:
    print(f"=== Semilla {seed} ===", flush=True)
    procs = []
    for variant in VARIANTS:
        tag    = f"{variant}_seed{seed}"
        stdout = (LOGDIR / f"{tag}.stdout.log").open("w", encoding="utf-8")
        stderr = (LOGDIR / f"{tag}.stderr.log").open("w", encoding="utf-8")
        p = subprocess.Popen(
            [PYTHON, str(SCRIPT), "--variant", variant, "--seed", str(seed), "--timesteps", str(STEPS)],
            cwd=str(ROOT),
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        print(f"  Lanzado {tag}  PID={p.pid}", flush=True)
        procs.append((p, stdout, stderr))
    for p, out, err in procs:
        p.wait()
        out.close(); err.close()
    print(f"  Semilla {seed} completada.", flush=True)

print("=== ENTRENAMIENTO FORMAL COMPLETO ===", flush=True)
