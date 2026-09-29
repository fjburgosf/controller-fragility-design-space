"""Compara FISICAMENTE el checkpoint 'best' contra el 'final' de cada semilla.

Motivo. La Seccion 4.7 afirma que dos de las cinco semillas tienen un checkpoint
final que supera al elegido como mejor por retorno. Esa afirmacion no puede
sostenerse con el retorno, porque best_eval es por definicion el maximo visto
durante el entrenamiento y el final nunca lo supera. Solo se sostiene midiendo el
comportamiento en la planta, y ese resultado no estaba guardado en ningun archivo:
vivia unicamente en la salida por consola de un script de sonda.

Aqui se evalua cada par de checkpoints sobre las MISMAS realizaciones pareadas y
se guarda el resultado, de modo que el manuscrito cite un dato y no un recuerdo.

Salida: results/processed/checkpoint_comparison.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from igrrl.evaluate_common import make_policy, run_episode, EVAL_SEED0
from stable_baselines3 import SAC, DDPG

N_EVAL = 24          # mismas realizaciones para los dos checkpoints
DV = 5.0             # dentro del rango de entrenamiento, donde ambos son usables
ALGOS = {"sac": SAC, "ddpg": DDPG}


def evalua(modelo) -> dict:
    """Metricas fisicas sobre un lote pareado de realizaciones."""
    pol = make_policy(modelo)
    filas = [run_episode(pol, EVAL_SEED0 + i, DV) for i in range(N_EVAL)]
    df = pd.DataFrame(filas)
    return {
        "fell_frac": float(df.fell.mean()),
        "rmse_theta": float(df.rmse_theta.mean()),
        "rmse_theta_p95": float(np.percentile(df.rmse_theta, 95)),
        "maxx_p95": float(np.percentile(df.maxx, 95)),
    }


filas = []
for algo, cls in ALGOS.items():
    for s in range(1, 6):
        base = ROOT / "results" / "training" / f"standalone_{algo}" / f"seed_{s}"
        if not (base / "best_model.zip").exists():
            print(f"  aviso, falta {base}"); continue
        meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
        for etiqueta, arch in (("best", "best_model"), ("final", "model_final")):
            if not (base / f"{arch}.zip").exists():
                print(f"  aviso, falta {arch} en {base}"); continue
            m = evalua(cls.load(str(base / arch)))
            m.update(algo=algo.upper(), seed=s, checkpoint=etiqueta,
                     retorno_meta=meta[f"{etiqueta}_eval"], dv=DV, n=N_EVAL)
            filas.append(m)
        print(f"  {algo} s{s} evaluado")

df = pd.DataFrame(filas)
sal = ROOT / "results" / "processed" / "checkpoint_comparison.csv"
df.to_csv(sal, index=False)

print()
print("=" * 92)
print("EL FINAL SUPERA AL BEST EN LA PLANTA?  (menor RMSE angular es mejor)")
print("=" * 92)
piv = df.pivot_table(index=["algo", "seed"], columns="checkpoint",
                     values=["rmse_theta", "retorno_meta"])
n_mejor = 0
for (algo, s), fila in piv.iterrows():
    rb, rf = fila[("rmse_theta", "best")], fila[("rmse_theta", "final")]
    qb, qf = fila[("retorno_meta", "best")], fila[("retorno_meta", "final")]
    gana = rf < rb
    n_mejor += int(gana)
    print(f"  {algo} s{s}  retorno {qb:8.1f} -> {qf:8.1f}   "
          f"RMSE th {rb:.4f} -> {rf:.4f}   {'FINAL MEJOR' if gana else 'best mejor'}")
print()
print(f"  semillas donde el final supera fisicamente al best: {n_mejor} de {len(piv)}")
sac = piv.loc["SAC"]
print(f"  de ellas en soft actor critic: "
      f"{int((sac[('rmse_theta','final')] < sac[('rmse_theta','best')]).sum())} de {len(sac)}")
print(f"\nGuardado: {sal}")
