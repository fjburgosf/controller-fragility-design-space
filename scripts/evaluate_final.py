"""Evaluacion final pareada de las tres familias (P2).

Barre la perturbacion a traves de la frontera de la distribucion de entrenamiento
del DRL, que esta en 6 V para perturbaciones sostenidas, de modo que dv <= 6 es
interpolacion y dv > 6 es extrapolacion.

Todas las familias ven las MISMAS realizaciones pareadas en cada nivel.
Guarda agregados con estadistica de cola y episodios individuales para el
analisis estadistico pareado posterior.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.evaluate_common import (make_lqr, make_policy, run_episode, RAIL,
                                   FALL_TH, EVAL_SEED0, NOM)
from igrrl.mpc_design import DesignMPC

ROOT = Path(__file__).resolve().parent.parent
OUT_A = ROOT / "results" / "processed" / "final_summary.csv"
OUT_E = ROOT / "results" / "processed" / "final_episodes.csv"

DVS = [4.0, 6.0, 8.0, 10.0]          # 4 y 6 dentro, 8 y 10 fuera
N = 48
MPC_BEST = dict(dt=0.01, T_pred=0.8, Nc=20, blocking="front", terminal="riccati")


def cola(eps):
    th = np.array([e["rmse_theta"] for e in eps]); mx = np.array([e["maxx"] for e in eps])
    en = np.array([e["energy"] for e in eps])
    return {"fell_frac": float(np.mean([e["fell"] for e in eps])),
            "rail_viol_frac": float(np.mean([e["rail_viol"] for e in eps])),
            "rmse_theta": float(th.mean()), "rmse_theta_p95": float(np.percentile(th, 95)),
            "rmse_theta_worst": float(th.max()),
            "maxx_p95": float(np.percentile(mx, 95)), "maxx_worst": float(mx.max()),
            "energy": float(en.mean())}


rows, eps_rows = [], []


def evalua(nombre, familia, ctrl, dv, semilla=""):
    eps = [run_episode(ctrl, EVAL_SEED0 + i, dv=dv) for i in range(N)]
    a = cola(eps); a.update(method=nombre, family=familia, seed=semilla, dv=dv, n=N,
                            in_distribution=int(dv <= 6.0))
    rows.append(a)
    for i, e in enumerate(eps):
        eps_rows.append({"method": nombre, "family": familia, "seed": semilla, "dv": dv,
                         "episode": i, "eval_seed": EVAL_SEED0 + i, **e})
    print(f"  {nombre:12} dv={dv:4.1f}  cae {a['fell_frac']:.2f}  rmseTh {a['rmse_theta']:7.4f}"
          f"  p95|x| {a['maxx_p95']:6.3f}  viola {a['rail_viol_frac']:.2f}  E {a['energy']:6.1f}",
          flush=True)


from stable_baselines3 import SAC, DDPG

print(f"n={N} por nivel | frontera de entrenamiento del DRL en 6 V | RAIL={RAIL}\n", flush=True)
mpc = DesignMPC(NOM, **MPC_BEST)

for dv in DVS:
    print(f"--- dv = {dv} V  ({'dentro' if dv <= 6 else 'FUERA'} de distribucion) ---", flush=True)
    evalua("LQR", "classical", make_lqr(), dv)
    evalua("MPC", "predictive", lambda t, xh: mpc.compute(t, xh), dv)
    for algo, cls in (("sac", SAC), ("ddpg", DDPG)):
        for s in range(1, 6):
            f = ROOT / "results" / "training" / f"standalone_{algo}" / f"seed_{s}" / "best_model.zip"
            if not f.exists():
                continue
            m = cls.load(str(f.with_suffix("")))
            evalua(f"{algo.upper()}_s{s}", "learned", make_policy(m), dv, semilla=str(s))
    print("", flush=True)

with OUT_A.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with OUT_E.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(eps_rows[0].keys())); w.writeheader(); w.writerows(eps_rows)
print(f"Guardado {OUT_A.name} ({len(rows)} filas) y {OUT_E.name} ({len(eps_rows)} episodios)")
