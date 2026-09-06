"""Curva riel-vs-pendulo con el MPC BIEN FORMULADO (rehace rail_tradeoff, D30).

La version anterior usaba la implementacion densa con blocking uniforme y coste
terminal inconsistente, cuyos fallos resultaron ser artefactos. Aqui se usa
DesignMPC con la mejor configuracion hallada en la rejilla
(blocking 'front' + terminal Riccati discreto), y se barre SOLO el limite de
diseno del carro, que es la variable de interes.

Pregunta: con un MPC correctamente formulado, la conciencia de restriccion
ayuda, es neutra, o perjudica?

Salida: results/processed/rail_correct.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.evaluate_common import run_episode, RAIL, EVAL_SEED0, NOM, make_lqr
from igrrl.mpc_design import DesignMPC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / "rail_correct.csv"

BEST = dict(dt=0.01, T_pred=0.8, Nc=20, blocking="front", terminal="riccati")
L_DESIGNS = [0.20, 0.25, 0.30, 0.40, 0.60, None]   # None = sin restriccion
DVS = [4.0, 6.0, 8.0]
N = 32


def stats(eps):
    th = np.array([e["rmse_theta"] for e in eps]); mx = np.array([e["maxx"] for e in eps])
    return {"fell_frac": float(np.mean([e["fell"] for e in eps])),
            "rail_viol_frac": float(np.mean([e["rail_viol"] for e in eps])),
            "rmse_theta": float(th.mean()), "rmse_theta_p95": float(np.percentile(th, 95)),
            "maxx_p95": float(np.percentile(mx, 95)), "maxx_worst": float(mx.max()),
            "energy": float(np.mean([e["energy"] for e in eps]))}


rows = []
print(f"MPC bien formulado {BEST} | riel FISICO={RAIL} | N={N}\n", flush=True)
print(f"{'dv':>4} {'L_diseno':>10} {'cae':>6} {'viola':>6} {'rmseTh':>8} {'p95|x|':>8} {'peor|x|':>8} {'E':>7}", flush=True)
for dv in DVS:
    a = stats([run_episode(make_lqr(), EVAL_SEED0 + i, dv=dv) for i in range(N)])
    a.update(dv=dv, method="LQR", L_design="")
    rows.append(a)
    print(f"{dv:4.1f} {'LQR':>10} {a['fell_frac']:6.2f} {a['rail_viol_frac']:6.2f} "
          f"{a['rmse_theta']:8.4f} {a['maxx_p95']:8.3f} {a['maxx_worst']:8.3f} {a['energy']:7.1f}", flush=True)
    for L in L_DESIGNS:
        m = DesignMPC(NOM, x_limit=L, **BEST)
        a = stats([run_episode(lambda t, xh, _m=m: _m.compute(t, xh), EVAL_SEED0 + i, dv=dv)
                   for i in range(N)])
        a.update(dv=dv, method="MPC", L_design=("sin" if L is None else L))
        rows.append(a)
        print(f"{dv:4.1f} {str(a['L_design']):>10} {a['fell_frac']:6.2f} {a['rail_viol_frac']:6.2f} "
              f"{a['rmse_theta']:8.4f} {a['maxx_p95']:8.3f} {a['maxx_worst']:8.3f} {a['energy']:7.1f}", flush=True)
    print("", flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dv", "method", "L_design", "fell_frac", "rail_viol_frac",
                                      "rmse_theta", "rmse_theta_p95", "maxx_p95", "maxx_worst", "energy"])
    w.writeheader(); w.writerows(rows)
print(f"Guardado: {OUT.name}", flush=True)
