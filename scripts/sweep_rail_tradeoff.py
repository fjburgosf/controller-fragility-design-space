"""Curva de compromiso RIEL vs PENDULO (hallazgo D28) — figura central de P2.

El riel FISICO es fijo: RAIL=0.30 m. Lo que el disenador elige es el limite de
DISENO L_design que impone al MPC. Se barre L_design de apretado a infinito
(= sin restriccion) y se mide, bajo rafaga transitoria + incertidumbre + ruido:

  - fell_frac      : fraccion de realizaciones donde el pendulo cae
  - rail_viol_frac : fraccion donde |x| supera el riel FISICO (0.30)
  - p95_maxx       : excursion de carro en peor caso
  - rmse_theta     : calidad de regulacion angular

Hipotesis (D28): apretar L_design mejora el cumplimiento del riel pero prohibe
la maniobra de recuperacion -> el pendulo cae. El LQR (sin restriccion) es el
extremo opuesto: viola el riel pero salva el pendulo. Compromiso genuino.

Salida: results/processed/rail_tradeoff.csv
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.mpc_constrained import ConstrainedMPC, Q_P2, R_P2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / "rail_tradeoff.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DT, T_FINAL = 0.02, 6.0
NSTEP = int(round(T_FINAL / DT))
K_ON, K_OFF = int(round(1.0 / DT)), int(round(1.4 / DT))
RAIL = 0.30                      # riel FISICO fijo
L_DESIGNS = [0.25, 0.30, 0.40, 0.60, 1.00, None]   # None = sin restriccion
DV_LEVELS = [6.0, 8.0]
ACT_LIM, MEAS_STD, N_EVAL, EVAL_SEED0 = 12.0, 1e-3, 24, 9000
FALL_TH = 0.30                   # rmse_theta > 0.30 rad => caido

NOM = load_pendulum_params()
DR = {"Mp": (0.80, 1.20), "Jp": (0.75, 1.25), "l": (0.85, 1.15), "kt": (0.90, 1.10)}
LQR = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=0.0)
print("construyendo MPCs (una vez por L_design)...", flush=True)
MPCS = {L: ConstrainedMPC(NOM, dt=DT, x_limit=L, x_ref=0.0) for L in L_DESIGNS}
print("listo\n", flush=True)


def simulate(method, dv, seed):
    rng = np.random.default_rng(seed)
    p = replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR.items()})
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xhat = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MEAS_STD**2] * 2))
    maxx = 0.0; ths = []; us = []
    for k in range(NSTEP):
        u = LQR.compute(k * DT, xhat) if method == "lqr" else MPCS[method].compute(k * DT, xhat)
        u = float(np.clip(u, -ACT_LIM, ACT_LIM))
        d = dv if K_ON <= k < K_OFF else 0.0
        y = x[[0, 1]] + rng.normal(0.0, MEAS_STD, size=2)
        xhat, P, _ = ekf.step(xhat, P, u, y, DT)
        x = rk4_step(x, u + d, p, DT)
        maxx = max(maxx, abs(x[0])); ths.append((x[1] - np.pi) ** 2); us.append(u)
    rmse = float(np.sqrt(np.mean(ths)))
    return {"maxx": maxx, "fell": int(rmse > FALL_TH), "rail_viol": int(maxx > RAIL),
            "rmse_theta": rmse, "energy": float(np.sum(np.square(us)) * DT)}


def batch(method, dv):
    rs = [simulate(method, dv, EVAL_SEED0 + i) for i in range(N_EVAL)]
    return {"fell_frac": float(np.mean([r["fell"] for r in rs])),
            "rail_viol_frac": float(np.mean([r["rail_viol"] for r in rs])),
            "p95_maxx": float(np.percentile([r["maxx"] for r in rs], 95)),
            "rmse_theta": float(np.mean([r["rmse_theta"] for r in rs])),
            "energy": float(np.mean([r["energy"] for r in rs]))}


rows = []
print(f"riel FISICO={RAIL}  N={N_EVAL}  rafaga [1.0,1.4]s\n", flush=True)
print(f"{'dv':>4} {'L_design':>10} {'cae':>6} {'viola_riel':>11} {'p95|x|':>7} {'rmseTh':>7} {'E':>7}", flush=True)
for dv in DV_LEVELS:
    for L in L_DESIGNS:
        a = batch(L, dv)
        a.update(dv=dv, L_design=("inf" if L is None else L), method="MPC")
        rows.append(a)
        print(f"{dv:4.1f} {('sin restr' if L is None else f'{L:.2f}'):>10} {a['fell_frac']:6.2f} "
              f"{a['rail_viol_frac']:11.2f} {a['p95_maxx']:7.3f} {a['rmse_theta']:7.4f} {a['energy']:7.1f}", flush=True)
    a = batch("lqr", dv)
    a.update(dv=dv, L_design="LQR", method="LQR")
    rows.append(a)
    print(f"{dv:4.1f} {'LQR':>10} {a['fell_frac']:6.2f} {a['rail_viol_frac']:11.2f} "
          f"{a['p95_maxx']:7.3f} {a['rmse_theta']:7.4f} {a['energy']:7.1f}\n", flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dv", "method", "L_design", "fell_frac",
                                      "rail_viol_frac", "p95_maxx", "rmse_theta", "energy"])
    w.writeheader(); w.writerows(rows)
print(f"Guardado: {OUT}", flush=True)
