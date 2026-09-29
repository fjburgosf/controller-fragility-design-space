"""Arnes de evaluacion COMUN a todos los controladores de P2.

Todo controlador (LQR, MPC(L_design), SAC, DDPG) se evalua sobre las MISMAS
realizaciones pareadas y con las MISMAS metricas, de modo que todos caen sobre
la misma curva de compromiso y admiten estadistica pareada.

Metricas por realizacion: caida del pendulo, violacion del riel FISICO,
excursion maxima del carro, rmse angular y de carro, energia.
Agregados: media Y COLA (P95, peor caso, success rate) — el eje central de la
hipotesis media-vs-cola (D29).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.mpc_constrained import ConstrainedMPC, Q_P2, R_P2

DT = 0.02
T_FINAL = 6.0
NSTEP = int(round(T_FINAL / DT))
K_ON, K_OFF = int(round(1.0 / DT)), int(round(1.4 / DT))
RAIL = 0.30
ACT_LIM = 12.0
MEAS_STD = 1e-3
FALL_TH = 0.30
EVAL_SEED0 = 9000
NOM = load_pendulum_params()
DR_RANGES = {"Mp": (0.80, 1.20), "Jp": (0.75, 1.25), "l": (0.85, 1.15), "kt": (0.90, 1.10)}


def make_lqr():
    c = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=0.0)
    return lambda t, xh: c.compute(t, xh)


def make_mpc(L_design):
    c = ConstrainedMPC(NOM, dt=DT, x_limit=L_design, x_ref=0.0)
    return lambda t, xh: c.compute(t, xh)


def make_policy(model):
    """Politica SB3 sobre la observacion de 5 dims de StandaloneBalanceEnv."""
    def f(t, xh):
        te = np.arctan2(np.sin(xh[1] - np.pi), np.cos(xh[1] - np.pi))
        obs = np.array([xh[0], np.sin(te), np.cos(te), xh[2], xh[3]], dtype=np.float32)
        a, _ = model.predict(obs, deterministic=True)
        # Mismo mapeo LINEAL que StandaloneBalanceEnv.step (rango completo del actuador).
        return float(ACT_LIM * np.clip(np.asarray(a).ravel()[0], -1.0, 1.0))
    return f


def run_episode(control, seed, dv=6.0, dr=True):
    """control(t, xhat) -> u. Mismo seed => misma planta, ruido y perturbacion."""
    rng = np.random.default_rng(seed)
    p = (replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR_RANGES.items()})
         if dr else NOM)
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xhat = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MEAS_STD**2] * 2))
    maxx = 0.0; ths = []; xs = []; us = []
    for k in range(NSTEP):
        u = float(np.clip(control(k * DT, xhat), -ACT_LIM, ACT_LIM))
        d = dv if K_ON <= k < K_OFF else 0.0
        y = x[[0, 1]] + rng.normal(0.0, MEAS_STD, size=2)
        xhat, P, _ = ekf.step(xhat, P, u, y, DT)
        x = rk4_step(x, u + d, p, DT)
        maxx = max(maxx, abs(x[0]))
        ths.append((x[1] - np.pi) ** 2); xs.append(x[0]); us.append(u)
    rmse_th = float(np.sqrt(np.mean(ths)))
    return {"maxx": maxx, "fell": int(rmse_th > FALL_TH), "rail_viol": int(maxx > RAIL),
            "rmse_theta": rmse_th, "rmse_x": float(np.sqrt(np.mean(np.square(xs)))),
            "energy": float(np.sum(np.square(us)) * DT)}


def evaluate(control, n=64, dv=6.0, dr=True, seed0=EVAL_SEED0):
    """Devuelve (agregados media+cola, lista de episodios para estadistica pareada)."""
    eps = [run_episode(control, seed0 + i, dv=dv, dr=dr) for i in range(n)]
    th = np.array([e["rmse_theta"] for e in eps])
    mx = np.array([e["maxx"] for e in eps])
    en = np.array([e["energy"] for e in eps])
    agg = {
        # --- media ---
        "rmse_theta_mean": float(th.mean()), "maxx_mean": float(mx.mean()),
        "energy_mean": float(en.mean()),
        # --- COLA (eje de la hipotesis media-vs-cola, D29) ---
        "rmse_theta_p95": float(np.percentile(th, 95)),
        "rmse_theta_worst": float(th.max()),
        "maxx_p95": float(np.percentile(mx, 95)),
        "maxx_worst": float(mx.max()),
        "fell_frac": float(np.mean([e["fell"] for e in eps])),
        "rail_viol_frac": float(np.mean([e["rail_viol"] for e in eps])),
        "success_rate": float(1.0 - np.mean([e["fell"] for e in eps])),
        "n": n,
    }
    return agg, eps
