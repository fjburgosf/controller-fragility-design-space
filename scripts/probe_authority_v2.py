"""Sonda de autoridad v2 (inversion corregida del mapa accion->residuo).

El entorno aplica  residual = residual_limit * tanh(clip(action,-1,1)).
Por tanto |residual| <= 0.7616 * residual_limit y la sonda anterior, que pasaba
atanh(...) como accion, quedaba recortada. Aqui se invierte bien:
  d_deseado = clip(u_LQR(R*) - u_LQR(0.01), -0.99*0.7616*lim, +0.99*0.7616*lim)
  action    = atanh(d_deseado / residual_limit)

Se mide, ademas del rmse, la magnitud media |u_t - u_b| que el residuo tendria
que aportar frente a lo que el entorno permite entregar.

Salida: results/processed/authority_v2.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.controller import GateConfig, GateMode
from igrrl.env import ResidualBalanceEnv
from src.config import load_pendulum_params
from src.controllers.lqr import LQRController

ROOT    = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "processed" / "authority_v2.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL     = 40
LIMITS     = [4.0, 8.0, 12.0, 24.0]
R_TARGET   = 10.0
TANH1      = float(np.tanh(1.0))
PARAMS     = load_pendulum_params()
SCEN = [("SC1_nominal", False, 0.0), ("SC3_dr_perturb", True, 0.5)]


def lqr_ref(base_R, dr, pb, n=N_EVAL):
    env = ResidualBalanceEnv(mode=GateMode.MINIMUM, domain_randomization=dr,
                             perturbation_probability=pb, base_R=base_R, residual_penalty=0.0)
    th, x = [], []
    for ep in range(n):
        env.reset(seed=EVAL_SEED0 + ep); d = t = False; ts, xs = [], []
        while not d:
            _, _, t, tr, _ = env.step(np.zeros(1, dtype="float32")); d = t or tr
            ts.append((env.state[1] - np.pi) ** 2); xs.append(env.state[0] ** 2)
        th.append(np.sqrt(np.mean(ts))); x.append(np.sqrt(np.mean(xs)))
    return float(np.mean(th)), float(np.mean(x))


def oracle(limit, dr, pb, n=N_EVAL):
    gc = GateConfig(rho_max=1.0)
    env = ResidualBalanceEnv(mode=GateMode.MAXIMUM, domain_randomization=dr,
                             perturbation_probability=pb, base_R=0.01, residual_penalty=0.0,
                             residual_limit=limit, gate_config=gc, gate_warmup_s=0.0)
    tgt = LQRController(PARAMS, R=R_TARGET)
    dmax = 0.99 * TANH1 * limit
    th, x, need, got, clipped = [], [], [], [], []
    for ep in range(n):
        env.reset(seed=EVAL_SEED0 + ep); d = t = False
        ts, xs = [], []
        while not d:
            u_t = tgt.compute(env.steps * env.dt, env.xhat.copy())
            u_b = env.lqr.compute(env.steps * env.dt, env.xhat.copy())
            want = u_t - u_b
            d_des = float(np.clip(want, -dmax, dmax))
            action = np.array([np.arctanh(d_des / limit)], dtype="float32")
            _, _, t, tr, info = env.step(action); d = t or tr
            need.append(abs(want)); got.append(abs(info["residual"]))
            clipped.append(1.0 if abs(want) > dmax else 0.0)
            ts.append((env.state[1] - np.pi) ** 2); xs.append(env.state[0] ** 2)
        th.append(np.sqrt(np.mean(ts))); x.append(np.sqrt(np.mean(xs)))
    return {"rmse_theta": float(np.mean(th)), "rmse_x": float(np.mean(x)),
            "need_abs_mean": float(np.mean(need)), "got_abs_mean": float(np.mean(got)),
            "frac_clipped": float(np.mean(clipped))}


rows = []
for sc, dr, pb in SCEN:
    print(f"\n=== {sc} ===", flush=True)
    b_th, b_x = lqr_ref(0.01, dr, pb)
    r_th, r_x = lqr_ref(R_TARGET, dr, pb)
    print(f"  LQR R=0.01 puro   rmse_th={b_th:.4f}", flush=True)
    print(f"  LQR R=10   puro   rmse_th={r_th:.4f}   <- objetivo del oraculo", flush=True)
    rows.append({"scenario": sc, "case": "LQR_R0.01", "limit": "-", "rmse_theta": b_th,
                 "rmse_x": b_x, "need_abs_mean": "", "got_abs_mean": "", "frac_clipped": ""})
    rows.append({"scenario": sc, "case": "LQR_R10", "limit": "-", "rmse_theta": r_th,
                 "rmse_x": r_x, "need_abs_mean": "", "got_abs_mean": "", "frac_clipped": ""})
    for lim in LIMITS:
        m = oracle(lim, dr, pb)
        rows.append({"scenario": sc, "case": "oracle_R10", "limit": lim, **m})
        print(f"  oracle lim={lim:5.1f}V  rmse_th={m['rmse_theta']:.4f}  "
              f"|need|={m['need_abs_mean']:7.2f}V  |got|={m['got_abs_mean']:5.2f}V  "
              f"clip_frac={m['frac_clipped']:.2f}", flush=True)

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "case", "limit", "rmse_theta", "rmse_x",
                                      "need_abs_mean", "got_abs_mean", "frac_clipped"])
    w.writeheader(); w.writerows(rows)
print(f"\nGuardado: {OUT_CSV}", flush=True)
