"""Sonda de margen (headroom) del controlador base.

Mide el desempeno del LQR puro (residuo = 0) para una familia de R de diseno.
Define headroom(R) = J(LQR_R) - J(LQR_0.01) sobre el mismo conjunto pareado de
realizaciones. Sirve para escoger los niveles de R del barrido de calidad.

Salida: results/processed/headroom_probe.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv

ROOT    = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "processed" / "headroom_probe.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL     = 64
R_LEVELS   = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

SCENARIOS = [
    ("SC1_nominal",    False, 0.0),
    ("SC2_dr_only",    True,  0.0),
    ("SC3_dr_perturb", True,  0.5),
]


def zero_action(obs):
    return np.zeros(1, dtype="float32")


def run(base_R, dr, perturb, n=N_EVAL, seed0=EVAL_SEED0):
    env = ResidualBalanceEnv(mode=GateMode.MINIMUM, domain_randomization=dr,
                             perturbation_probability=perturb, base_R=base_R)
    rets, th, xr, en, ok = [], [], [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        ret, done, term = 0.0, False, False
        th_sq, x_sq, e_sq = [], [], []
        while not done:
            obs, rew, term, trunc, info = env.step(zero_action(obs))
            done = term or trunc
            ret += rew
            th_sq.append((env.state[1] - np.pi) ** 2)
            x_sq.append(env.state[0] ** 2)
            e_sq.append(info["u_control"] ** 2)
        rets.append(ret); ok.append(0 if term else 1)
        th.append(np.sqrt(np.mean(th_sq)))
        xr.append(np.sqrt(np.mean(x_sq)))
        en.append(np.mean(e_sq))
    return {"return_mean": float(np.mean(rets)), "rmse_theta": float(np.mean(th)),
            "rmse_x": float(np.mean(xr)), "energy_mean": float(np.mean(en)),
            "success_rate": float(np.mean(ok))}


rows = []
for sc, dr, pb in SCENARIOS:
    print(f"=== {sc} ===", flush=True)
    for R in R_LEVELS:
        m = run(R, dr, pb)
        rows.append({"scenario": sc, "base_R": R, **m})
        print(f"  R={R:8g}  rmse_th={m['rmse_theta']:.4f}  rmse_x={m['rmse_x']:.4f} "
              f" E={m['energy_mean']:8.3f}  succ={m['success_rate']:.3f} "
              f" ret={m['return_mean']:9.2f}", flush=True)

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "base_R", "return_mean", "rmse_theta",
                                      "rmse_x", "energy_mean", "success_rate"])
    w.writeheader(); w.writerows(rows)
print(f"\nGuardado: {OUT_CSV}", flush=True)
