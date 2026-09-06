"""Sonda de residuo oraculo.

Pregunta decisiva: dentro de los limites de autoridad del entorno, existe un
residuo que mejore la tarea, y la recompensa tal como esta escrita lo premia?

Se construye un residuo analitico r*(x) = u_{LQR,R_target}(x) - u_{LQR,0.01}(x),
saturado por el propio entorno. Es el residuo que un agente perfecto podria
aprender si su objetivo fuera imitar un LQR mejor sintonizado. Se mide:
  - retorno con la recompensa actual (incluye penalizacion 0.05*residuo^2)
  - retorno con la penalizacion de residuo retirada
  - descomposicion de la recompensa por termino

Salida: results/processed/oracle_residual_probe.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv
from src.config import load_pendulum_params
from src.controllers.lqr import LQRController

ROOT    = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "processed" / "oracle_residual_probe.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL     = 48
R_TARGETS  = [0.1, 1.0, 10.0, 100.0]
W_RES      = 0.05   # peso actual de la penalizacion de residuo en env.py

PARAMS = load_pendulum_params()


def reward_terms(state, residual, u):
    th = np.arctan2(np.sin(state[1] - np.pi), np.cos(state[1] - np.pi))
    return {
        "theta": 12.0 * th ** 2,
        "x":      1.0 * state[0] ** 2,
        "xdot":   0.05 * state[2] ** 2,
        "thdot":  0.02 * state[3] ** 2,
        "res":    W_RES * residual ** 2,
        "u":      0.002 * u ** 2,
    }


def run(mode, R_target, dr, perturb, n=N_EVAL, seed0=EVAL_SEED0):
    """R_target=None -> residuo cero (LQR canonico puro)."""
    env = ResidualBalanceEnv(mode=mode, domain_randomization=dr,
                             perturbation_probability=perturb, base_R=0.01)
    tgt = None if R_target is None else LQRController(PARAMS, R=R_target)
    acc = {k: [] for k in ["theta", "x", "xdot", "thdot", "res", "u"]}
    th_rms, x_rms, ok, res_abs, rho_mean = [], [], [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        done, term = False, False
        ep_acc = {k: 0.0 for k in acc}
        th_sq, x_sq, ra, rh = [], [], [], []
        while not done:
            if tgt is None:
                raw = 0.0
            else:
                # residuo deseado en voltios -> accion normalizada via atanh
                u_t = tgt.compute(env.steps * env.dt, env.xhat.copy())
                u_b = env.lqr.compute(env.steps * env.dt, env.xhat.copy())
                want = np.clip((u_t - u_b) / env.residual_limit, -0.999, 0.999)
                raw = float(np.arctanh(want))
            obs, rew, term, trunc, info = env.step(np.array([raw], dtype="float32"))
            done = term or trunc
            t = reward_terms(env.state, info["residual"], info["u_control"])
            for k in acc:
                ep_acc[k] += t[k]
            th_sq.append((env.state[1] - np.pi) ** 2)
            x_sq.append(env.state[0] ** 2)
            ra.append(abs(info["residual"])); rh.append(info["rho"])
        for k in acc:
            acc[k].append(ep_acc[k])
        th_rms.append(np.sqrt(np.mean(th_sq))); x_rms.append(np.sqrt(np.mean(x_sq)))
        res_abs.append(np.mean(ra)); rho_mean.append(np.mean(rh))
        ok.append(0 if term else 1)
    out = {f"cost_{k}": float(np.mean(v)) for k, v in acc.items()}
    total = sum(out.values())
    out["return_actual"]   = -total
    out["return_no_respen"] = -(total - out["cost_res"])
    out["rmse_theta"] = float(np.mean(th_rms))
    out["rmse_x"]     = float(np.mean(x_rms))
    out["res_abs"]    = float(np.mean(res_abs))
    out["rho_mean"]   = float(np.mean(rho_mean))
    out["success_rate"] = float(np.mean(ok))
    return out


rows = []
SCEN = [("SC1_nominal", False, 0.0), ("SC3_dr_perturb", True, 0.5)]
MODES = [("rho_min(a2)", GateMode.MINIMUM), ("rho_max(a3)", GateMode.MAXIMUM)]

for sc, dr, pb in SCEN:
    print(f"\n=== {sc} ===", flush=True)
    base = run(GateMode.MINIMUM, None, dr, pb)
    rows.append({"scenario": sc, "mode": "-", "R_target": "none(LQR)", **base})
    print(f"  LQR puro           ret={base['return_actual']:9.2f} "
          f"(sin pen.res {base['return_no_respen']:9.2f})  rmse_th={base['rmse_theta']:.4f} "
          f" rmse_x={base['rmse_x']:.4f}", flush=True)
    for mname, mode in MODES:
        for Rt in R_TARGETS:
            m = run(mode, Rt, dr, pb)
            rows.append({"scenario": sc, "mode": mname, "R_target": Rt, **m})
            d = m["return_actual"] - base["return_actual"]
            dn = m["return_no_respen"] - base["return_no_respen"]
            print(f"  {mname:12s} R*={Rt:6g}  ret={m['return_actual']:9.2f} (d={d:+7.2f}) "
                  f" sin_pen={m['return_no_respen']:9.2f} (d={dn:+7.2f}) "
                  f" rmse_th={m['rmse_theta']:.4f}  |r|={m['res_abs']:.2f} "
                  f" rho={m['rho_mean']:.2f}  succ={m['success_rate']:.2f}", flush=True)

fields = list(rows[0].keys())
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f"\nGuardado: {OUT_CSV}", flush=True)
