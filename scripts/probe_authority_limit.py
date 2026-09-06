"""Sonda de limite de autoridad.

El oraculo con |r|<=4V no captura el margen. Pregunta: es el limite de
autoridad la restriccion que ata? Se repite el residuo oraculo
u_LQR(R*) - u_LQR(0.01) variando residual_limit en {4, 8, 12} V, a rho=1.0
(sin compuerta), con la penalizacion de residuo desactivada.

Si con residual_limit=12 el residuo convierte el base R=0.01 en el
comportamiento de R=10 (rmse_th ~ 0.010), el limite de autoridad ES la
restriccion. Si no, el resultado negativo es estructural.

Salida: results/processed/authority_limit_probe.csv
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
OUT_CSV = ROOT / "results" / "processed" / "authority_limit_probe.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL     = 48
LIMITS     = [4.0, 8.0, 12.0]
R_TARGETS  = [1.0, 10.0]
PARAMS     = load_pendulum_params()

SCEN = [("SC1_nominal", False, 0.0), ("SC3_dr_perturb", True, 0.5)]


def run(residual_limit, R_target, dr, perturb, n=N_EVAL, seed0=EVAL_SEED0):
    # rho fijo a 1.0 -> GateMode.MAXIMUM con rho_max=1.0
    from igrrl.controller import GateConfig
    gc = GateConfig(rho_max=1.0)
    env = ResidualBalanceEnv(mode=GateMode.MAXIMUM, domain_randomization=dr,
                             perturbation_probability=perturb, base_R=0.01,
                             residual_penalty=0.0, residual_limit=residual_limit,
                             gate_config=gc)
    tgt = None if R_target is None else LQRController(PARAMS, R=R_target)
    th_rms, x_rms, ok, rsat = [], [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        done, term = False, False
        th_sq, x_sq, sat = [], [], []
        while not done:
            if tgt is None:
                raw = 0.0
            else:
                u_t = tgt.compute(env.steps * env.dt, env.xhat.copy())
                u_b = env.lqr.compute(env.steps * env.dt, env.xhat.copy())
                want = np.clip((u_t - u_b) / env.residual_limit, -0.999999, 0.999999)
                raw = float(np.arctanh(want))
            obs, rew, term, trunc, info = env.step(np.array([raw], dtype="float32"))
            done = term or trunc
            th_sq.append((env.state[1] - np.pi) ** 2)
            x_sq.append(env.state[0] ** 2)
            sat.append(1.0 if abs(info["residual"]) > 0.98 * residual_limit else 0.0)
        th_rms.append(np.sqrt(np.mean(th_sq)))
        x_rms.append(np.sqrt(np.mean(x_sq)))
        rsat.append(np.mean(sat))
        ok.append(0 if term else 1)
    return {"rmse_theta": float(np.mean(th_rms)), "rmse_x": float(np.mean(x_rms)),
            "residual_sat_frac": float(np.mean(rsat)), "success_rate": float(np.mean(ok))}


rows = []
for sc, dr, pb in SCEN:
    print(f"\n=== {sc} ===", flush=True)
    base = run(4.0, None, dr, pb)
    rows.append({"scenario": sc, "residual_limit": "-", "R_target": "none(LQR 0.01)", **base})
    print(f"  base LQR R=0.01           rmse_th={base['rmse_theta']:.4f}  rmse_x={base['rmse_x']:.4f}", flush=True)
    for Rt in R_TARGETS:
        ref = run(4.0, None, dr, pb)  # placeholder; direct LQR(R*) ref below
        # referencia directa: LQR(R*) puro
        env_ref = ResidualBalanceEnv(mode=GateMode.MINIMUM, domain_randomization=dr,
                                     perturbation_probability=pb, base_R=Rt,
                                     residual_penalty=0.0)
        th_r, x_r = [], []
        for ep in range(N_EVAL):
            o, _ = env_ref.reset(seed=EVAL_SEED0 + ep)
            d, t = False, False
            ts, xs = [], []
            while not d:
                o, r, t, tr, i = env_ref.step(np.zeros(1, dtype="float32"))
                d = t or tr
                ts.append((env_ref.state[1] - np.pi) ** 2); xs.append(env_ref.state[0] ** 2)
            th_r.append(np.sqrt(np.mean(ts))); x_r.append(np.sqrt(np.mean(xs)))
        print(f"  [ref] LQR R={Rt:g} puro        rmse_th={np.mean(th_r):.4f}  rmse_x={np.mean(x_r):.4f}", flush=True)
        rows.append({"scenario": sc, "residual_limit": "ref", "R_target": f"LQR {Rt:g} puro",
                     "rmse_theta": float(np.mean(th_r)), "rmse_x": float(np.mean(x_r)),
                     "residual_sat_frac": float("nan"), "success_rate": 1.0})
        for lim in LIMITS:
            m = run(lim, Rt, dr, pb)
            rows.append({"scenario": sc, "residual_limit": lim, "R_target": f"oracle->R{Rt:g}", **m})
            print(f"  oracle->R{Rt:g}  lim={lim:4.0f}V   rmse_th={m['rmse_theta']:.4f}  "
                  f"rmse_x={m['rmse_x']:.4f}  sat={m['residual_sat_frac']:.2f}  "
                  f"succ={m['success_rate']:.2f}", flush=True)

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "residual_limit", "R_target",
                                      "rmse_theta", "rmse_x", "residual_sat_frac", "success_rate"])
    w.writeheader(); w.writerows(rows)
print(f"\nGuardado: {OUT_CSV}", flush=True)
