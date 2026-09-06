"""Sonda 0 + E de D (version seguimiento de referencia).

Tarea: llevar el carro de x=0 a x_ref con el pendulo arriba, sujeto a
|x_cart| <= L. Simulador propio (rk4_step + LQR + ConstrainedMPC + EKF de P1),
sin ResidualBalanceEnv.

Compara, bajo dominio aleatorizado + ruido de medicion + EKF:
  - LQR de seguimiento (coste Q_P2, R_P2), ciego a la restriccion -> overshoot
  - MPC de seguimiento con |x|<=L (holgura L1) -> moldea la aproximacion
  - LQR + residuo oraculo r*=clip(u_MPC-u_LQR) acotado a residual_limit

VERDE (fijado antes de ver datos): el LQR+oraculo acotado reduce la fraccion
que viola L en >= 50 % respecto al LQR Y reduce el overshoot medio en >= 50 %,
con |need| mediana <= residual_limit.

Salida: results/processed/D_setpoint_probe.csv
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
OUT = ROOT / "results" / "processed" / "D_setpoint_probe.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DT = 0.02
T_FINAL = 6.0
NSTEP = int(round(T_FINAL / DT))
X_REF = 0.25
L = 0.30
ACT_LIM = 12.0
MEAS_STD = 1e-3
N_EVAL = 64
EVAL_SEED0 = 9000
RESID_LIMS = [4.0, 8.0, 12.0, 20.0]
TANH1 = float(np.tanh(1.0))

NOM = load_pendulum_params()
DR = {"Mp": (0.80, 1.20), "Jp": (0.75, 1.25), "l": (0.85, 1.15), "kt": (0.90, 1.10)}

# Controladores construidos UNA vez (usan parametros nominales, no la planta
# muestreada). Construir el QP de CVXPY es caro: no repetir por realizacion.
LQR = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=X_REF)
print("construyendo MPC restringido (CVXPY)...", flush=True)
MPC = ConstrainedMPC(NOM, dt=DT, x_limit=L, x_ref=X_REF)
print("listo\n", flush=True)


def sample_params(rng):
    f = {k: rng.uniform(*v) for k, v in DR.items()}
    return replace(NOM, **{k: getattr(NOM, k) * v for k, v in f.items()})


def simulate(kind, resid_lim, seed):
    rng = np.random.default_rng(seed)
    p = sample_params(rng)
    x = np.array([0.0, np.pi + rng.uniform(-0.05, 0.05), 0.0, 0.0])
    xhat = x.copy()
    P = np.eye(4) * 1e-3
    Rm = np.diag([MEAS_STD**2, MEAS_STD**2])
    ekf = ExtendedKalmanFilter(NOM, R=Rm)

    xs, ths, us, needs = [], [], [], []
    sat = 0
    for k in range(NSTEP):
        ub = LQR.compute(k * DT, xhat)
        if kind == "lqr":
            u = ub
        else:
            um = MPC.compute(k * DT, xhat)
            target = um - ub
            needs.append(abs(target))
            if kind == "mpc":
                u = um
            else:
                cap = TANH1 * resid_lim
                if abs(target) > cap:
                    sat += 1
                r = float(np.clip(target, -cap, cap))
                u = ub + r
        u = float(np.clip(u, -ACT_LIM, ACT_LIM))
        y = x[[0, 1]] + rng.normal(0.0, MEAS_STD, size=2)
        xhat, P, _ = ekf.step(xhat, P, u, y, DT)
        x = rk4_step(x, u, p, DT)
        xs.append(x[0]); ths.append((x[1] - np.pi) ** 2); us.append(u)

    xs = np.array(xs); ths = np.array(ths); us = np.array(us)
    peak = float(xs.max())
    err = np.abs(xs - X_REF)
    idx = np.where(err < 0.02)[0]
    settle_t = idx[0] * DT if len(idx) and np.all(err[idx[0]:] < 0.05) else T_FINAL
    return {
        "peak_x": peak,
        "overshoot": max(0.0, peak - X_REF),
        "violate": int(peak > L),
        "settle_t": float(settle_t),
        "rmse_theta": float(np.sqrt(ths.mean())),
        "energy": float(np.sum(us**2) * DT),
        "need_mean": float(np.mean(needs)) if needs else 0.0,
        "sat_frac": sat / NSTEP,
    }


def batch(kind, resid_lim=None):
    rs = [simulate(kind, resid_lim, EVAL_SEED0 + i) for i in range(N_EVAL)]
    agg = {}
    for key in ("peak_x", "overshoot", "settle_t", "rmse_theta", "energy", "need_mean", "sat_frac"):
        agg[key + "_mean"] = float(np.mean([r[key] for r in rs]))
    agg["violate_frac"] = float(np.mean([r["violate"] for r in rs]))
    agg["overshoot_p90"] = float(np.percentile([r["overshoot"] for r in rs], 90))
    pos = [r["need_mean"] for r in rs if r["need_mean"] > 0]
    agg["need_p50"] = float(np.percentile(pos, 50)) if pos else 0.0
    agg["need_p90"] = float(np.percentile(pos, 90)) if pos else 0.0
    return agg


rows = []
print(f"x_ref={X_REF}  L={L}  N={N_EVAL}  t={T_FINAL}s\n", flush=True)

m = batch("lqr")
rows.append({"method": "LQR_seguimiento", "resid_lim": "", **m})
lqr_v, lqr_os = m["violate_frac"], m["overshoot_mean"]
print(f"LQR seguim.  viola={m['violate_frac']:.3f}  overshoot={m['overshoot_mean']:.3f} "
      f"(p90 {m['overshoot_p90']:.3f})  peak={m['peak_x_mean']:.3f}  settle={m['settle_t_mean']:.2f}s "
      f"rmseTh={m['rmse_theta_mean']:.4f}  E={m['energy_mean']:.1f}", flush=True)

m = batch("mpc")
rows.append({"method": "MPC_restringido", "resid_lim": "", **m})
print(f"MPC restr.   viola={m['violate_frac']:.3f}  overshoot={m['overshoot_mean']:.3f} "
      f"(p90 {m['overshoot_p90']:.3f})  peak={m['peak_x_mean']:.3f}  settle={m['settle_t_mean']:.2f}s "
      f"rmseTh={m['rmse_theta_mean']:.4f}  E={m['energy_mean']:.1f}  "
      f"|need|={m['need_mean_mean']:.2f} (p50 {m['need_p50']:.2f} p90 {m['need_p90']:.2f})", flush=True)

for rl in RESID_LIMS:
    m = batch("oracle", rl)
    rows.append({"method": "LQR+oraculo", "resid_lim": rl, **m})
    print(f"LQR+orac {rl:4.0f}V viola={m['violate_frac']:.3f}  overshoot={m['overshoot_mean']:.3f} "
          f"peak={m['peak_x_mean']:.3f}  settle={m['settle_t_mean']:.2f}s  rmseTh={m['rmse_theta_mean']:.4f}  "
          f"|need|={m['need_mean_mean']:.2f}  sat={m['sat_frac_mean']:.2f}", flush=True)

keys = sorted({k for r in rows for k in r})
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"\nGuardado: {OUT}", flush=True)

orac = [r for r in rows if r["method"] == "LQR+oraculo"]
best = min(orac, key=lambda r: (r["violate_frac"], r["overshoot_mean"]))
rv = 1 - best["violate_frac"] / lqr_v if lqr_v > 0 else float("nan")
ro = 1 - best["overshoot_mean"] / lqr_os if lqr_os > 0 else float("nan")
print(f"\nLQR viola {lqr_v:.2f} os {lqr_os:.3f}  ->  mejor oraculo ({best['resid_lim']}V) "
      f"viola {best['violate_frac']:.2f} os {best['overshoot_mean']:.3f}  "
      f"(red viola {rv*100:.0f}%, red os {ro*100:.0f}%, need p50 {best['need_p50']:.1f}V)", flush=True)
verde = (rv >= 0.50) and (ro >= 0.50) and (best["need_p50"] <= best["resid_lim"])
print(f">>> {'VERDE: entrenar D (seguimiento)' if verde else 'ROJO: no entrenar; D pasa a estudio comparativo LQR vs MPC'}", flush=True)
