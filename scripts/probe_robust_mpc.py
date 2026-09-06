"""Sonda de headroom — MPC robusto por estrechamiento de restriccion (tube proxy).

Diseno congelado en docs/PROBE_ROBUST_MPC.md. Regulacion x_ref=0, |x|<=L, bajo
perturbacion sostenida NO medida (offset de voltaje d_v) + dominio aleatorizado
+ ruido + EKF. Compara LQR, MPC(L), MPC(L-0.05), MPC(L-0.10). MPC construido UNA
vez por x_limit (CVXPY caro). Criterio VERDE fijado antes de ver datos.
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
OUT = ROOT / "results" / "processed" / "robust_mpc_probe.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DT = 0.02
T_FINAL = 5.0
NSTEP = int(round(T_FINAL / DT))
X_REF = 0.0
L = 0.30
DELTAS = [0.0, 0.05, 0.10]      # 0.0 = MPC nominal; >0 = estrechado (robusto)
DV_LEVELS = [2.0, 4.0, 6.0, 8.0]
ACT_LIM = 12.0
MEAS_STD = 1e-3
N_EVAL = 32
EVAL_SEED0 = 9000

NOM = load_pendulum_params()
DR = {"Mp": (0.80, 1.20), "Jp": (0.75, 1.25), "l": (0.85, 1.15), "kt": (0.90, 1.10)}

LQR = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=X_REF)
print("construyendo 3 MPC (CVXPY, una vez c/u)...", flush=True)
MPCS = {d: ConstrainedMPC(NOM, dt=DT, x_limit=L - d, x_ref=X_REF) for d in DELTAS}
print("listo\n", flush=True)


def sample_params(rng):
    return replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR.items()})


def simulate(method, dv, seed):
    """method: 'lqr' o ('mpc', delta). Perturbacion d_v sostenida NO medida."""
    rng = np.random.default_rng(seed)
    p = sample_params(rng)
    x = np.array([0.0, np.pi + rng.uniform(-0.03, 0.03), 0.0, 0.0])
    xhat = x.copy()
    P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MEAS_STD**2, MEAS_STD**2]))

    maxx = 0.0
    ths, us = [], []
    sat = 0
    for k in range(NSTEP):
        if method == "lqr":
            u = LQR.compute(k * DT, xhat)
        else:
            u = MPCS[method[1]].compute(k * DT, xhat)
        u = float(np.clip(u, -ACT_LIM, ACT_LIM))
        if abs(u) >= ACT_LIM - 1e-6:
            sat += 1
        # EKF ve solo u (perturbacion no medida); la planta ve u + dv
        y = x[[0, 1]] + rng.normal(0.0, MEAS_STD, size=2)
        xhat, P, _ = ekf.step(xhat, P, u, y, DT)
        x = rk4_step(x, u + dv, p, DT)
        maxx = max(maxx, abs(x[0]))
        ths.append((x[1] - np.pi) ** 2); us.append(u)

    ths = np.array(ths); us = np.array(us)
    return {
        "maxx": maxx,
        "violate": int(maxx > L),
        "rmse_theta": float(np.sqrt(ths.mean())),
        "energy": float(np.sum(us**2) * DT),
        "sat_frac": sat / NSTEP,
    }


def batch(method, dv):
    rs = [simulate(method, dv, EVAL_SEED0 + i) for i in range(N_EVAL)]
    return {
        "violate_frac": float(np.mean([r["violate"] for r in rs])),
        "p95_maxx": float(np.percentile([r["maxx"] for r in rs], 95)),
        "mean_maxx": float(np.mean([r["maxx"] for r in rs])),
        "rmse_theta": float(np.mean([r["rmse_theta"] for r in rs])),
        "energy": float(np.mean([r["energy"] for r in rs])),
        "sat_frac": float(np.mean([r["sat_frac"] for r in rs])),
    }


METHODS = [("LQR", "lqr")] + [(f"MPC(L-{d:.2f})", ("mpc", d)) for d in DELTAS]
rows = []
print(f"x_ref={X_REF}  L={L}  N={N_EVAL}  t={T_FINAL}s  deltas={DELTAS}\n", flush=True)
print(f"{'dv':>4} {'metodo':>12} {'viol':>6} {'p95|x|':>7} {'med|x|':>7} {'rmseTh':>7} {'E':>6} {'sat':>5}", flush=True)
for dv in DV_LEVELS:
    for name, m in METHODS:
        a = batch(m, dv)
        a.update(dv=dv, method=name)
        rows.append(a)
        print(f"{dv:4.1f} {name:>12} {a['violate_frac']:6.3f} {a['p95_maxx']:7.3f} "
              f"{a['mean_maxx']:7.3f} {a['rmse_theta']:7.4f} {a['energy']:6.1f} {a['sat_frac']:5.2f}", flush=True)
    print("", flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dv", "method", "violate_frac", "p95_maxx",
                                      "mean_maxx", "rmse_theta", "energy", "sat_frac"])
    w.writeheader(); w.writerows(rows)

# --- Evaluacion criterio VERDE al nivel de d_v donde MPC(L) viola mas ---
nomrows = [r for r in rows if r["method"] == "MPC(L-0.00)"]
worst = max(nomrows, key=lambda r: r["violate_frac"])
dv_w = worst["dv"]
vf_nom = worst["violate_frac"]; rt_nom = worst["rmse_theta"]
tight = [r for r in rows if r["dv"] == dv_w and r["method"].startswith("MPC(L-") and r["method"] != "MPC(L-0.00)"]
print(f"=== VERDE @ dv={dv_w} (MPC nominal viola mas) ===", flush=True)
print(f"MPC(L) viola={vf_nom:.3f}  rmseTh={rt_nom:.4f}", flush=True)
P = vf_nom >= 0.20
best = min(tight, key=lambda r: r["violate_frac"]) if tight else None
a = (best is not None) and (best["violate_frac"] <= 0.5 * vf_nom)
b = (best is not None) and (best["rmse_theta"] <= 1.5 * rt_nom)
if best:
    print(f"mejor estrechado: {best['method']} viola={best['violate_frac']:.3f} "
          f"rmseTh={best['rmse_theta']:.4f}", flush=True)
print(f"P(precond viol>=0.20)={P}  a(viola<=0.5xnom)={a}  b(rmseTh<=1.5xnom)={b}", flush=True)
verde = P and a and b
print(f"\n>>> {'VERDE: Opcion 1 (MPC robusto con restricciones)' if verde else 'ROJO'}", flush=True)
if not verde:
    if not P:
        print("    motivo: !P -> ni bajo perturbacion la restriccion es el gap; replantear escenario", flush=True)
    elif not a:
        print("    motivo: !a -> estrechar no reduce peor-caso -> Opcion 2 (comparativo dif. de P1)", flush=True)
    elif not b:
        print("    motivo: !b -> reduce violacion pero destruye estabilizacion", flush=True)
