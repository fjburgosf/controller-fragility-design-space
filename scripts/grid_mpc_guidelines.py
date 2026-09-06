"""Rejilla de contraste de las GUIAS DE SINTONIA de MPC (P2, D31).

Formaliza las guias estandar como reglas contrastables y las prueba bajo
perturbacion + incertidumbre parametrica, con metricas de COLA.

  G1 (muestreo): Ts entre 10% y 25% del tiempo de respuesta en lazo cerrado.
                 tau_dominante = 0.725 s  ->  Ts in [0.073, 0.181] s
  G2 (horizonte): T_pred cubre la dinamica dominante (asentamiento 4*tau = 2.9 s)

Hipotesis: G1 es INCORRECTA para esta planta. El sistema tiene escalas separadas
(tau_lento 0.725 s vs tau_rapido 0.007 s, razon ~100); muestrear segun el modo
dominante no puede estabilizar el modo rapido. Y G2 es INSUFICIENTE: cumplirla no
garantiza robustez si el blocking o el coste terminal estan mal elegidos.

Sin restriccion de carro (aisla el efecto de las variables de diseno).
Salida: results/processed/mpc_guidelines_grid.csv
"""
from __future__ import annotations

import csv, itertools, os, time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.mpc_design import DesignMPC, Q_P2, R_P2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / os.environ.get("GRID_OUT", "mpc_guidelines_grid.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

T_FINAL, DV, N_EVAL, SEED0 = 6.0, 8.0, 16, 9000
MS, ACT, RAIL, FALL_TH = 1e-3, 12.0, 0.30, 0.30
TAU_SLOW, TAU_FAST = 0.725, 0.007
G1_LO, G1_HI = 0.10 * TAU_SLOW, 0.25 * TAU_SLOW      # guia de muestreo
G2_MIN = 4 * TAU_SLOW                                 # guia de horizonte

import os
DTS = [float(v) for v in os.environ.get("GRID_DTS", "0.005,0.01,0.02,0.05,0.10").split(",")]
TPREDS = [float(v) for v in os.environ.get("GRID_TPREDS","0.4,0.8,1.5,3.0").split(",")]
BLOCKS = ["front", "uniform"]
TERMS = ["riccati", "stage"]

NOM = load_pendulum_params()
DR = {"Mp": (0.8, 1.2), "Jp": (0.75, 1.25), "l": (0.85, 1.15), "kt": (0.9, 1.1)}
LQR = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=0.0)


def run(ctrl, dt, seed):
    nstep = int(round(T_FINAL / dt))
    k_on, k_off = int(round(1.0 / dt)), int(round(1.4 / dt))
    rng = np.random.default_rng(seed)
    p = replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR.items()})
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xh = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MS**2] * 2))
    mx = 0.0; ths = []; us = []
    for k in range(nstep):
        # guarda de divergencia ANTES de usar el lazo: estado, estimado y covarianza
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(xh)) and np.all(np.isfinite(P))):
            return {"fell": 1, "rmse": float("inf"), "maxx": float("inf"),
                    "E": float("inf"), "diverged": 1}
        try:
            u = float(np.clip(ctrl(k * dt, xh), -ACT, ACT))
            d = DV if k_on <= k < k_off else 0.0
            y = x[[0, 1]] + rng.normal(0, MS, 2)
            xh, P, _ = ekf.step(xh, P, u, y, dt)
            x = rk4_step(x, u + d, p, dt)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return {"fell": 1, "rmse": float("inf"), "maxx": float("inf"),
                    "E": float("inf"), "diverged": 1}
        if not np.all(np.isfinite(x)):
            return {"fell": 1, "rmse": float("inf"), "maxx": float("inf"),
                    "E": float("inf"), "diverged": 1}
        mx = max(mx, abs(x[0])); ths.append((x[1] - np.pi) ** 2); us.append(u)
    r = float(np.sqrt(np.mean(ths)))
    return {"fell": int(r > FALL_TH), "rmse": r, "maxx": mx,
            "E": float(np.sum(np.square(us)) * dt), "diverged": 0}


def agg(ctrl, dt):
    rs = [run(ctrl, dt, SEED0 + i) for i in range(N_EVAL)]
    fin = [r for r in rs if not r["diverged"]]
    def m(key, default=float("nan")):
        return float(np.mean([r[key] for r in fin])) if fin else default
    def p95(key, default=float("nan")):
        return float(np.percentile([r[key] for r in fin], 95)) if fin else default
    return {"fell_frac": float(np.mean([r["fell"] for r in rs])),
            "diverged_frac": float(np.mean([r["diverged"] for r in rs])),
            "rmse_mean": m("rmse"), "rmse_p95": p95("rmse"),
            "maxx_p95": p95("maxx"), "energy": m("E")}


rows = []
print(f"dv={DV} N={N_EVAL} SIN restriccion | G1: Ts in [{G1_LO:.3f},{G1_HI:.3f}] s | G2: T_pred >= {G2_MIN:.2f} s\n", flush=True)
a = agg(lambda t, xh: LQR.compute(t, xh), 0.005)
rows.append({"ctrl": "LQR", "dt": 0.005, "T_pred": "", "blocking": "", "terminal": "",
             "G1_ok": "", "G2_ok": "", "ms_step": 0.0, **a})
print(f"LQR referencia: cae={a['fell_frac']:.2f} rmse={a['rmse_mean']:.4f} p95|x|={a['maxx_p95']:.3f} E={a['energy']:.1f}\n", flush=True)
print(f"{'dt':>6} {'Tpred':>6} {'block':>8} {'term':>8} {'G1':>3} {'G2':>3} {'cae':>6} {'rmse':>7} {'p95|x|':>7} {'E':>7} {'ms':>6}", flush=True)

for dt, Tp, bl, tm in itertools.product(DTS, TPREDS, BLOCKS, TERMS):
    g1 = G1_LO <= dt <= G1_HI
    g2 = Tp >= G2_MIN
    try:
        m = DesignMPC(NOM, dt=dt, T_pred=Tp, Nc=20, blocking=bl, terminal=tm)
    except Exception as ex:
        print(f"{dt:6.3f} {Tp:6.1f} {bl:>8} {tm:>8}  BUILD FAIL {ex.__class__.__name__}", flush=True)
        continue
    t0 = time.time()
    a = agg(lambda t, xh: m.compute(t, xh), dt)
    ms = (time.time() - t0) / (N_EVAL * int(round(T_FINAL / dt))) * 1000
    rows.append({"ctrl": "MPC", "dt": dt, "T_pred": Tp, "blocking": bl, "terminal": tm,
                 "G1_ok": int(g1), "G2_ok": int(g2), "ms_step": round(ms, 2), **a})
    print(f"{dt:6.3f} {Tp:6.1f} {bl:>8} {tm:>8} {int(g1):3d} {int(g2):3d} {a['fell_frac']:6.2f} "
          f"{a['rmse_mean']:7.4f} {a['maxx_p95']:7.3f} {a['energy']:7.1f} {ms:6.2f}", flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ctrl", "dt", "T_pred", "blocking", "terminal", "G1_ok",
                                      "G2_ok", "fell_frac", "diverged_frac", "rmse_mean", "rmse_p95",
                                      "maxx_p95", "energy", "ms_step"])
    w.writeheader(); w.writerows(rows)
print(f"\nGuardado: {OUT}", flush=True)
