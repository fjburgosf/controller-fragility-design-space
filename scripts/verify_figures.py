"""Verificacion de las AFIRMACIONES NUMERICAS asociadas a cada figura.

Cada numero que el texto atribuye a una figura se recomputa aqui desde su fuente,
con el mismo codigo que genera la figura, y se compara contra el manuscrito. Si la
figura y el parrafo se desincronizan, esto lo detecta.
"""
from __future__ import annotations

import re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
TXT = (ROOT / "paper" / "manuscript_en.md").read_text(encoding="utf-8")

OK, MAL = [], []


def chk(nombre, esperado, calculado, tol):
    bien = abs(esperado - calculado) <= tol
    (OK if bien else MAL).append((nombre, f"texto {esperado}  recomputado {calculado:.4f}"))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<52} texto {esperado:<8} recomputado {calculado:.4f}")


def en_texto(frag):
    bien = frag in TXT
    (OK if bien else MAL).append((f"literal: {frag[:44]}", "presente" if bien else "AUSENTE"))
    print(f"  [{'ok ' if bien else 'MAL'}] literal en el texto                                  \"{frag[:52]}\"")


print("=" * 100)
print("FIGURA 4. PERCENTILES DE COLA POR FAMILIA Y NIVEL DE PERTURBACION")
print("=" * 100)
e = pd.read_csv(ROOT / "results" / "processed" / "final_episodes.csv")
e["fam"] = e.method.str.replace(r"_s\d+", "", regex=True)
p95 = {(dv, f): np.percentile(e[(e.dv == dv) & (e.fam == f)].rmse_theta, 95)
       for dv in sorted(e.dv.unique()) for f in ("LQR", "MPC", "SAC", "DDPG")
       if len(e[(e.dv == dv) & (e.fam == f)])}

for dv, lo, hi in ((4.0, 0.041, 0.050), (6.0, 0.060, 0.073)):
    vals = [p95[(dv, f)] for f in ("LQR", "MPC", "SAC", "DDPG")]
    chk(f"dv={dv:.0f} minimo del rango citado", lo, min(vals), 5e-4)
    chk(f"dv={dv:.0f} maximo del rango citado", hi, max(vals), 5e-4)

chk("dv=8 p95 LQR", 0.079, p95[(8.0, "LQR")], 5e-4)
chk("dv=8 p95 MPC", 0.080, p95[(8.0, "MPC")], 5e-4)
chk("dv=8 p95 SAC", 53.1, p95[(8.0, "SAC")], 0.05)
chk("dv=8 p95 DDPG", 4.9, p95[(8.0, "DDPG")], 0.05)
chk("razon SAC frente al regulador", 670, p95[(8.0, "SAC")] / p95[(8.0, "LQR")], 15)
chk("razon DDPG frente al regulador", 60, p95[(8.0, "DDPG")] / p95[(8.0, "LQR")], 5)
en_texto("0.041 rad to 0.050 rad")
en_texto("0.060 rad to 0.073 rad")
en_texto("0.079 rad and 0.080 rad")
en_texto("53.1 rad and deep deterministic policy gradient to\n4.9 rad")

print()
print("=" * 100)
print("FIGURA 5. RESPUESTA TEMPORAL SOBRE UNA MISMA REALIZACION")
print("=" * 100)
import importlib.util
spec = importlib.util.spec_from_file_location("_f5", ROOT / "scripts" / "fig5_timeseries.py")
# reutilizar la funcion de trazado sin volver a dibujar seria fragil, asi que se
# repite aqui el mismo calculo con las mismas constantes del modulo de evaluacion.
from dataclasses import replace
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.evaluate_common import (make_lqr, make_policy, NOM, DT, MEAS_STD,
                                   ACT_LIM, RAIL, DR_RANGES, EVAL_SEED0)
from igrrl.mpc_design import DesignMPC
from stable_baselines3 import SAC, DDPG

NSTEP = int(round(6.0 / DT))
K_ON, K_OFF = int(round(1.0 / DT)), int(round(1.4 / DT))


def traza(ctrl, dv, seed):
    rng = np.random.default_rng(seed)
    p = replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR_RANGES.items()})
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xh = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MEAS_STD**2] * 2))
    th, cx = [], []
    for k in range(NSTEP):
        if not np.all(np.isfinite(x)):
            break
        u = float(np.clip(ctrl(k * DT, xh), -ACT_LIM, ACT_LIM))
        d = dv if K_ON <= k < K_OFF else 0.0
        y = x[[0, 1]] + rng.normal(0, MEAS_STD, 2)
        xh, P, _ = ekf.step(xh, P, u, y, DT)
        x = rk4_step(x, u + d, p, DT)
        th.append(np.arctan2(np.sin(x[1] - np.pi), np.cos(x[1] - np.pi)))
        cx.append(x[0])
    return np.abs(th).max(), np.abs(cx).max()


mpc = DesignMPC(NOM, dt=0.01, T_pred=0.8, Nc=20, blocking="front", terminal="riccati")
ctrls = {"LQR": make_lqr(), "MPC": lambda t, xh: mpc.compute(t, xh)}
for algo, cls in (("SAC", SAC), ("DDPG", DDPG)):
    f = ROOT / "results" / "training" / f"standalone_{algo.lower()}" / "seed_1" / "best_model.zip"
    if f.exists():
        ctrls[algo] = make_policy(cls.load(str(f.with_suffix(""))))

SEED = EVAL_SEED0 + 3
r = {(dv, n): traza(c, dv, SEED) for dv in (6.0, 8.0) for n, c in ctrls.items()}
th6 = [r[(6.0, n)][0] for n in ctrls]
chk("dv=6 excursion angular minima", 0.211, min(th6), 5e-4)
chk("dv=6 excursion angular maxima", 0.279, max(th6), 5e-4)
chk("dv=6 el carro se queda dentro del riel",
    1.0, float(max(r[(6.0, n)][1] for n in ctrls) < RAIL), 0.0)
chk("dv=8 pico angular LQR", 0.391, r[(8.0, "LQR")][0], 5e-4)
chk("dv=8 pico angular MPC", 0.385, r[(8.0, "MPC")][0], 5e-4)
chk("dv=8 ambos agentes superan 3 rad",
    1.0, float(min(r[(8.0, n)][0] for n in ("SAC", "DDPG")) > 3.0), 0.0)
chk("dv=8 recorrido del carro LQR", 0.432, r[(8.0, "LQR")][1], 5e-4)
chk("dv=8 recorrido del carro DDPG", 2.079, r[(8.0, "DDPG")][1], 5e-4)
chk("recorrido DDPG frente a la media longitud del riel", 7.0, r[(8.0, "DDPG")][1] / RAIL, 0.3)
en_texto("0.211 rad to 0.279 rad")
en_texto("0.391 rad and\n0.385 rad")
en_texto("2.079 m")
en_texto("0.432 m, past the")

print()
print("=" * 100)
print(f"RESUMEN FIGURAS   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 100)
for n, d in MAL:
    print(f"  DISCREPANCIA  {n}\n                {d}")
