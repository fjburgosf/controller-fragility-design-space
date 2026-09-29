"""Coste computacional por paso de control, medido en lazo cerrado real.

Corrige un defecto de la medicion anterior. Repetir el MISMO estado deja al MPC
resolver un QP identico con warm start, que converge en casi cero iteraciones y
subestima su coste. Aqui cada llamada recibe un estado distinto, el que produce la
trayectoria real, que es la condicion de uso.

Se reporta la mediana y el percentil 95 por paso, porque en un lazo embebido lo
que limita es el peor caso y no el promedio. Se mide tambien el estimador por
separado, ya que todas las familias lo pagan por igual.

Salida. tables/P2/p2_tabla6_coste.{csv,md,tex}
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_pendulum_params
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.evaluate_common import make_lqr, make_policy, NOM, DT, MEAS_STD, ACT_LIM, DR_RANGES
from igrrl.mpc_design import DesignMPC

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "tables" / "P2"; TAB.mkdir(parents=True, exist_ok=True)
NSTEP, DV = 300, 6.0
K_ON, K_OFF = int(round(1.0 / DT)), int(round(1.4 / DT))


def recorre(ctrl, seed=9000):
    """Simula un episodio y devuelve el tiempo de CADA llamada al controlador."""
    rng = np.random.default_rng(seed)
    p = replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR_RANGES.items()})
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xh = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MEAS_STD**2] * 2))
    t_ctrl, t_ekf = [], []
    for k in range(NSTEP):
        t0 = time.perf_counter()
        u = float(np.clip(ctrl(k * DT, xh), -ACT_LIM, ACT_LIM))
        t_ctrl.append((time.perf_counter() - t0) * 1000)
        d = DV if K_ON <= k < K_OFF else 0.0
        y = x[[0, 1]] + rng.normal(0, MEAS_STD, 2)
        t0 = time.perf_counter()
        xh, P, _ = ekf.step(xh, P, u, y, DT)
        t_ekf.append((time.perf_counter() - t0) * 1000)
        x = rk4_step(x, u + d, p, DT)
        if not np.all(np.isfinite(x)):
            break
    return np.array(t_ctrl), np.array(t_ekf)


from stable_baselines3 import SAC, DDPG

ctrls = {"LQR": make_lqr()}
mpc = DesignMPC(NOM, dt=0.01, T_pred=0.8, Nc=20, blocking="front", terminal="riccati")
ctrls["MPC"] = lambda t, xh: mpc.compute(t, xh)
for algo, cls in (("SAC", SAC), ("DDPG", DDPG)):
    f = ROOT / "results" / "training" / f"standalone_{algo.lower()}" / "seed_1" / "best_model.zip"
    if f.exists():
        ctrls[algo] = make_policy(cls.load(str(f.with_suffix(""))))

LIBRE = {"LQR": "0", "MPC": "6", "SAC": "hyperparameters and seed",
         "DDPG": "hyperparameters and seed"}

filas, ekf_ref = [], None
print(f"medicion en lazo cerrado, {NSTEP} pasos por controlador\n")
print(f"{'controlador':<8} {'mediana':>9} {'p95':>9} {'max':>9}   (ms por paso)")
for nom, c in ctrls.items():
    tc, te = recorre(c)
    if ekf_ref is None:
        ekf_ref = np.median(te)
    print(f"{nom:<8} {np.median(tc):9.3f} {np.percentile(tc,95):9.3f} {tc.max():9.3f}")
    filas.append({"Controller": nom,
                  "Median [ms]": f"{np.median(tc):.3f}",
                  "P95 [ms]": f"{np.percentile(tc, 95):.3f}",
                  "Max [ms]": f"{tc.max():.3f}",
                  "Design variables exposed": LIBRE.get(nom, "")})

base = float(filas[0]["Median [ms]"])
for f in filas:
    f["Relative to LQR"] = f"{float(f['Median [ms]'])/base:.0f}x"
orden = ["Controller", "Median [ms]", "P95 [ms]", "Max [ms]", "Relative to LQR",
         "Design variables exposed"]
df = pd.DataFrame(filas)[orden]

pie = ("Computational cost per control step measured along a closed loop trajectory, so "
       "that each call solves a different problem, together with the number of design "
       f"decisions each family exposes. The state estimator adds {ekf_ref:.3f} ms per step "
       "for every family alike. Values come from a desktop processor and are indicative "
       "of relative rather than absolute embedded cost.")
df.to_csv(TAB / "p2_tabla6_coste.csv", index=False)
(TAB / "p2_tabla6_coste.md").write_text(f"**{pie}**\n\n" + df.to_markdown(index=False), encoding="utf-8")
(TAB / "p2_tabla6_coste.tex").write_text(
    df.to_latex(index=False, escape=False, caption=pie, label="tab:p2_tabla6_coste"), encoding="utf-8")
print(f"\nEKF comun a todas las familias  {ekf_ref:.3f} ms por paso")
print("tabla 6 regenerada")
