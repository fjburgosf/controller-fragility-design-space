"""Figura 6. Respuesta temporal de las tres familias a un lado y otro de la frontera.

Misma realizacion para todas las familias, de modo que las diferencias sean del
controlador y no del sorteo. Se muestran angulo, posicion del carro y accion de
control, dentro de la distribucion de entrenamiento y fuera de ella.
"""
import sys
from dataclasses import replace
from pathlib import Path

_R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_R / "src")); sys.path.insert(0, str(_R))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_pendulum_params
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.evaluate_common import (make_lqr, make_policy, NOM, DT, MEAS_STD,
                                   ACT_LIM, RAIL, DR_RANGES, EVAL_SEED0)
from igrrl.mpc_design import DesignMPC

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
T_FINAL = 6.0
NSTEP = int(round(T_FINAL / DT))
K_ON, K_OFF = int(round(1.0 / DT)), int(round(1.4 / DT))
COL = {"LQR": "#2E7D32", "MPC": "#6A1B9A", "SAC": "#1565C0", "DDPG": "#EF6C00"}


def traza(ctrl, dv, seed):
    rng = np.random.default_rng(seed)
    p = replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR_RANGES.items()})
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xh = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MEAS_STD**2] * 2))
    th, cx, cu = [], [], []
    for k in range(NSTEP):
        if not np.all(np.isfinite(x)):
            break
        u = float(np.clip(ctrl(k * DT, xh), -ACT_LIM, ACT_LIM))
        d = dv if K_ON <= k < K_OFF else 0.0
        y = x[[0, 1]] + rng.normal(0, MEAS_STD, 2)
        xh, P, _ = ekf.step(xh, P, u, y, DT)
        x = rk4_step(x, u + d, p, DT)
        th.append(np.arctan2(np.sin(x[1] - np.pi), np.cos(x[1] - np.pi)))
        cx.append(x[0]); cu.append(u)
    t = np.arange(len(th)) * DT
    return t, np.array(th), np.array(cx), np.array(cu)


from stable_baselines3 import SAC, DDPG

mpc = DesignMPC(NOM, dt=0.01, T_pred=0.8, Nc=20, blocking="front", terminal="riccati")
ctrls = {"LQR": make_lqr(), "MPC": lambda t, xh: mpc.compute(t, xh)}
for algo, cls in (("SAC", SAC), ("DDPG", DDPG)):
    f = ROOT / "results" / "training" / f"standalone_{algo.lower()}" / "seed_1" / "best_model.zip"
    if f.exists():
        ctrls[algo] = make_policy(cls.load(str(f.with_suffix(""))))

SEED = EVAL_SEED0 + 3
fig, ax = plt.subplots(3, 2, figsize=(11.4, 7.4), sharex=True)
for c, dv in enumerate((6.0, 8.0)):
    for nom, ctrl in ctrls.items():
        t, th, cx, cu = traza(ctrl, dv, SEED)
        for r, serie in enumerate((th, cx, cu)):
            ax[r, c].plot(t, serie, color=COL[nom], lw=1.5, label=nom if r == 0 else None)
    for r in range(3):
        ax[r, c].axvspan(1.0, 1.4, color="0.85", zorder=0)
        ax[r, c].grid(alpha=0.3); ax[r, c].tick_params(labelsize=8)
    ax[1, c].axhline(RAIL, color="#B71C1C", ls="--", lw=1.2)
    ax[1, c].axhline(-RAIL, color="#B71C1C", ls="--", lw=1.2)
    ax[2, c].axhline(ACT_LIM, color="0.4", ls=":", lw=1.1)
    ax[2, c].axhline(-ACT_LIM, color="0.4", ls=":", lw=1.1)
    dentro = dv <= 6
    ax[0, c].set_title(f"$d_v$ = {dv:.0f} V, {'inside' if dentro else 'outside'} training range",
                       fontsize=10, color="#2E7D32" if dentro else "#B71C1C")
    ax[2, c].set_xlabel("time [s]", fontsize=9)

ax[0, 0].set_ylabel("angular error [rad]", fontsize=9)
ax[1, 0].set_ylabel("cart position [m]", fontsize=9)
ax[2, 0].set_ylabel("control [V]", fontsize=9)
ax[0, 0].legend(fontsize=8, ncol=4, loc="upper left")
for c in range(2):
    ax[0, c].text(1.2, 0.955, "gust", fontsize=8, ha="center", va="top",
                  color="#37474F", weight="bold", transform=ax[0, c].get_xaxis_transform())
for c in range(2):
    ax[1, c].text(5.95, RAIL, "rail", fontsize=8, color="#B71C1C", ha="right",
                  va="bottom", weight="bold")

fig.suptitle("Closed loop response on the same realisation. Inside the training range the families "
             "agree, outside it they do not", fontsize=10.5, y=0.98)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig6_timeseries.{ext}", bbox_inches="tight", dpi=600)
print("fig6_timeseries guardada")
for dv in (6.0, 8.0):
    for nom, ctrl in ctrls.items():
        _, th, cx, _ = traza(ctrl, dv, SEED)
        print(f"  dv={dv:.0f} {nom:5} |theta|max {np.abs(th).max():7.3f} rad   |x|max {np.abs(cx).max():6.3f} m")
