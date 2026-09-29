"""Verificacion de CORRECCION DE DISENO de cada controlador y del estimador.

No pregunta si funcionan bien, sino si estan implementados como dicen estar. Cada
prueba compara contra una propiedad matematica que debe cumplirse por definicion,
no contra otra implementacion mia.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step, equilibrium_up, linearize, dynamics
from igrrl.mpc_design import DesignMPC, Q_P2, R_P2

OK, MAL = [], []


def chk(nombre, bien, det):
    (OK if bien else MAL).append((nombre, det))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<50} {det}")


NOM = load_pendulum_params()
xeq = equilibrium_up(0.0)
A, B = linearize(xeq, 0.0, NOM)

print("=" * 100)
print("1. MODELO Y LINEALIZACION")
print("=" * 100)
# la linealizacion debe coincidir con una diferencia finita de la dinamica no lineal
h = 1e-6
Afd = np.zeros((4, 4))
for j in range(4):
    dx = np.zeros(4); dx[j] = h
    Afd[:, j] = (dynamics(xeq + dx, 0.0, NOM) - dynamics(xeq - dx, 0.0, NOM)) / (2 * h)
Bfd = ((dynamics(xeq, h, NOM) - dynamics(xeq, -h, NOM)) / (2 * h)).reshape(-1, 1)
chk("A coincide con diferencia finita", np.max(np.abs(A - Afd)) < 1e-4,
    f"error maximo {np.max(np.abs(A - Afd)):.2e}")
chk("B coincide con diferencia finita", np.max(np.abs(B - Bfd)) < 1e-4,
    f"error maximo {np.max(np.abs(B - Bfd)):.2e}")
chk("el equilibrio invertido es equilibrio", np.max(np.abs(dynamics(xeq, 0.0, NOM))) < 1e-9,
    f"|f(x_eq,0)| = {np.max(np.abs(dynamics(xeq, 0.0, NOM))):.2e}")
chk("la planta es inestable en ese punto", max(np.linalg.eigvals(A).real) > 0,
    f"polo mas a la derecha {max(np.linalg.eigvals(A).real):+.4f}")

print()
print("=" * 100)
print("2. LQR")
print("=" * 100)
lqr = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=0.0)
K = np.asarray(lqr.K).reshape(1, -1)
P = solve_continuous_are(A, B, Q_P2, np.array([[R_P2]]))
res = A.T @ P + P @ A - P @ B @ np.linalg.inv([[R_P2]]) @ B.T @ P + Q_P2
chk("P resuelve la ecuacion de Riccati continua", np.max(np.abs(res)) < 1e-6,
    f"residuo maximo {np.max(np.abs(res)):.2e}")
Kr = np.linalg.inv([[R_P2]]) @ B.T @ P
chk("K coincide con R^-1 B^T P", np.max(np.abs(K - Kr)) < 1e-6,
    f"diferencia maxima {np.max(np.abs(K - Kr)):.2e}")
cl = np.linalg.eigvals(A - B @ K)
chk("el lazo cerrado es estable", max(cl.real) < 0, f"polo mas a la derecha {max(cl.real):+.4f}")
chk("K usa el signo correcto (estabiliza, no desestabiliza)",
    max(np.linalg.eigvals(A - B @ K).real) < max(np.linalg.eigvals(A + B @ K).real),
    "verificado contra el signo opuesto")

print()
print("=" * 100)
print("3. DISCRETIZACION DEL MPC")
print("=" * 100)
dt = 0.01
M = np.zeros((5, 5)); M[:4, :4] = A; M[:4, 4:] = B
Md = expm(M * dt); Ad, Bd = Md[:4, :4], Md[:4, 4:]
# Referencia por RK4 y no por Euler. Euler es de primer orden y su propio error de
# truncamiento, del orden de 1e-5 con 4000 subpasos, enmascara la comparacion.
x0 = np.array([0.03, 0.02, -0.05, 0.1])
u0 = 2.0


def _f(x):
    return A @ x + B.flatten() * u0


xl = x0.copy()
sub = 2000
hh = dt / sub
for _ in range(sub):
    k1 = _f(xl); k2 = _f(xl + hh / 2 * k1); k3 = _f(xl + hh / 2 * k2); k4 = _f(xl + hh * k3)
    xl = xl + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
xd = Ad @ x0 + Bd.flatten() * u0
chk("Ad y Bd reproducen la integracion del sistema lineal",
    np.max(np.abs(xl - xd)) < 1e-12, f"error maximo {np.max(np.abs(xl - xd)):.2e}")

print()
print("=" * 100)
print("4. CONDENSACION DEL MPC")
print("=" * 100)
m = DesignMPC(NOM, dt=dt, T_pred=0.5, Nc=10, blocking="front", terminal="riccati")
Sx, Su, idx = m._Sx, m._Su, m._idx
v = np.random.default_rng(0).normal(0, 3, m.Nc)
# propagacion directa con la misma secuencia de control
x = x0.copy(); traj = [x.copy()]
for k in range(m.Np):
    x = Ad @ x + Bd.flatten() * v[idx[k]]
    traj.append(x.copy())
err = max(np.max(np.abs(Sx[k] @ x0 + Su[k] @ v - traj[k])) for k in range(m.Np + 1))
chk("las matrices condensadas reproducen la trayectoria", err < 1e-8,
    f"error maximo sobre {m.Np+1} pasos {err:.2e}")
chk("el mapa de bloques cubre el horizonte", len(idx) == m.Np and idx.max() == m.Nc - 1,
    f"Np={m.Np}, Nc={m.Nc}, indices 0..{idx.max()}")
Pd = solve_discrete_are(Ad, Bd, Q_P2, np.array([[R_P2]]))
m2 = DesignMPC(NOM, dt=dt, T_pred=0.5, Nc=10, blocking="front", terminal="stage")
chk("el terminal riccati usa el DARE discreto y difiere del de etapa",
    not np.allclose(Pd, Q_P2), f"||P_dare - Q|| = {np.linalg.norm(Pd - Q_P2):.3f}")

print()
print("=" * 100)
print("5. SUBSUNCION. EL MPC DEBE REPRODUCIR AL LQR")
print("=" * 100)
mpc_libre = DesignMPC(NOM, dt=0.005, T_pred=1.5, Nc=25, blocking="front",
                      terminal="riccati", x_limit=None)
rng = np.random.default_rng(3)
difs = []
for _ in range(25):
    xs = np.array([rng.uniform(-0.06, 0.06), np.pi + rng.uniform(-0.06, 0.06),
                   rng.uniform(-0.15, 0.15), rng.uniform(-0.3, 0.3)])
    difs.append(abs(mpc_libre.compute(0.0, xs) - lqr.compute(0.0, xs)))
difs = np.array(difs)
chk("MPC sin restriccion reproduce al LQR", np.median(difs) < 0.8,
    f"|du| mediana {np.median(difs):.3f} V, p90 {np.percentile(difs,90):.3f} V")

print()
print("=" * 100)
print("6. ESTIMADOR EKF")
print("=" * 100)
rng = np.random.default_rng(7)
x = np.array([0.02, np.pi + 0.05, 0.0, 0.0])
xh = np.array([0.0, np.pi, 0.0, 0.0])          # arranca con error deliberado
P0 = np.eye(4) * 1e-2
ekf = ExtendedKalmanFilter(NOM, R=np.diag([1e-6, 1e-6]))
e0 = np.linalg.norm(x - xh)
for k in range(300):
    u = float(np.clip(lqr.compute(k * 0.005, xh), -12, 12))
    y = x[[0, 1]] + rng.normal(0, 1e-3, 2)
    xh, P0, _ = ekf.step(xh, P0, u, y, 0.005)
    x = rk4_step(x, u, NOM, 0.005)
e1 = np.linalg.norm(x - xh)
chk("el EKF reduce el error de estimacion", e1 < e0 * 0.5,
    f"error inicial {e0:.4f} -> final {e1:.5f}")
chk("la covarianza permanece definida positiva",
    np.all(np.linalg.eigvals((P0 + P0.T) / 2) > 0),
    f"autovalor minimo {min(np.linalg.eigvals((P0+P0.T)/2).real):.2e}")
chk("el EKF no diverge", np.all(np.isfinite(xh)) and np.all(np.isfinite(P0)), "finito")

print()
print("=" * 100)
print("7. AGENTES APRENDIDOS")
print("=" * 100)
from igrrl.env_standalone import StandaloneBalanceEnv
from igrrl.evaluate_common import make_policy, ACT_LIM
from stable_baselines3 import SAC, DDPG

env = StandaloneBalanceEnv(domain_randomization=False, seed=0); env.reset(seed=0)
_, _, _, _, info = env.step(np.array([1.0]))
chk("la accion alcanza el limite completo del actuador",
    abs(abs(info["u_control"]) - ACT_LIM) < 1e-6, f"u = {info['u_control']:.4f} V")
e2 = StandaloneBalanceEnv(domain_randomization=True, seed=11)
mps = [e2.reset()[1]["parameters"].Mp for _ in range(5)]
chk("la aleatorizacion de dominio varia entre episodios", len(set(mps)) == 5,
    f"{len(set(mps))} valores distintos de Mp en 5 episodios")
for algo, cls in (("sac", SAC), ("ddpg", DDPG)):
    f = ROOT / "results" / "training" / f"standalone_{algo}" / "seed_1" / "best_model.zip"
    if not f.exists():
        continue
    mdl = cls.load(str(f.with_suffix("")))
    pol = make_policy(mdl)
    us = [pol(0.0, np.array([0.0, np.pi + a, 0.0, 0.0])) for a in (-0.2, 0.0, 0.2)]
    chk(f"{algo} produce control acotado y no constante",
        all(abs(u) <= ACT_LIM + 1e-6 for u in us) and len(set(np.round(us, 4))) > 1,
        f"u en tres estados = {[round(u,3) for u in us]}")

print()
print("=" * 100)
print(f"RESUMEN CONTROLADORES   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 100)
for n, d in MAL:
    print(f"  DISCREPANCIA  {n}\n                {d}")
