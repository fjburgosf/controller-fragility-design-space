"""Mide el CONDICIONAMIENTO del problema condensado y el estado del solver.

Motivo. Los numeros de condicionamiento que cita la Seccion 3.4 y que dibuja la
Figura 1 estaban escritos a mano dentro de los scripts de figura, sin archivo que
los respaldase, y el manuscrito afirmaba ademas que el solver devuelve soluciones
inexactas sin que eso quedase registrado en ningun sitio. Este script mide las dos
cosas y las guarda, de modo que la figura y el texto lean un dato y no una
constante recordada.

Que se mide, para cada horizonte de prediccion:

  cond(H)   numero de condicion de la Hessiana del problema condensado, formada
            explicitamente a partir de las matrices de prediccion del propio
            controlador, no de una reconstruccion aparte.
  digitos   digitos significativos que sobreviven en doble precision, estimados
            como 16 menos el logaritmo decimal del condicionamiento.
  estado    reparto de estados que devuelve el solver a lo largo de una
            trayectoria en lazo cerrado, y numero de llamadas sin solucion.

Salida: results/processed/mpc_conditioning.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from src.config import load_pendulum_params
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step
from igrrl.mpc_design import DesignMPC, Q_P2, R_P2

NOM = load_pendulum_params()
DT = 0.01
HORIZONTES = [0.4, 0.8, 1.5, 2.2, 3.0]
T_FINAL, DV, MS, ACT = 6.0, 8.0, 1e-3, 12.0
DR = {"Mp": (0.8, 1.2), "Jp": (0.75, 1.25), "l": (0.85, 1.15), "kt": (0.9, 1.1)}
SEED = 9000


def hessiana(m: DesignMPC) -> np.ndarray:
    """La H que el propio controlador construye y pasa al solver.

    Se lee del controlador en vez de reconstruirla aqui. Una reconstruccion
    paralela mediria mi copia, no el problema que se resuelve de verdad, y una
    discrepancia entre ambas pasaria inadvertida.
    """
    return m._H


def trayectoria(m: DesignMPC) -> Counter:
    """Estados que devuelve el solver a lo largo de un lazo cerrado real."""
    rng = np.random.default_rng(SEED)
    p = replace(NOM, **{k: getattr(NOM, k) * rng.uniform(*v) for k, v in DR.items()})
    x = np.array([0.0, np.pi + rng.uniform(-0.02, 0.02), 0.0, 0.0])
    xh = x.copy(); P = np.eye(4) * 1e-3
    ekf = ExtendedKalmanFilter(NOM, R=np.diag([MS**2] * 2))
    nstep = int(round(T_FINAL / DT))
    k_on, k_off = int(round(1.0 / DT)), int(round(1.4 / DT))
    estados = Counter()
    for k in range(nstep):
        if not np.all(np.isfinite(x)):
            estados["planta divergida"] += 1
            break
        u = m.compute(k * DT, xh)
        estados[str(m._prob.status)] += 1
        u = float(np.clip(u, -ACT, ACT))
        d = DV if k_on <= k < k_off else 0.0
        y = x[[0, 1]] + rng.normal(0, MS, 2)
        xh, P, _ = ekf.step(xh, P, u, y, DT)
        x = rk4_step(x, u + d, p, DT)
    return estados


filas = []
for T in HORIZONTES:
    m = DesignMPC(NOM, dt=DT, T_pred=T, Nc=20, blocking="front", terminal="riccati")
    H = hessiana(m)
    cond = float(np.linalg.cond(H))
    est = trayectoria(m)
    total = sum(est.values())
    optimo = est.get("optimal", 0)
    filas.append({
        "T_pred": T,
        "Np": m.Np,
        "Nc": m.Nc,
        "cond_H": cond,
        "digitos_significativos": max(0.0, 16.0 - np.log10(cond)),
        "llamadas": total,
        "estado_optimo": optimo,
        "frac_optimo": optimo / total if total else np.nan,
        "fallos_solver": m.solver_failures,
        "estados": "; ".join(f"{k}={v}" for k, v in sorted(est.items())),
    })
    print(f"  T_pred={T:<4} Np={m.Np:<4} cond(H)={cond:.3e}  "
          f"digitos={filas[-1]['digitos_significativos']:.1f}  "
          f"optimo={optimo}/{total}  fallos={m.solver_failures}")

df = pd.DataFrame(filas)
sal = ROOT / "results" / "processed" / "mpc_conditioning.csv"
df.to_csv(sal, index=False)
print(f"\nGuardado: {sal}")
