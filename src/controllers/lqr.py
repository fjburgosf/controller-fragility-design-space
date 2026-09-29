"""B2 — LQR.

Regulador cuadrático lineal sobre el modelo linealizado en el equilibrio
invertido (theta=pi). Q y R reconstruidos textualmente de
`Clase 5/real system simulation/LQRrealpendulum.m` (líneas 35-37) y
confirmados en `LQR_LQE_KFonemeasurement.m` / `LQR_LQE_KFTwomeasurement.m`
(ver AUDIT_REPORT.md §3.1).
"""
from __future__ import annotations

import numpy as np
from control import lqr

from src.config import PendulumParams
from src.controllers.base import StateFeedbackController
from src.models.pendulum import equilibrium_up, linearize

# Pesos canónicos, textuales de LQRrealpendulum.m (líneas 35-36).
Q_DEFAULT = np.diag([1000.0, 2000.0, 0.0, 0.0])
R_DEFAULT = 0.01


class LQRController(StateFeedbackController):
    """LQR en torno al equilibrio invertido (theta=pi)."""

    def __init__(
        self,
        params: PendulumParams,
        Q: np.ndarray = Q_DEFAULT,
        R: float = R_DEFAULT,
        x_ref: float = 0.0,
    ):
        x_eq = equilibrium_up(x_ref)
        A, B = linearize(x_eq, 0.0, params)
        K, S, E = lqr(A, B, Q, R)
        super().__init__(K=np.asarray(K), x_eq=x_eq, u_eq=0.0)
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.S = S
        self.closed_loop_eigs = E
