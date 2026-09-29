"""B4 (observador) — Filtro de Kalman Extendido (EKF).

Reconstrucción fiel de
`Clase 6/3.EKF/1.EKFinMatlab/EKF.m` (idéntico en las 4 ubicaciones donde
aparece, ver AUDIT_REPORT.md §3.6): predicción no lineal por RK4,
relinealización (Jacobiano) en cada paso, discretización por exponencial
de matriz, covarianza propagada y actualizada en forma Joseph
(numéricamente estable), innovación angular con `atan2` para evitar el
salto de ±pi.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.config import PendulumParams
from src.models.pendulum import dynamics, make_jacobian_fn

H_DEFAULT = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
Q_DEFAULT = np.diag([1e-9, 1e-9, 1e-8, 1e-8])
R_DEFAULT = np.diag([1e-8, 1e-6])


class ExtendedKalmanFilter:
    def __init__(
        self,
        params: PendulumParams,
        H: np.ndarray = H_DEFAULT,
        Q: np.ndarray = Q_DEFAULT,
        R: np.ndarray = R_DEFAULT,
    ):
        self.params = params
        self.H = H
        self.Q = Q
        self.R = R
        self._jacobian = make_jacobian_fn(params)
        # Estadísticos de innovación del último paso (para UAOR / B9, §3 de
        # docs/UAOR_DESIGN.md). No afectan a B4/B6: son solo lectura.
        self.last_innovation: np.ndarray | None = None
        self.last_S: np.ndarray | None = None
        self.last_nis: float = float(H.shape[0])  # línea base nominal E[NIS]=m

    def _rk4_predict(self, xhat: np.ndarray, u: float, dt: float) -> np.ndarray:
        p = self.params
        k1 = dynamics(xhat, u, p)
        k2 = dynamics(xhat + dt / 2 * k1, u, p)
        k3 = dynamics(xhat + dt / 2 * k2, u, p)
        k4 = dynamics(xhat + dt * k3, u, p)
        return xhat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(
        self, xhat: np.ndarray, P: np.ndarray, u: float, y: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Predicción (RK4 sobre la planta no lineal)
        x_pred = self._rk4_predict(xhat, u, dt)

        # Jacobiano evaluado en el estado predicho, discretizado por exponencial
        F, _ = self._jacobian(x_pred, u)
        Ad = expm(F * dt)
        P_pred = Ad @ P @ Ad.T + self.Q

        # Innovación (con wrap angular en theta)
        y_pred = self.H @ x_pred
        innovation = y - y_pred
        innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))

        # Ganancia de Kalman y corrección
        S = self.H @ P_pred @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = P_pred @ self.H.T @ S_inv
        xhat_new = x_pred + K @ innovation

        # NIS = innovación normalizada al cuadrado (medida de incertidumbre
        # online para B9; ~chi2(m) bajo modelo nominal correcto).
        self.last_innovation = innovation.copy()
        self.last_S = S
        self.last_nis = float(innovation @ S_inv @ innovation)

        I = np.eye(4)
        P_new = (I - K @ self.H) @ P_pred @ (I - K @ self.H).T + K @ self.R @ K.T

        return xhat_new, P_new, K
