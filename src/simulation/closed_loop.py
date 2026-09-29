"""Simulación en lazo cerrado con realimentación de salida (observador +
controlador), para los baselines B3 (LQR+KF) y B4 (LQR+EKF) y, en general,
cualquier combinación controlador/observador del benchmark.

A diferencia de `src.models.pendulum.simulate` (que asume realimentación
de estado completo, ideal para B1/B2), aquí el controlador solo ve el
estado ESTIMADO `xhat`, nunca el estado verdadero `x`, y las mediciones
`y` pueden incluir ruido gaussiano — la planta real siempre es la no
lineal completa (RK4), igual para todos los controladores (comparación
fair).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import PendulumParams
from src.models.pendulum import rk4_step

H_DEFAULT = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass
class ClosedLoopResult:
    t: np.ndarray
    X_true: np.ndarray
    X_hat: np.ndarray
    Y: np.ndarray
    U: np.ndarray


def simulate_output_feedback(
    x0: np.ndarray,
    xhat0: np.ndarray,
    params: PendulumParams,
    dt: float,
    t_final: float,
    controller,
    estimator,
    P0: np.ndarray | None = None,
    H: np.ndarray = H_DEFAULT,
    measurement_noise_std: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> ClosedLoopResult:
    """Simula el lazo cerrado planta no lineal + observador + controlador.

    `estimator` debe exponer `.step(xhat, u, y, dt) -> xhat_next` (KF de
    ganancia fija) o `.step(xhat, P, u, y, dt) -> (xhat_next, P_next, K)`
    (EKF) — se detecta automáticamente por introspección del resultado.
    `controller` es cualquier `StateFeedbackController`/`Controller`
    (recibe `xhat`, no el estado real).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n_steps = int(round(t_final / dt))
    t = np.arange(n_steps + 1) * dt

    X_true = np.zeros((n_steps + 1, 4))
    X_hat = np.zeros((n_steps + 1, 4))
    Y = np.zeros((n_steps, H.shape[0]))
    U = np.zeros(n_steps)

    x = np.array(x0, dtype=float)
    xhat = np.array(xhat0, dtype=float)
    P = P0.copy() if P0 is not None else np.eye(4)

    X_true[0] = x
    X_hat[0] = xhat

    is_ekf = P0 is not None

    for k in range(n_steps):
        u = float(np.clip(controller(t[k], xhat), -12.0, 12.0))
        U[k] = u

        y = H @ x
        if measurement_noise_std is not None:
            y = y + rng.normal(0.0, measurement_noise_std)
        Y[k] = y

        if is_ekf:
            xhat, P, _ = estimator.step(xhat, P, u, y, dt)
        else:
            xhat = estimator.step(xhat, u, y, dt)

        x = rk4_step(x, u, params, dt)

        X_true[k + 1] = x
        X_hat[k + 1] = xhat

    return ClosedLoopResult(t=t, X_true=X_true, X_hat=X_hat, Y=Y, U=U)
