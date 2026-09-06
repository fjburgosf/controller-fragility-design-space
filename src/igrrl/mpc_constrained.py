"""MPC con restriccion de via (blanda) para el estudio D de P2.

No modifica el MPC de P1. Reconstruye el mismo QH lineal sobre el equilibrio
invertido y anade una restriccion de posicion de carro |x| <= x_limit con
holgura penalizada, de modo que el problema nunca resulta infactible.

Su papel en P2-D es doble:
  1. baseline consciente de restricciones (caro en linea);
  2. oraculo del residuo: r*(x) = u_MPC(x) - u_LQR(x) en la sonda previa al
     entrenamiento, que decide si un residuo acotado puede capturar la
     conciencia de restriccion antes de gastar horas de entrenamiento.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.linalg import expm

from src.config import PendulumParams
from src.models.pendulum import equilibrium_up, linearize

# Coste de etapa alineado con la recompensa del entorno de P2
# (12*theta^2 + 1*x^2 + 0.05*xdot^2 + 0.02*thetadot^2 + 0.002*u^2),
# en el orden de estado [x, theta, xdot, thetadot].
Q_P2 = np.diag([1.0, 12.0, 0.05, 0.02])
R_P2 = 0.002


class ConstrainedMPC:
    def __init__(
        self,
        params: PendulumParams,
        dt: float = 0.02,
        Np: int = 40,
        Nc: int = 10,
        Q: np.ndarray = Q_P2,
        R: float = R_P2,
        x_limit: float | None = 0.30,
        slack_weight: float = 1e4,
        x_ref: float = 0.0,
        u_min: float = -12.0,
        u_max: float = 12.0,
    ):
        self.dt = dt
        self.Np = Np
        self.Nc = Nc
        self.x_limit = x_limit
        self.x_eq = equilibrium_up(x_ref)
        self.u_min = u_min
        self.u_max = u_max

        A, B = linearize(self.x_eq, 0.0, params)
        n = A.shape[0]
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n:] = B
        Md = expm(M * dt)
        self.Ad = Md[:n, :n]
        self.Bd = Md[:n, n:]

        self._x0 = cp.Parameter(n)
        self._u = cp.Variable(self.Nc)
        self._x = cp.Variable((self.Np + 1, n))
        cons = [self._x[0] == self._x0,
                self._u >= self.u_min, self._u <= self.u_max]
        cost = 0
        for k in range(self.Np):
            u_k = self._u[k] if k < self.Nc else self._u[self.Nc - 1]
            cons += [self._x[k + 1] == self.Ad @ self._x[k] + self.Bd.flatten() * u_k]
            cost += cp.quad_form(self._x[k], Q) + R * cp.square(u_k)
        cost += cp.quad_form(self._x[self.Np], Q)

        if x_limit is not None:
            # Restriccion ABSOLUTA |x_cart| <= x_limit. _x es el estado en
            # coordenadas de error (x_cart - x_ref), asi que se compensa el ref.
            # Holgura L1: practicamente dura pero el QP siempre factible.
            xr = float(self.x_eq[0])
            self._s = cp.Variable(self.Np + 1, nonneg=True)
            cons += [self._x[:, 0] + xr <= x_limit + self._s,
                     self._x[:, 0] + xr >= -x_limit - self._s]
            cost += slack_weight * cp.sum(self._s)

        self._problem = cp.Problem(cp.Minimize(cost), cons)
        self.last_status: str | None = None
        self.solver_failures = 0

    def compute(self, t: float, x: np.ndarray) -> float:
        error = np.asarray(x, dtype=float) - self.x_eq
        error[1] = np.arctan2(np.sin(error[1]), np.cos(error[1]))
        self._x0.value = error
        self._problem.solve(solver=cp.CLARABEL, warm_start=True,
                            tol_gap_abs=1e-5, tol_feas=1e-5, tol_gap_rel=1e-5,
                            max_iter=100, verbose=False)
        self.last_status = self._problem.status
        if self.last_status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or self._u.value is None:
            self.solver_failures += 1
            return 0.0
        return float(np.clip(self._u.value[0], self.u_min, self.u_max))
