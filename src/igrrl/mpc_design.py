"""MPC configurable para el estudio del ESPACIO DE DISENO (P2, D31).

Formulacion CONDENSADA (densa): las variables de decision son solo los Nc
movimientos libres, no los estados. Es la forma que se implementa realmente en
embebido (STM32/RPi con qpOASES/OSQP) y permite horizontes largos sin coste extra.

Variables de diseno expuestas (el objeto de estudio del paper):
  dt        : periodo de muestreo [s]
  T_pred    : horizonte de prediccion [s]  (Np = T_pred/dt)
  Nc        : numero de movimientos libres
  blocking  : 'uniform' | 'front'  (front = resolucion fina al inicio)
  terminal  : 'stage' (Q de etapa) | 'riccati' (P del DARE discreto)
  x_limit   : restriccion de carro (None = sin restriccion)
  slack_w   : peso de la holgura L1
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from src.config import PendulumParams
from src.models.pendulum import equilibrium_up, linearize

Q_P2 = np.diag([1.0, 12.0, 0.05, 0.02])
R_P2 = 0.002


def block_lengths(Np: int, Nc: int, scheme: str) -> list[int]:
    """Longitud de cada bloque de control; suman Np."""
    if Nc >= Np:
        return [1] * Np
    if scheme == "uniform":
        base, rem = divmod(Np, Nc)
        return [base + (1 if i < rem else 0) for i in range(Nc)]
    if scheme == "front":
        # crecimiento geometrico: fino al inicio, grueso al final
        w = np.array([1.35 ** i for i in range(Nc)], dtype=float)
        L = np.maximum(1, np.round(w / w.sum() * Np)).astype(int)
        while L.sum() > Np and L.max() > 1:
            L[int(np.argmax(L))] -= 1
        while L.sum() < Np:
            L[-1] += 1
        return L.tolist()
    raise ValueError(f"blocking desconocido: {scheme}")


class DesignMPC:
    def __init__(self, params: PendulumParams, dt: float, T_pred: float, Nc: int = 20,
                 blocking: str = "front", terminal: str = "riccati",
                 x_limit: float | None = None, slack_w: float = 1e4,
                 x_ref: float = 0.0, u_max: float = 12.0,
                 Q: np.ndarray = Q_P2, R: float = R_P2):
        self.dt = dt
        self.Np = max(1, int(round(T_pred / dt)))
        self.Nc = min(Nc, self.Np)
        self.x_eq = equilibrium_up(x_ref)
        self.u_max = u_max
        self.cfg = dict(dt=dt, T_pred=T_pred, Nc=self.Nc, Np=self.Np,
                        blocking=blocking, terminal=terminal,
                        x_limit=x_limit, slack_w=slack_w)

        A, B = linearize(self.x_eq, 0.0, params)
        n = A.shape[0]
        M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n:] = B
        Md = expm(M * dt)
        Ad, Bd = Md[:n, :n], Md[:n, n:]

        # Coste terminal: DARE DISCRETO (consistente con el MPC discreto)
        if terminal == "riccati":
            P = solve_discrete_are(Ad, Bd, Q, np.array([[R]]))
        elif terminal == "stage":
            P = Q
        else:
            raise ValueError(terminal)

        blk = block_lengths(self.Np, self.Nc, blocking)
        idx = np.concatenate([[i] * L for i, L in enumerate(blk)])[: self.Np]

        # --- condensacion: x = Sx x0 + Su v ---
        Sx = np.zeros((self.Np + 1, n, n)); Su = np.zeros((self.Np + 1, n, self.Nc))
        Sx[0] = np.eye(n)
        for k in range(self.Np):
            Sx[k + 1] = Ad @ Sx[k]
            Su[k + 1] = Ad @ Su[k]
            Su[k + 1][:, idx[k]] += Bd.flatten()
        self._Sx, self._Su, self._idx = Sx, Su, idx

        # --- coste cuadratico en v ---
        H = np.zeros((self.Nc, self.Nc)); F = np.zeros((n, self.Nc)); 
        for k in range(self.Np):
            Qk = Q
            H += Su[k].T @ Qk @ Su[k]
            F += Sx[k].T @ Qk @ Su[k]
            e = np.zeros(self.Nc); e[idx[k]] = 1.0
            H += R * np.outer(e, e)
        H += Su[self.Np].T @ P @ Su[self.Np]
        F += Sx[self.Np].T @ P @ Su[self.Np]
        H = 0.5 * (H + H.T) + 1e-9 * np.eye(self.Nc)
        self._H, self._F = H, F

        self._x0 = cp.Parameter(n)
        v = cp.Variable(self.Nc)
        cost = cp.quad_form(v, cp.psd_wrap(H)) + 2 * (self._x0 @ F) @ v
        cons = [v >= -u_max, v <= u_max]
        if x_limit is not None:
            xr = float(self.x_eq[0])
            Cx = np.stack([Su[k][0] for k in range(self.Np + 1)])       # (Np+1, Nc)
            Dx = np.stack([Sx[k][0] for k in range(self.Np + 1)])       # (Np+1, n)
            s = cp.Variable(self.Np + 1, nonneg=True)
            pos = Dx @ self._x0 + Cx @ v + xr
            cons += [pos <= x_limit + s, pos >= -x_limit - s]
            cost += slack_w * cp.sum(s)
        self._v = v
        self._prob = cp.Problem(cp.Minimize(cost), cons)
        self.solver_failures = 0

    def compute(self, t: float, x: np.ndarray) -> float:
        e = np.asarray(x, dtype=float) - self.x_eq
        if not np.all(np.isfinite(e)):
            self.solver_failures += 1
            return 0.0
        e[1] = np.arctan2(np.sin(e[1]), np.cos(e[1]))
        self._x0.value = e
        try:
            self._prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        except Exception:
            self.solver_failures += 1
            return 0.0
        if self._v.value is None:
            self.solver_failures += 1
            return 0.0
        return float(np.clip(self._v.value[0], -self.u_max, self.u_max))
