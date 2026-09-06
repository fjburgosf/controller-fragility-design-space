"""Interfaz común de controlador para el benchmark.

Todo controlador del benchmark implementa `compute(t, x) -> u`, para que
`src.models.pendulum.simulate` pueda usarlo indistintamente como
`controller` en cualquier experimento (nominal, perturbaciones,
incertidumbre, Monte Carlo).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StateFeedbackController(ABC):
    """Controlador de realimentación de estado en torno a un equilibrio."""

    def __init__(self, K: np.ndarray, x_eq: np.ndarray, u_eq: float = 0.0):
        self.K = np.asarray(K).reshape(-1)
        self.x_eq = np.asarray(x_eq)
        self.u_eq = u_eq

    def compute(self, t: float, x: np.ndarray) -> float:
        error = x - self.x_eq
        # theta (índice 1) es un ángulo: envolver el error a [-pi, pi].
        # Sin esto, un estado con theta fuera de [0, 2*pi) (p.ej. tras un
        # swing-up que gira en sentido negativo, theta≈-2.98 en vez de
        # +3.30) produce un error crudo enorme (~-6.12 rad) en vez del
        # error físico real (~0.16 rad), generando esfuerzo de control
        # erróneo y aparentando una "cuenca de atracción asimétrica" que
        # en realidad era este bug (ver docs de diseño del swing-up).
        error[1] = np.arctan2(np.sin(error[1]), np.cos(error[1]))
        u = self.u_eq - self.K @ error
        return float(u)

    def __call__(self, t: float, x: np.ndarray) -> float:
        return self.compute(t, x)


class Controller(ABC):
    """Interfaz genérica para controladores no necesariamente lineales
    (MPC, DDPG, SAC, nuevo controlador)."""

    @abstractmethod
    def compute(self, t: float, x: np.ndarray) -> float:
        ...

    def __call__(self, t: float, x: np.ndarray) -> float:
        return self.compute(t, x)
