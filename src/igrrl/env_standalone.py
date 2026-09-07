"""Entorno de control STANDALONE (no residual) para el estudio DRL vs clasico.

Diferencias con ResidualBalanceEnv (que NO se modifica; 48 modelos dependen de el):
  1. La accion mapea LINEALMENTE al control COMPLETO:
     u = actuator_limit * clip(a, -1, 1)   (ver env.step, linea del mapeo).
     Antes usaba actuator_limit * tanh(a), que topaba en 12*tanh(1) = 9.14 V y
     dejaba al agente con el 76% de la autoridad del actuador (bug corregido).
     No hay LQR base ni compuerta de autoridad.
  2. La recompensa NO lleva el termino de penalizacion del residuo. Ese termino
     favorecia estructuralmente al residuo cero (artefacto D23) y aqui no aplica.
  3. Observacion = la MISMA informacion que recibe el LQR de realimentacion de
     salida: [x_hat, sin(e_theta), cos(e_theta), xdot_hat, thetadot_hat].
     Sin NIS ni rho. Comparacion justa: mismo conjunto de informacion.

Se conservan identicos: planta no lineal RK4, EKF, ruido de medicion,
aleatorizacion de dominio (Mp, Jp, l, kt), perturbaciones no medidas,
condicion inicial, criterio de terminacion y el resto del coste de etapa.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import PendulumParams, load_pendulum_params
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step


class StandaloneBalanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        params: PendulumParams | None = None,
        domain_randomization: bool = True,
        perturbation_probability: float = 0.5,
        dt: float = 0.02,
        t_final: float = 8.0,
        actuator_limit: float = 12.0,
        measurement_noise_std: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.nominal_params = params if params is not None else load_pendulum_params()
        self.domain_randomization = domain_randomization
        self.perturbation_probability = perturbation_probability
        self.dt = dt
        self.max_steps = int(round(t_final / dt))
        self.actuator_limit = actuator_limit
        self.measurement_noise_std = measurement_noise_std
        self._seed = seed
        self._seeded = False

        high = np.array([1.5, 1.0, 1.0, 20.0, 30.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.state = np.zeros(4)
        self.xhat = np.zeros(4)
        self.P = np.eye(4)
        self.params = self.nominal_params
        self.ekf: ExtendedKalmanFilter | None = None
        self.steps = 0
        self.disturbance: dict[str, float | int] = {"kind": "none"}

    # --- identicos al entorno residual (misma distribucion de escenarios) ---
    def _sample_params(self) -> PendulumParams:
        if not self.domain_randomization:
            return self.nominal_params
        from dataclasses import replace
        factors = {
            "Mp": self.np_random.uniform(0.80, 1.20),
            "Jp": self.np_random.uniform(0.75, 1.25),
            "l": self.np_random.uniform(0.85, 1.15),
            "kt": self.np_random.uniform(0.90, 1.10),
        }
        return replace(self.nominal_params,
                       **{k: getattr(self.nominal_params, k) * v for k, v in factors.items()})

    def _sample_disturbance(self) -> dict[str, float | int]:
        if self.np_random.uniform() > self.perturbation_probability:
            return {"kind": "none"}
        start = int(self.np_random.integers(int(2.0 / self.dt), self.max_steps - 10))
        if self.np_random.uniform() < 0.5:
            return {"kind": "impulse", "start": start, "steps": 1,
                    "amplitude": float(self.np_random.uniform(-40.0, 40.0))}
        return {"kind": "sustained", "start": start,
                "steps": int(self.np_random.integers(10, 60)),
                "amplitude": float(self.np_random.uniform(-6.0, 6.0))}

    def _disturbance_at(self, step: int) -> float:
        d = self.disturbance
        if d["kind"] == "none":
            return 0.0
        s = int(d["start"])
        if s <= step < s + int(d["steps"]):
            return float(d["amplitude"])
        return 0.0

    def _observation(self) -> np.ndarray:
        te = np.arctan2(np.sin(self.xhat[1] - np.pi), np.cos(self.xhat[1] - np.pi))
        return np.array([self.xhat[0], np.sin(te), np.cos(te), self.xhat[2], self.xhat[3]],
                        dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Gymnasium: reset(seed=None) NO debe re-sembrar; continua el flujo del
        # RNG. Re-sembrar con self._seed en cada episodio congelaba la
        # aleatorizacion de dominio (misma planta y misma perturbacion siempre).
        if seed is None and self._seed is not None and not self._seeded:
            seed = self._seed
        self._seeded = True
        super().reset(seed=seed)
        self.params = self._sample_params()
        R = np.diag([self.measurement_noise_std**2, self.measurement_noise_std**2])
        self.ekf = ExtendedKalmanFilter(self.nominal_params, R=R)
        theta0 = np.pi + self.np_random.uniform(-0.30, 0.30)
        self.state = np.array([0.0, theta0, 0.0, 0.0])
        self.xhat = self.state.copy()
        self.P = np.eye(4) * 1e-3
        self.steps = 0
        self.disturbance = self._sample_disturbance()
        return self._observation(), {"parameters": self.params, "disturbance": self.disturbance}

    def step(self, action: np.ndarray):
        if self.ekf is None:
            raise RuntimeError("reset debe llamarse antes de step")
        # Mapeo LINEAL al rango COMPLETO del actuador. Antes: actuator_limit*tanh(clip(a))
        # producia doble saturacion y topaba en 12*tanh(1)=9.14 V, dejando al DRL con
        # solo el 76 % de la autoridad que si tienen LQR y MPC (comparacion injusta).
        raw = float(np.clip(action[0], -1.0, 1.0))
        u_control = float(self.actuator_limit * raw)
        disturbance = self._disturbance_at(self.steps)

        y = self.state[[0, 1]] + self.np_random.normal(0.0, self.measurement_noise_std, size=2)
        self.xhat, self.P, _ = self.ekf.step(self.xhat, self.P, u_control, y, self.dt)
        self.state = rk4_step(self.state, u_control + disturbance, self.params, self.dt)
        self.steps += 1

        te = np.arctan2(np.sin(self.state[1] - np.pi), np.cos(self.state[1] - np.pi))
        reward = -(12.0 * te**2 + 1.0 * self.state[0] ** 2 + 0.05 * self.state[2] ** 2
                   + 0.02 * self.state[3] ** 2 + 0.002 * u_control**2)
        terminated = bool(abs(self.state[0]) > 1.5 or abs(te) > 1.4
                          or not np.all(np.isfinite(self.state)))
        if terminated:
            reward -= 100.0
        truncated = self.steps >= self.max_steps
        info = {"u_control": u_control, "disturbance": disturbance, "parameters": self.params}
        return self._observation(), float(reward), terminated, truncated, info
