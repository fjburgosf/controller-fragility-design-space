"""Entorno de estabilización residual para IG RRL.

La planta, el LQR y el EKF provienen de la infraestructura reproducida de P1.
Este entorno añade los elementos exclusivos de P2: aleatorización de dominio,
perturbaciones no modeladas y la autoridad residual basada en NIS.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from igrrl.controller import GateConfig, GateMode, InnovationGate
from src.config import PendulumParams, load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import rk4_step


class ResidualBalanceEnv(gym.Env):
    """Estabilización cerca de arriba para entrenar solamente el residuo.

    El agente entrega una acción normalizada. El entorno combina esa acción con
    LQR y el gate seleccionado. El estado real de la planta y los parámetros
    muestreados no se exponen a la política.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: PendulumParams | None = None,
        gate_config: GateConfig | None = None,
        mode: GateMode | str = GateMode.INNOVATION,
        domain_randomization: bool = True,
        perturbation_probability: float = 0.5,
        dt: float = 0.02,
        t_final: float = 8.0,
        residual_limit: float = 4.0,
        actuator_limit: float = 12.0,
        measurement_noise_std: float = 1e-3,
        gate_warmup_s: float = 1.5,
        base_R: float = 0.01,
        residual_penalty: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= perturbation_probability <= 1.0:
            raise ValueError("perturbation_probability debe estar entre 0 y 1")
        if base_R <= 0.0:
            raise ValueError("base_R debe ser positivo")
        if residual_penalty < 0.0:
            raise ValueError("residual_penalty no puede ser negativo")
        self.nominal_params = params if params is not None else load_pendulum_params()
        self.gate_config = gate_config if gate_config is not None else GateConfig()
        self.mode = GateMode(mode)
        self.domain_randomization = domain_randomization
        self.perturbation_probability = perturbation_probability
        self.dt = dt
        self.max_steps = int(round(t_final / dt))
        self.residual_limit = residual_limit
        self.actuator_limit = actuator_limit
        self.measurement_noise_std = measurement_noise_std
        self.gate_warmup_steps = int(round(gate_warmup_s / dt))
        # Calidad del controlador base como variable independiente. base_R=0.01 es
        # el LQR canónico de P1. Valores mayores penalizan más el control y
        # producen una base progresivamente más conservadora, es decir con mayor
        # margen disponible para el residuo.
        self.base_R = float(base_R)
        # Precio del residuo en la recompensa. Regulariza la accion aprendida pero
        # compite directamente contra la mejora de estado que el residuo puede
        # obtener. Es la segunda variable independiente del estudio.
        self.residual_penalty = float(residual_penalty)
        self._seed = seed
        self._seeded = False

        # [e_x, sin(e_theta), cos(e_theta), xdot_hat, theta_dot_hat, nis_norm, rho]
        high = np.array([1.5, 1.0, 1.0, 20.0, 30.0, 50.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.state = np.zeros(4)
        self.xhat = np.zeros(4)
        self.P = np.eye(4)
        self.params = self.nominal_params
        self.lqr: LQRController | None = None
        self.ekf: ExtendedKalmanFilter | None = None
        self.gate = InnovationGate(self.gate_config)
        self.steps = 0
        self.disturbance: dict[str, float | int] = {"kind": "none"}

    def _sample_params(self) -> PendulumParams:
        if not self.domain_randomization:
            return self.nominal_params
        # Rangos priorizados por la sensibilidad de P1. La política nunca los observa.
        factors = {
            "Mp": self.np_random.uniform(0.80, 1.20),
            "Jp": self.np_random.uniform(0.75, 1.25),
            "l": self.np_random.uniform(0.85, 1.15),
            "kt": self.np_random.uniform(0.90, 1.10),
        }
        return replace(self.nominal_params, **{key: getattr(self.nominal_params, key) * value for key, value in factors.items()})

    def _sample_disturbance(self) -> dict[str, float | int]:
        if self.np_random.uniform() >= self.perturbation_probability:
            return {"kind": "none"}
        kind = "impulse" if self.np_random.uniform() < 0.5 else "sustained"
        # La sonda NIS de P1 se interpreta tras 1.5 s de asentamiento del EKF.
        # No se usan perturbaciones dentro de esa ventana.
        start = int(self.np_random.integers(int(2.0 / self.dt), int(3.5 / self.dt)))
        if kind == "impulse":
            return {"kind": kind, "start": start, "duration": int(0.10 / self.dt), "amplitude": float(self.np_random.uniform(4.0, 10.0))}
        return {"kind": kind, "start": start, "duration": int(0.75 / self.dt), "amplitude": float(self.np_random.uniform(1.0, 4.0))}

    def _disturbance_at(self, step: int) -> float:
        if self.disturbance["kind"] == "none":
            return 0.0
        start = int(self.disturbance["start"])
        duration = int(self.disturbance["duration"])
        if start <= step < start + duration:
            return float(self.disturbance["amplitude"])
        return 0.0

    def _authority(self) -> float:
        # Durante el transitorio inicial el NIS mide inicialización del EKF y
        # dinámica de captura, no una perturbación externa. Por diseño, el gate
        # permanece cerrado y conserva su línea base chi cuadrado.
        if self.steps < self.gate_warmup_steps:
            return self.gate_config.rho_min
        dynamic = self.gate.update(float(self.ekf.last_nis))
        if self.mode is GateMode.MINIMUM:
            return self.gate_config.rho_min
        if self.mode is GateMode.MAXIMUM:
            return self.gate_config.rho_max
        return dynamic

    def _observation(self, rho: float | None = None) -> np.ndarray:
        theta_error = np.arctan2(np.sin(self.xhat[1] - np.pi), np.cos(self.xhat[1] - np.pi))
        rho = self.gate.authority() if rho is None else rho
        nis_normalized = np.clip((self.gate.eta - self.gate_config.nis_dimension) / self.gate_config.nis_dimension, 0.0, 50.0)
        return np.array(
            [self.xhat[0], np.sin(theta_error), np.cos(theta_error), self.xhat[2], self.xhat[3], nis_normalized, rho],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Gymnasium: reset(seed=None) NO debe re-sembrar; continua el flujo del
        # RNG. Re-sembrar con self._seed en cada episodio congelaba la
        # aleatorizacion de dominio (misma planta y misma perturbacion siempre).
        if seed is None and self._seed is not None and not self._seeded:
            seed = self._seed
        self._seeded = True
        super().reset(seed=seed)
        self.params = self._sample_params()
        self.lqr = LQRController(self.nominal_params, R=self.base_R)
        # R coincide con la varianza que se inyecta. Esto calibra NIS cerca de chi2(2).
        R = np.diag([self.measurement_noise_std**2, self.measurement_noise_std**2])
        self.ekf = ExtendedKalmanFilter(self.nominal_params, R=R)
        self.gate.reset()
        theta0 = np.pi + self.np_random.uniform(-0.30, 0.30)
        # Condición coherente con las sondas de NIS de P1. La variación angular
        # mantiene una tarea de regulación no trivial sin introducir un error de
        # inicialización artificial que active la compuerta.
        self.state = np.array([0.0, theta0, 0.0, 0.0])
        self.xhat = self.state.copy()
        self.P = np.eye(4) * 1e-3
        self.steps = 0
        self.disturbance = self._sample_disturbance()
        return self._observation(self.gate.authority()), {"parameters": self.params, "disturbance": self.disturbance}

    def step(self, action: np.ndarray):
        if self.lqr is None or self.ekf is None:
            raise RuntimeError("reset debe llamarse antes de step")
        rho = self._authority()
        raw = float(np.clip(action[0], -1.0, 1.0))
        residual = self.residual_limit * np.tanh(raw)
        u_base = self.lqr.compute(self.steps * self.dt, self.xhat.copy())
        u_control = float(np.clip(u_base + rho * residual, -self.actuator_limit, self.actuator_limit))
        disturbance = self._disturbance_at(self.steps)

        y = self.state[[0, 1]] + self.np_random.normal(0.0, self.measurement_noise_std, size=2)
        self.xhat, self.P, _ = self.ekf.step(self.xhat, self.P, u_control, y, self.dt)
        self.state = rk4_step(self.state, u_control + disturbance, self.params, self.dt)
        self.steps += 1

        theta_error = np.arctan2(np.sin(self.state[1] - np.pi), np.cos(self.state[1] - np.pi))
        reward = -(
            12.0 * theta_error**2 + 1.0 * self.state[0] ** 2 + 0.05 * self.state[2] ** 2
            + 0.02 * self.state[3] ** 2 + self.residual_penalty * residual**2 + 0.002 * u_control**2
        )
        terminated = bool(abs(self.state[0]) > 1.5 or abs(theta_error) > 1.4 or not np.all(np.isfinite(self.state)))
        if terminated:
            reward -= 100.0
        truncated = self.steps >= self.max_steps
        info = {
            "u_base": u_base,
            "u_control": u_control,
            "residual": residual,
            "rho": rho,
            "nis": self.ekf.last_nis,
            "eta": self.gate.eta,
            "disturbance": disturbance,
            "parameters": self.params,
        }
        return self._observation(rho), float(reward), terminated, truncated, info
