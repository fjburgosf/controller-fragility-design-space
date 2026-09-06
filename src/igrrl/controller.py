"""Núcleo independiente de la arquitectura IG RRL v3.

No contiene el modelo del péndulo ni la implementación de SAC. Recibe tres
interfaces explícitas: un control base, una política residual normalizada y la
señal NIS del estimador. Esto permite evaluar exactamente la misma política con
las ablaciones de autoridad fija y con el gate por innovación.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

BasePolicy = Callable[[float, np.ndarray], float]
ResidualPolicy = Callable[[np.ndarray], float]
ObservationBuilder = Callable[[np.ndarray, float, float], np.ndarray]


class GateMode(str, Enum):
    """Modos de autoridad usados por P2 y sus ablaciones."""

    INNOVATION = "innovation"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


@dataclass(frozen=True)
class GateConfig:
    """Parámetros del gate NIS causal y suavizado."""

    lam: float = 0.10
    eta_bar: float = 10.0
    tau: float = 2.0
    rho_min: float = 0.30
    rho_max: float = 1.00
    nis_dimension: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.lam <= 1.0:
            raise ValueError("lam debe pertenecer a (0, 1]")
        if self.tau <= 0.0:
            raise ValueError("tau debe ser positivo")
        if not 0.0 <= self.rho_min <= self.rho_max <= 1.0:
            raise ValueError("se requiere 0 <= rho_min <= rho_max <= 1")
        if self.nis_dimension < 1:
            raise ValueError("nis_dimension debe ser positivo")


@dataclass
class InnovationGate:
    """EWMA causal del NIS y conversión sigmoidal a autoridad residual."""

    config: GateConfig = field(default_factory=GateConfig)
    eta: float = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.eta = float(self.config.nis_dimension)

    def update(self, nis: float) -> float:
        """Actualiza el indicador con el NIS disponible del paso anterior."""
        if not np.isfinite(nis) or nis < 0.0:
            raise ValueError("nis debe ser finito y no negativo")
        self.eta = (1.0 - self.config.lam) * self.eta + self.config.lam * float(nis)
        return self.authority()

    def authority(self) -> float:
        z = np.clip((self.eta - self.config.eta_bar) / self.config.tau, -60.0, 60.0)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return float(self.config.rho_min + (self.config.rho_max - self.config.rho_min) * sigmoid)


class InnovationGatedResidualController:
    """Suma una política residual limitada a una política base.

    La política residual retorna una acción escalar normalizada. La señal se
    limita con ``tanh`` antes de aplicar ``residual_limit``. El comando total
    se satura en ``u_limit``. El parámetro ``mode`` implementa las ablaciones
    A2 y A3 sin modificar la política entrenada.
    """

    def __init__(
        self,
        base_policy: BasePolicy,
        residual_policy: ResidualPolicy,
        observation_builder: ObservationBuilder,
        gate: InnovationGate | None = None,
        residual_limit: float = 4.0,
        u_limit: float = 12.0,
        mode: GateMode | str = GateMode.INNOVATION,
    ) -> None:
        if residual_limit <= 0.0 or u_limit <= 0.0:
            raise ValueError("los límites de control deben ser positivos")
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.observation_builder = observation_builder
        self.gate = gate if gate is not None else InnovationGate()
        self.residual_limit = float(residual_limit)
        self.u_limit = float(u_limit)
        self.mode = GateMode(mode)
        self.reset()

    def reset(self) -> None:
        self.gate.reset()
        self.rho_history: list[float] = []
        self.eta_history: list[float] = []
        self.residual_history: list[float] = []
        self.control_history: list[float] = []

    def _authority(self, nis: float) -> float:
        dynamic_rho = self.gate.update(nis)
        if self.mode is GateMode.MINIMUM:
            return self.gate.config.rho_min
        if self.mode is GateMode.MAXIMUM:
            return self.gate.config.rho_max
        return dynamic_rho

    def compute(self, t: float, xhat: np.ndarray, nis: float) -> float:
        """Calcula el comando total usando solo el estado estimado y el NIS."""
        rho = self._authority(nis)
        observation = np.asarray(self.observation_builder(xhat, self.gate.eta, rho), dtype=float)
        raw_action = float(self.residual_policy(observation))
        if not np.isfinite(raw_action):
            raise ValueError("la política residual produjo una acción no finita")
        residual = self.residual_limit * np.tanh(raw_action)
        u_base = float(self.base_policy(t, np.asarray(xhat, dtype=float)))
        u_total = float(np.clip(u_base + rho * residual, -self.u_limit, self.u_limit))

        self.rho_history.append(rho)
        self.eta_history.append(self.gate.eta)
        self.residual_history.append(residual)
        self.control_history.append(u_total)
        return u_total
