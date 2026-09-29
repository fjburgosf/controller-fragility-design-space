"""Carga de la configuración centralizada del proyecto.

Fuente de verdad para la planta: configs/p1_canonical_parameters.yaml.
La configuración propia de IG RRL vive por separado en
configs/experiment.yaml. No duplicar parámetros físicos en otros módulos:
siempre construir PendulumParams a partir de este loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "p1_canonical_parameters.yaml"


@dataclass(frozen=True)
class PendulumParams:
    """Parámetros físicos del péndulo invertido (docs/CANONICAL_MODEL.md)."""

    r: float
    Jm: float
    Rm: float
    kt: float
    M_c: float
    c: float
    Mp: float
    Jp: float
    l: float
    b: float
    g: float

    @property
    def Mt(self) -> float:
        """Masa equivalente carrito + rotor: Mt = M_c + Jm / r^2."""
        return self.M_c + self.Jm / self.r**2

    @property
    def alpha(self) -> float:
        """Término auxiliar: alpha = Jp*(Mt+Mp) + Mt*Mp*l^2."""
        return self.Jp * (self.Mt + self.Mp) + self.Mt * self.Mp * self.l**2

    @property
    def Ra(self) -> float:
        """Alias de Rm, usado como nombre en PendulumNonlinear.m original."""
        return self.Rm


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pendulum_params(path: str | Path = DEFAULT_CONFIG_PATH) -> PendulumParams:
    cfg = load_config(path)
    p = cfg["physical_parameters"]
    return PendulumParams(
        r=p["r"],
        Jm=p["Jm"],
        Rm=p["Rm"],
        kt=p["kt"],
        M_c=p["M_c"],
        c=p["c"],
        Mp=p["Mp"],
        Jp=p["Jp"],
        l=p["l"],
        b=p["b"],
        g=p["g"],
    )
