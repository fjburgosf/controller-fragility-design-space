import numpy as np
import pytest

from igrrl.controller import GateConfig, GateMode, InnovationGate, InnovationGatedResidualController


def _base(_t: float, _x: np.ndarray) -> float:
    return 1.0


def _residual(_observation: np.ndarray) -> float:
    return 1.0


def _observation(xhat: np.ndarray, eta: float, rho: float) -> np.ndarray:
    return np.r_[xhat, eta, rho]


def test_gate_is_near_minimum_for_nominal_nis() -> None:
    gate = InnovationGate(GateConfig(lam=0.1, eta_bar=10.0, tau=2.0, rho_min=0.3, rho_max=1.0))
    for _ in range(40):
        rho = gate.update(2.0)
    assert gate.eta == pytest.approx(2.0)
    assert rho < 0.33


def test_gate_increases_authority_after_large_innovation() -> None:
    gate = InnovationGate(GateConfig(lam=0.1, eta_bar=10.0, tau=2.0, rho_min=0.3, rho_max=1.0))
    low = gate.update(2.0)
    high = gate.update(200.0)
    assert high > low
    assert high > 0.95


def test_fixed_authority_ablations_ignore_nis() -> None:
    gate = InnovationGate(GateConfig(rho_min=0.25, rho_max=0.9))
    controller = InnovationGatedResidualController(_base, _residual, _observation, gate=gate, mode=GateMode.MINIMUM)
    controller.compute(0.0, np.zeros(4), nis=1000.0)
    assert controller.rho_history[-1] == pytest.approx(0.25)

    controller = InnovationGatedResidualController(_base, _residual, _observation, gate=gate, mode=GateMode.MAXIMUM)
    controller.compute(0.0, np.zeros(4), nis=0.0)
    assert controller.rho_history[-1] == pytest.approx(0.9)


def test_residual_and_total_control_are_bounded() -> None:
    controller = InnovationGatedResidualController(
        _base,
        lambda _obs: 1e6,
        _observation,
        residual_limit=4.0,
        u_limit=2.0,
        mode=GateMode.MAXIMUM,
    )
    u = controller.compute(0.0, np.zeros(4), nis=100.0)
    assert controller.residual_history[-1] == pytest.approx(4.0)
    assert u == pytest.approx(2.0)
