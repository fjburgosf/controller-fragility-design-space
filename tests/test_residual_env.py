import pytest

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv


def test_gate_remains_at_minimum_during_warmup() -> None:
    env = ResidualBalanceEnv(
        mode=GateMode.INNOVATION,
        domain_randomization=False,
        perturbation_probability=0.0,
        gate_warmup_s=0.10,
        dt=0.02,
        seed=4101,
    )
    try:
        env.reset()
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            assert info["rho"] == pytest.approx(env.gate_config.rho_min)
            assert not terminated
            assert not truncated
    finally:
        env.close()
