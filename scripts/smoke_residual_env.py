"""Comprobación corta de la integración de ResidualBalanceEnv."""
from __future__ import annotations

from igrrl.env import ResidualBalanceEnv


def main() -> None:
    env = ResidualBalanceEnv(seed=4101)
    observation, info = env.reset()
    total_reward = 0.0
    for _ in range(50):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    print(
        "smoke OK",
        f"obs={observation.shape}",
        f"reward={total_reward:.3f}",
        f"rho={info['rho']:.3f}",
        f"nis={info['nis']:.3f}",
    )


if __name__ == "__main__":
    main()
