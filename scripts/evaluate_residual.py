"""Evaluación diagnóstica de una política residual entrenada.

Este script no sustituye la evaluación MC1 a MC3 final. Su objetivo es vigilar
aprendizaje por semilla durante el diseño y detectar políticas numéricamente
inválidas antes de gastar cómputo en validación formal.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa una política residual SAC")
    parser.add_argument("--model", type=Path, required=True, help="Ruta al modelo SAC sin extensión obligatoria")
    parser.add_argument("--variant", choices=["a1", "a2", "a3", "p2"], required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def variant_settings(variant: str) -> tuple[GateMode, bool]:
    settings = {
        "a1": (GateMode.MINIMUM, False),
        "a2": (GateMode.MINIMUM, True),
        "a3": (GateMode.MAXIMUM, True),
        "p2": (GateMode.INNOVATION, True),
    }
    return settings[variant]


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes debe ser positivo")
    mode, domain_randomization = variant_settings(args.variant)
    model = SAC.load(str(args.model), device="cpu")
    env = ResidualBalanceEnv(mode=mode, domain_randomization=domain_randomization)
    rows: list[dict[str, float | int | bool]] = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            rho_values: list[float] = []
            nis_values: list[float] = []
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                rho_values.append(float(info["rho"]))
                nis_values.append(float(info["nis"]))
            rows.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "return": episode_return,
                    "steps": env.steps,
                    "terminated": terminated,
                    "mean_rho": float(np.mean(rho_values)),
                    "max_rho": float(np.max(rho_values)),
                    "mean_nis": float(np.mean(nis_values)),
                    "max_nis": float(np.max(nis_values)),
                }
            )
    finally:
        env.close()

    output = args.output if args.output is not None else args.model.parent / "diagnostic_evaluation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    success_rate = 1.0 - float(np.mean([bool(row["terminated"]) for row in rows]))
    print(
        f"episodes={len(rows)} success={success_rate:.3f} "
        f"return_median={np.median([row['return'] for row in rows]):.3f} output={output}"
    )


if __name__ == "__main__":
    main()
