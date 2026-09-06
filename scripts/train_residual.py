"""Entrenamiento reproducible de las políticas residuales de P2.

Ejemplos desde la raíz P2:
  PYTHONPATH=.;src python scripts/train_residual.py --variant p2 --seed 4101
  PYTHONPATH=.;src python scripts/train_residual.py --variant a2 --seed 4101

No evalúa ni selecciona resultados de validación. Guarda solamente el modelo,
la curva de aprendizaje y los metadatos del entrenamiento especificado.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "a1": {"mode": GateMode.MINIMUM, "domain_randomization": False},
    "a2": {"mode": GateMode.MINIMUM, "domain_randomization": True},
    "a3": {"mode": GateMode.MAXIMUM, "domain_randomization": True},
    "p2": {"mode": GateMode.INNOVATION, "domain_randomization": True},
}


def make_run_dir(variant: str, seed: int, base_R: float = 0.01, w_res: float = 0.05) -> Path:
    """Crea un directorio nuevo sin sobrescribir un intento anterior."""
    parent = ROOT / "results" / "training" / variant
    if base_R != 0.01 or w_res != 0.05:
        parent = parent / f"baseR_{base_R:g}_wres_{w_res:g}"
    base = parent / f"seed_{seed}"
    if not base.exists():
        return base
    index = 2
    while (parent / f"seed_{seed}_attempt_{index:02d}").exists():
        index += 1
    return parent / f"seed_{seed}_attempt_{index:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena una política SAC residual de P2")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--residual-penalty", type=float, default=0.05,
                        help="Peso de la penalizacion del residuo en la recompensa.")
    parser.add_argument("--base-r", type=float, default=0.01,
                        help="R de diseno del LQR base. 0.01 es el canonico de P1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("timesteps debe ser positivo")
    variant = VARIANTS[args.variant]
    torch.set_num_threads(2)
    set_random_seed(args.seed)

    run_dir = make_run_dir(args.variant, args.seed, args.base_r, args.residual_penalty)
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir()

    env = ResidualBalanceEnv(
        mode=variant["mode"],
        domain_randomization=variant["domain_randomization"],
        base_R=args.base_r,
        residual_penalty=args.residual_penalty,
        seed=args.seed,
    )
    env = Monitor(env, filename=str(run_dir / "monitor.csv"))
    callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(checkpoint_dir),
        name_prefix="sac_residual",
    )
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=150_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        device=args.device,
        verbose=1,
        # TensorBoard es opcional y no forma parte de las dependencias fijadas.
        tensorboard_log=None,
    )
    metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "timesteps_requested": args.timesteps,
        "mode": variant["mode"].value,
        "domain_randomization": variant["domain_randomization"],
        "base_R": args.base_r,
        "residual_penalty": args.residual_penalty,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "policy_architecture": [64, 64],
        "sac": {
            "learning_rate": 3e-4,
            "buffer_size": 150_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "gamma": 0.99,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        # La barra de progreso requiere extras opcionales no fijados en requirements.
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
        model.save(str(run_dir / "model"))
        np.savez_compressed(run_dir / "rng_state.npz", numpy_state=np.array([np.random.get_state()], dtype=object))
    finally:
        env.close()


if __name__ == "__main__":
    main()
