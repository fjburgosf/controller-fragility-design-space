"""Evaluación física formal de los 20 modelos entrenados (4 variantes × 5 semillas).

Calcula RMSE_theta, RMSE_x, energía, success_rate y return en tres escenarios:
  SC1 - nominal        : sin DR, sin perturbación
  SC2 - incertidumbre  : DR activo, sin perturbación
  SC3 - DR + perturb   : DR activo, perturbación_prob=0.5

Guarda resultados en results/processed/formal_evaluation.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results" / "training"
OUT_CSV   = ROOT / "results" / "processed" / "formal_evaluation.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

VARIANTS = ["a1", "a2", "a3", "p2"]
SEEDS    = [1000, 2000, 3000, 4000, 5000]
N_EVAL   = 64   # episodios por escenario
# PROTOCOL.md exige que las realizaciones Monte Carlo se emparejen entre metodos.
# Una unica semilla base garantiza que LQR y todas las variantes vean exactamente
# los mismos parametros muestreados, condiciones iniciales, ruido y perturbaciones.
EVAL_SEED0 = 9000

SCENARIOS = [
    ("SC1_nominal",    False, 0.0),
    ("SC2_dr_only",    True,  0.0),
    ("SC3_dr_perturb", True,  0.5),
]

VARIANT_SETTINGS = {
    "a1": (GateMode.MINIMUM, False),
    "a2": (GateMode.MINIMUM, True),
    "a3": (GateMode.MAXIMUM, True),
    "p2": (GateMode.INNOVATION, True),
}


class ZeroResidualModel:
    """LQR puro: acción residual = 0."""
    def predict(self, obs, deterministic=True):
        return np.zeros(1, dtype="float32"), None


def run_episodes(model, mode: GateMode, dr: bool, perturb: float, n: int, seed0: int = EVAL_SEED0):
    env = ResidualBalanceEnv(mode=mode, domain_randomization=dr, perturbation_probability=perturb)
    rets, th_rms, x_rms, energies, successes = [], [], [], [], []
    per_ep: list[dict] = []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        ret, done = 0.0, False
        th_sq, x_sq, en_sq = [], [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            ret += rew
            th_sq.append((env.state[1] - np.pi) ** 2)
            x_sq.append(env.state[0] ** 2)
            en_sq.append(info["u_control"] ** 2)
        rets.append(ret)
        th_rms.append(np.sqrt(np.mean(th_sq)))
        x_rms.append(np.sqrt(np.mean(x_sq)))
        energies.append(np.mean(en_sq))
        successes.append(0 if term else 1)
        per_ep.append({"episode": ep, "eval_seed": seed0 + ep, "return": float(ret),
                       "rmse_theta": float(th_rms[-1]), "rmse_x": float(x_rms[-1]),
                       "energy": float(energies[-1]), "success": successes[-1]})
    return {
        "return_mean":  float(np.mean(rets)),
        "return_std":   float(np.std(rets)),
        "rmse_theta":   float(np.mean(th_rms)),
        "rmse_x":       float(np.mean(x_rms)),
        "energy_mean":  float(np.mean(energies)),
        "success_rate": float(np.mean(successes)),
    }, per_ep


rows: list[dict] = []
ep_rows: list[dict] = []

# LQR baseline (no residual)
lqr = ZeroResidualModel()
for sc_name, dr, perturb in SCENARIOS:
    print(f"  LQR  {sc_name}", flush=True)
    metrics, eps = run_episodes(lqr, GateMode.MINIMUM, dr, perturb, N_EVAL, EVAL_SEED0)
    rows.append({"variant": "lqr", "seed": 0, "scenario": sc_name, **metrics})
    ep_rows += [{"variant": "lqr", "seed": 0, "scenario": sc_name, **e} for e in eps]

# Trained variants
for variant in VARIANTS:
    mode, _ = VARIANT_SETTINGS[variant]
    for seed in SEEDS:
        model_path = MODEL_DIR / variant / f"seed_{seed}" / "model"
        if not (MODEL_DIR / variant / f"seed_{seed}" / "model.zip").exists():
            print(f"  SKIP {variant} seed {seed} (model not found)", flush=True)
            continue
        model = SAC.load(str(model_path), device="cpu")
        for sc_name, dr, perturb in SCENARIOS:
            print(f"  {variant} seed {seed}  {sc_name}", flush=True)
            metrics, eps = run_episodes(model, mode, dr, perturb, N_EVAL, EVAL_SEED0)
            rows.append({"variant": variant, "seed": seed, "scenario": sc_name, **metrics})
            ep_rows += [{"variant": variant, "seed": seed, "scenario": sc_name, **e} for e in eps]

fieldnames = ["variant", "seed", "scenario",
              "return_mean", "return_std",
              "rmse_theta", "rmse_x",
              "energy_mean", "success_rate"]

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\nGuardado: {OUT_CSV}", flush=True)
print(f"Total filas: {len(rows)}", flush=True)

OUT_EP = ROOT / "results" / "processed" / "formal_evaluation_episodes.csv"
ep_fields = ["variant", "seed", "scenario", "episode", "eval_seed",
             "return", "rmse_theta", "rmse_x", "energy", "success"]
with OUT_EP.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ep_fields)
    w.writeheader()
    w.writerows(ep_rows)
print(f"Guardado: {OUT_EP}  ({len(ep_rows)} filas)", flush=True)
