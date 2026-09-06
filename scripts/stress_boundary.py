"""Diagnóstico de frontera de fallo.

Pregunta: ¿existe un régimen donde LQR se degrada y el residuo lo sostiene?

Escala incertidumbre paramétrica y amplitud de perturbación en niveles crecientes.
Para cada nivel mide success_rate y RMSE de LQR puro vs IG-RRL (P2) vs A3.

Salida: results/processed/stress_boundary.csv
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results" / "training"
OUT_CSV   = ROOT / "results" / "processed" / "stress_boundary.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

SEEDS  = [1000, 2000, 3000, 4000, 5000]
N_EVAL = 48
# Semilla base unica: todas las metodologias comparten las mismas realizaciones.
EVAL_SEED0 = 9000

# (nivel, escala_param, escala_disturbio) — escala_param multiplica la semiamplitud
# nominal de cada parámetro; escala_disturbio multiplica la amplitud del disturbio.
STRESS_LEVELS = [
    ("L0", 1.0, 1.0),
    ("L1", 1.5, 1.5),
    ("L2", 2.0, 2.0),
    ("L3", 2.5, 3.0),
    ("L4", 3.0, 4.0),
    ("L5", 4.0, 5.0),
]

# Semiamplitudes nominales del entorno base (factor multiplicativo sobre el parámetro)
NOMINAL_HALFWIDTH = {"Mp": 0.20, "Jp": 0.25, "l": 0.15, "kt": 0.10}


class StressEnv(ResidualBalanceEnv):
    """Entorno con incertidumbre y perturbación escaladas para hallar el límite."""

    param_scale: float = 1.0
    dist_scale: float = 1.0

    def _sample_params(self):
        if not self.domain_randomization:
            return self.nominal_params
        factors = {}
        for key, hw in NOMINAL_HALFWIDTH.items():
            half = min(hw * self.param_scale, 0.95)  # no permitir parámetro <=0
            factors[key] = self.np_random.uniform(1.0 - half, 1.0 + half)
        return replace(self.nominal_params,
                       **{k: getattr(self.nominal_params, k) * v for k, v in factors.items()})

    def _sample_disturbance(self):
        if self.np_random.uniform() >= self.perturbation_probability:
            return {"kind": "none"}
        kind = "impulse" if self.np_random.uniform() < 0.5 else "sustained"
        start = int(self.np_random.integers(int(2.0 / self.dt), int(3.5 / self.dt)))
        if kind == "impulse":
            amp = float(self.np_random.uniform(4.0, 10.0)) * self.dist_scale
            return {"kind": kind, "start": start, "duration": int(0.10 / self.dt), "amplitude": amp}
        amp = float(self.np_random.uniform(1.0, 4.0)) * self.dist_scale
        return {"kind": kind, "start": start, "duration": int(0.75 / self.dt), "amplitude": amp}


class ZeroResidualModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(1, dtype="float32"), None


def evaluate(model, mode, param_scale, dist_scale, n, seed0):
    env = StressEnv(mode=mode, domain_randomization=True, perturbation_probability=1.0)
    env.param_scale = param_scale
    env.dist_scale = dist_scale
    successes, th_rms, x_rms = [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        done, term = False, False
        th_sq, x_sq = [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            th_sq.append((env.state[1] - np.pi) ** 2)
            x_sq.append(env.state[0] ** 2)
        successes.append(0 if term else 1)
        if not term:
            th_rms.append(np.sqrt(np.mean(th_sq)))
            x_rms.append(np.sqrt(np.mean(x_sq)))
    return {
        "success_rate": float(np.mean(successes)),
        "rmse_theta": float(np.mean(th_rms)) if th_rms else float("nan"),
        "rmse_x": float(np.mean(x_rms)) if x_rms else float("nan"),
        "n_stable": int(np.sum(successes)),
    }


rows = []

# Precarga modelos
p2_models = {s: SAC.load(str(MODEL_DIR / "p2" / f"seed_{s}" / "model"), device="cpu") for s in SEEDS}
a3_models = {s: SAC.load(str(MODEL_DIR / "a3" / f"seed_{s}" / "model"), device="cpu") for s in SEEDS}
lqr = ZeroResidualModel()

for level, pscale, dscale in STRESS_LEVELS:
    print(f"=== {level}  param×{pscale}  dist×{dscale} ===", flush=True)

    m = evaluate(lqr, GateMode.MINIMUM, pscale, dscale, N_EVAL, seed0=EVAL_SEED0)
    rows.append({"level": level, "param_scale": pscale, "dist_scale": dscale,
                 "variant": "lqr", "seed": 0, **m})
    print(f"  LQR      success={m['success_rate']:.3f}  rmse_th={m['rmse_theta']:.4f}", flush=True)

    for s in SEEDS:
        m = evaluate(p2_models[s], GateMode.INNOVATION, pscale, dscale, N_EVAL, seed0=EVAL_SEED0)
        rows.append({"level": level, "param_scale": pscale, "dist_scale": dscale,
                     "variant": "p2", "seed": s, **m})
    p2_succ = np.mean([r["success_rate"] for r in rows if r["level"] == level and r["variant"] == "p2"])
    print(f"  P2  (5s) success={p2_succ:.3f}", flush=True)

    for s in SEEDS:
        m = evaluate(a3_models[s], GateMode.MAXIMUM, pscale, dscale, N_EVAL, seed0=EVAL_SEED0)
        rows.append({"level": level, "param_scale": pscale, "dist_scale": dscale,
                     "variant": "a3", "seed": s, **m})
    a3_succ = np.mean([r["success_rate"] for r in rows if r["level"] == level and r["variant"] == "a3"])
    print(f"  A3  (5s) success={a3_succ:.3f}", flush=True)

fieldnames = ["level", "param_scale", "dist_scale", "variant", "seed",
              "success_rate", "rmse_theta", "rmse_x", "n_stable"]
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\nGuardado: {OUT_CSV}", flush=True)

# Resumen: tabla success_rate por nivel
print("\n=== FRONTERA DE FALLO (success_rate) ===", flush=True)
print(f"{'Nivel':6s} {'LQR':>8s} {'P2':>8s} {'A3':>8s}", flush=True)
for level, pscale, dscale in STRESS_LEVELS:
    lqr_s = np.mean([r["success_rate"] for r in rows if r["level"] == level and r["variant"] == "lqr"])
    p2_s  = np.mean([r["success_rate"] for r in rows if r["level"] == level and r["variant"] == "p2"])
    a3_s  = np.mean([r["success_rate"] for r in rows if r["level"] == level and r["variant"] == "a3"])
    print(f"{level:6s} {lqr_s:8.3f} {p2_s:8.3f} {a3_s:8.3f}", flush=True)
