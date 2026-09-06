"""SONDA de viabilidad — SAC standalone vs LQR coste-igualado.

Decide CON DATOS si vale la pena la matriz completa (2 algos x 5 semillas) del
estudio "DRL vs control optimo clasico bajo incertidumbre y perturbaciones".

Entrena 1 semilla de SAC standalone (control completo, sin LQR base) a 300k
pasos con domain randomization, y la evalua fisicamente contra el LQR
coste-igualado (Q_P2, R_P2 = el coste exacto de la recompensa) sobre las MISMAS
realizaciones (mismo seed -> mismos parametros, perturbacion y ruido).

CRITERIO VERDE (congelado ANTES de ver datos):
  (i)  exito >= 0.95 en escenario NOMINAL (sin DR, sin perturbacion)
       -> es un controlador funcional, no ruido.
  (ii) bajo DR + perturbaciones, rmse_theta(SAC) <= 2.0 * rmse_theta(LQR)
       -> es competitivo; la comparacion del paper seria justa.
VERDE = (i) y (ii). Si falla (i): el DRL standalone no resuelve esto de forma
fiable ni a 300k -> hallazgo reportable y se cae al plan B (Monte Carlo colas).

Salida: results/processed/standalone_rl_probe.csv
"""
from __future__ import annotations

import argparse, csv, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / "standalone_rl_probe.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
RUN = ROOT / "results" / "training" / "standalone_sac" / "probe_seed1"
RUN.mkdir(parents=True, exist_ok=True)

N_EVAL = 64
EVAL_SEED0 = 9000


def train(timesteps: int, seed: int):
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from igrrl.env_standalone import StandaloneBalanceEnv

    set_random_seed(seed)
    env = Monitor(StandaloneBalanceEnv(domain_randomization=True, seed=seed),
                  filename=str(RUN / "monitor.csv"))
    model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=150_000,
                learning_starts=5_000, batch_size=256, tau=0.005, gamma=0.99,
                train_freq=1, policy_kwargs={"net_arch": [64, 64]}, seed=seed,
                verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(str(RUN / "model"))
    print(f"entrenado {timesteps} pasos en {(time.time()-t0)/60:.1f} min -> {RUN/'model.zip'}", flush=True)
    return model


def rollout(policy, dr: bool, perturb: float, seed: int):
    """policy=None -> LQR coste-igualado. Mismo seed => misma realizacion."""
    from igrrl.env_standalone import StandaloneBalanceEnv
    from src.controllers.lqr import LQRController
    from igrrl.mpc_constrained import Q_P2, R_P2

    env = StandaloneBalanceEnv(domain_randomization=dr, perturbation_probability=perturb)
    obs, _ = env.reset(seed=seed)
    lqr = None if policy is not None else LQRController(env.nominal_params, Q=Q_P2.copy(), R=R_P2)
    ths, xs, us = [], [], []
    term = False
    while True:
        if policy is not None:
            a, _ = policy.predict(obs, deterministic=True)
        else:
            u = lqr.compute(env.steps * env.dt, env.xhat.copy())
            a = np.array([np.arctanh(np.clip(u / env.actuator_limit, -0.999999, 0.999999))])
        obs, r, terminated, truncated, info = env.step(np.asarray(a, dtype=float).reshape(1))
        te = np.arctan2(np.sin(env.state[1] - np.pi), np.cos(env.state[1] - np.pi))
        ths.append(te**2); xs.append(env.state[0]); us.append(info["u_control"])
        if terminated or truncated:
            term = terminated
            break
    return {"success": int(not term),
            "rmse_theta": float(np.sqrt(np.mean(ths))),
            "rmse_x": float(np.sqrt(np.mean(np.square(xs)))),
            "energy": float(np.sum(np.square(us)) * env.dt)}


def batch(policy, dr, perturb):
    rs = [rollout(policy, dr, perturb, EVAL_SEED0 + i) for i in range(N_EVAL)]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    from stable_baselines3 import SAC
    if args.skip_train and (RUN / "model.zip").exists():
        model = SAC.load(str(RUN / "model"))
        print("modelo cargado", flush=True)
    else:
        model = train(args.timesteps, args.seed)

    SCEN = [("SC1_nominal", False, 0.0), ("SC3_dr_perturb", True, 0.5)]
    rows = []
    print(f"\n{'escenario':>16} {'metodo':>6} {'exito':>6} {'rmseTh':>8} {'rmseX':>8} {'E':>8}", flush=True)
    res = {}
    for name, dr, pp in SCEN:
        for tag, pol in (("SAC", model), ("LQR", None)):
            a = batch(pol, dr, pp)
            a.update(scenario=name, method=tag)
            rows.append(a); res[(name, tag)] = a
            print(f"{name:>16} {tag:>6} {a['success']:6.3f} {a['rmse_theta']:8.4f} "
                  f"{a['rmse_x']:8.4f} {a['energy']:8.2f}", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "method", "success", "rmse_theta", "rmse_x", "energy"])
        w.writeheader(); w.writerows(rows)

    i = res[("SC1_nominal", "SAC")]["success"] >= 0.95
    lqr3 = res[("SC3_dr_perturb", "LQR")]["rmse_theta"]
    ii = res[("SC3_dr_perturb", "SAC")]["rmse_theta"] <= 2.0 * lqr3
    print(f"\n(i)  exito nominal >= 0.95 : {i}  ({res[('SC1_nominal','SAC')]['success']:.3f})", flush=True)
    print(f"(ii) rmseTh SAC <= 2x LQR  : {ii}  "
          f"({res[('SC3_dr_perturb','SAC')]['rmse_theta']:.4f} vs {2*lqr3:.4f})", flush=True)
    print(f"\n>>> {'VERDE: lanzar matriz completa (2 algos x 5 semillas)' if (i and ii) else 'ROJO: caer a plan B (Monte Carlo colas)'}", flush=True)
