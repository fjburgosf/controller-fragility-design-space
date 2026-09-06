"""Barrido de hiperparametros del DRL — analogo al espacio de diseno del MPC (D31).

Responde: cuan fragil es el desempeno del DRL frente a su propia configuracion?
Es la contraparte del barrido de {dt, T_pred, blocking, terminal} del MPC.

Separacion tuning/validacion: el barrido usa SEMILLAS DE TUNING
(101+) y su propio entorno de evaluacion. Las semillas de validacion (1-5) y las
realizaciones de test (EVAL_SEED0=9000) NO se tocan aqui.

Uso: python scripts/sweep_rl_hyperparams.py --algo sac [--timesteps 150000]
"""
from __future__ import annotations

import argparse, csv, itertools, json, os, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / "rl_hyperparam_sweep.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

LRS = [1e-4, 3e-4, 1e-3]
ARCHS = [[64, 64], [256, 256]]
TUNE_SEED = 101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "ddpg"], required=True)
    ap.add_argument("--timesteps", type=int, default=150_000)
    args = ap.parse_args()

    from stable_baselines3 import SAC, DDPG
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.utils import set_random_seed
    from igrrl.env_standalone import StandaloneBalanceEnv

    rows = []
    print(f"[{args.algo}] barrido {len(LRS)}x{len(ARCHS)} configs, {args.timesteps} pasos, "
          f"semilla de tuning {TUNE_SEED}\n", flush=True)
    print(f"{'lr':>8} {'arch':>12} {'eval_mean':>11} {'eval_std':>9} {'min':>10} {'max':>10} {'min_ep':>8}", flush=True)

    for lr, arch in itertools.product(LRS, ARCHS):
        set_random_seed(TUNE_SEED)
        env = Monitor(StandaloneBalanceEnv(domain_randomization=True, seed=TUNE_SEED))
        common = dict(learning_rate=lr, buffer_size=150_000, learning_starts=5_000,
                      batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                      policy_kwargs={"net_arch": arch}, seed=TUNE_SEED, verbose=0)
        model = (SAC("MlpPolicy", env, **common) if args.algo == "sac" else
                 DDPG("MlpPolicy", env,
                      action_noise=NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)), **common))
        t0 = time.time()
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        mins = (time.time() - t0) / 60

        # evaluacion en un entorno con semilla DISTINTA (no vista en entrenamiento)
        ev = StandaloneBalanceEnv(domain_randomization=True, seed=TUNE_SEED + 900)
        rew, _ = evaluate_policy(model, ev, n_eval_episodes=30, deterministic=True,
                                 return_episode_rewards=True)
        rew = np.asarray(rew, dtype=float)
        # longitud de episodio como proxy de supervivencia
        ev2 = StandaloneBalanceEnv(domain_randomization=True, seed=TUNE_SEED + 900)
        lens = []
        for _ in range(30):
            o, _ = ev2.reset(); n = 0
            while True:
                a, _ = model.predict(o, deterministic=True)
                o, r, term, trunc, _i = ev2.step(a); n += 1
                if term or trunc:
                    break
            lens.append(n)
        row = {"algo": args.algo, "lr": lr, "arch": "x".join(map(str, arch)),
               "timesteps": args.timesteps, "minutes": round(mins, 1),
               "eval_mean": float(rew.mean()), "eval_std": float(rew.std()),
               "eval_min": float(rew.min()), "eval_max": float(rew.max()),
               "len_min": int(min(lens)), "len_mean": float(np.mean(lens))}
        rows.append(row)
        print(f"{lr:8.0e} {row['arch']:>12} {row['eval_mean']:11.1f} {row['eval_std']:9.1f} "
              f"{row['eval_min']:10.1f} {row['eval_max']:10.1f} {row['len_min']:8d}", flush=True)

    hdr = list(rows[0].keys())
    exists = OUT.exists()
    with OUT.open("a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["eval_mean"])
    print(f"\nMEJOR [{args.algo}]: lr={best['lr']:.0e} arch={best['arch']} "
          f"eval={best['eval_mean']:.1f}+-{best['eval_std']:.1f}", flush=True)


if __name__ == "__main__":
    main()
