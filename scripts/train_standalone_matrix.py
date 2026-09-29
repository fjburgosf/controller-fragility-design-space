"""Matriz completa DRL standalone: {SAC, DDPG} x 5 semillas (P2, D31).

Corrige el defecto de la sonda: SB3 guarda el modelo FINAL, y el entrenamiento
OSCILA (colapso/recuperacion). Aqui se evalua periodicamente sobre un entorno
determinista y se guarda el MEJOR modelo, ademas del final. La diferencia
best-vs-final es en si un indicador de inestabilidad del entrenamiento — una de
las metricas del paper (eje de fragilidad del DRL).

Uso: python scripts/train_standalone_matrix.py --algo sac --seed 1 [--timesteps 300000]
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "ddpg"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    args = ap.parse_args()

    from stable_baselines3 import SAC, DDPG
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.utils import set_random_seed
    from igrrl.env_standalone import StandaloneBalanceEnv

    run = ROOT / "results" / "training" / f"standalone_{args.algo}" / f"seed_{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    env = Monitor(StandaloneBalanceEnv(domain_randomization=True, seed=args.seed),
                  filename=str(run / "monitor.csv"))
    eval_env = Monitor(StandaloneBalanceEnv(domain_randomization=True, seed=args.seed + 50_000))
    cb = EvalCallback(eval_env, best_model_save_path=str(run), log_path=str(run),
                      eval_freq=args.eval_every, n_eval_episodes=10,
                      deterministic=True, verbose=0)

    # Mejor configuracion por algoritmo, hallada en el barrido de hiperparametros
    # (scripts/sweep_rl_hyperparams.py, semilla de tuning 101, disjunta de estas).
    BEST = {"sac": (3e-4, [64, 64]), "ddpg": (1e-4, [256, 256])}
    lr, arch = BEST[args.algo]
    common = dict(learning_rate=lr, buffer_size=150_000, learning_starts=5_000,
                  batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                  policy_kwargs={"net_arch": arch}, seed=args.seed, verbose=0)
    if args.algo == "sac":
        model = SAC("MlpPolicy", env, **common)
    else:
        model = DDPG("MlpPolicy", env,
                     action_noise=NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)), **common)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    model.save(str(run / "model_final"))
    mins = (time.time() - t0) / 60

    ev = run / "evaluations.npz"
    meta = {"algo": args.algo, "seed": args.seed, "timesteps": args.timesteps,
            "minutes": round(mins, 1)}
    if ev.exists():
        d = np.load(ev)
        res = d["results"].mean(axis=1)
        meta.update(best_eval=float(res.max()), final_eval=float(res[-1]),
                    best_at=int(d["timesteps"][int(np.argmax(res))]),
                    # indicador de inestabilidad: cuanto cae respecto al mejor
                    degradation=float(res.max() - res[-1]))
    (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{args.algo} s{args.seed}] {mins:.1f} min  " +
          (f"best={meta['best_eval']:.1f} final={meta['final_eval']:.1f} "
           f"degrad={meta['degradation']:.1f}" if "best_eval" in meta else ""), flush=True)


if __name__ == "__main__":
    main()
