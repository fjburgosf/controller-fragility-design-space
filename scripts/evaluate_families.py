"""Evaluacion final comparativa de las TRES familias (P2, D31).

Todas sobre las MISMAS realizaciones pareadas, con metricas de media Y COLA.
  - LQR            : sin libertad de diseno
  - MPC(config)    : varias configuraciones del espacio de diseno
  - SAC / DDPG     : 5 semillas c/u, modelo MEJOR (no el final)

Guarda agregados y episodios individuales para estadistica pareada
(Wilcoxon/Holm/McNemar con statistical_analysis.py).

Uso: python scripts/evaluate_families.py [--n 64] [--dv 8.0]
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np

from igrrl.evaluate_common import (make_lqr, make_policy, run_episode, RAIL,
                                   FALL_TH, EVAL_SEED0, NOM)
from igrrl.mpc_design import DesignMPC

ROOT = Path(__file__).resolve().parent.parent
OUT_AGG = ROOT / "results" / "processed" / "families_summary.csv"
OUT_EP = ROOT / "results" / "processed" / "families_episodes.csv"

MPC_CONFIGS = [
    ("MPC_rapido_front_ricc", dict(dt=0.005, T_pred=1.5, blocking="front", terminal="riccati")),
    ("MPC_lento_front_ricc",  dict(dt=0.020, T_pred=1.5, blocking="front", terminal="riccati")),
    ("MPC_lento_unif_stage",  dict(dt=0.020, T_pred=0.8, blocking="uniform", terminal="stage")),
]


def tail_stats(eps):
    th = np.array([e["rmse_theta"] for e in eps]); mx = np.array([e["maxx"] for e in eps])
    en = np.array([e["energy"] for e in eps]); rx = np.array([e["rmse_x"] for e in eps])
    return {"rmse_theta_mean": float(th.mean()), "rmse_theta_p95": float(np.percentile(th, 95)),
            "rmse_theta_worst": float(th.max()), "rmse_x_mean": float(rx.mean()),
            "maxx_mean": float(mx.mean()), "maxx_p95": float(np.percentile(mx, 95)),
            "maxx_worst": float(mx.max()), "energy_mean": float(en.mean()),
            "fell_frac": float(np.mean([e["fell"] for e in eps])),
            "rail_viol_frac": float(np.mean([e["rail_viol"] for e in eps])),
            "success_rate": float(1 - np.mean([e["fell"] for e in eps]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dv", type=float, default=8.0)
    args = ap.parse_args()

    agg_rows, ep_rows = [], []

    def add(name, family, ctrl, seed_tag=""):
        eps = [run_episode(ctrl, EVAL_SEED0 + i, dv=args.dv, dr=True) for i in range(args.n)]
        agg_rows.append({"method": name, "family": family, "seed": seed_tag, "n": args.n,
                         "dv": args.dv, **tail_stats(eps)})
        for i, e in enumerate(eps):
            ep_rows.append({"method": name, "family": family, "seed": seed_tag,
                            "episode": i, "eval_seed": EVAL_SEED0 + i, **e})
        a = agg_rows[-1]
        print(f"{name:26} cae={a['fell_frac']:.2f} rmseTh={a['rmse_theta_mean']:.4f} "
              f"(p95 {a['rmse_theta_p95']:.4f}) rmseX={a['rmse_x_mean']:.4f} "
              f"p95|x|={a['maxx_p95']:.3f} viola={a['rail_viol_frac']:.2f} "
              f"E={a['energy_mean']:.1f}", flush=True)

    print(f"n={args.n} dv={args.dv} DR activo | RAIL={RAIL} FALL_TH={FALL_TH}\n", flush=True)
    add("LQR", "clasico", make_lqr())

    for name, cfg in MPC_CONFIGS:
        try:
            m = DesignMPC(NOM, **cfg)
            add(name, "predictivo", lambda t, xh, _m=m: _m.compute(t, xh))
        except Exception as ex:
            print(f"{name}: FALLO {ex.__class__.__name__}: {ex}", flush=True)

    from stable_baselines3 import SAC, DDPG
    for algo, cls in (("sac", SAC), ("ddpg", DDPG)):
        for s in (1, 2, 3, 4, 5):
            p = ROOT / "results" / "training" / f"standalone_{algo}" / f"seed_{s}" / "best_model.zip"
            if not p.exists():
                continue
            try:
                model = cls.load(str(p.with_suffix("")))
                add(f"{algo.upper()}_s{s}", "aprendido", make_policy(model), seed_tag=str(s))
            except Exception as ex:
                print(f"{algo} s{s}: FALLO {ex.__class__.__name__}: {ex}", flush=True)

    if agg_rows:
        with OUT_AGG.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys())); w.writeheader(); w.writerows(agg_rows)
        with OUT_EP.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys())); w.writeheader(); w.writerows(ep_rows)
        print(f"\nGuardado: {OUT_AGG.name}, {OUT_EP.name}", flush=True)


if __name__ == "__main__":
    main()
