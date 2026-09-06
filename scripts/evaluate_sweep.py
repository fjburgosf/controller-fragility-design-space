"""Evalua el barrido factorial a3 (base_R x w_res x seed) con arnes pareado.

Para cada celda: modelo SAC entrenado vs LQR(base_R) puro (residuo=0), sobre las
MISMAS realizaciones (EVAL_SEED0). Mide la GANANCIA del residuo por episodio
(delta = rmse_theta_base - rmse_theta_trained) y si escala con base_R.

Salida:
  results/processed/sweep_evaluation.csv        (agregado por celda)
  results/processed/sweep_evaluation_episodes.csv (por episodio, pareado)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "results" / "training" / "a3"
OUT = ROOT / "results" / "processed" / "sweep_evaluation.csv"
OUT_EP = ROOT / "results" / "processed" / "sweep_evaluation_episodes.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL = 64
BASE_R = [0.01, 1.0, 10.0, 100.0]
W_RES = [0.0, 0.05]
SEEDS = [1000, 2000, 3000]
SCENARIOS = [("SC1_nominal", False, 0.0), ("SC3_dr_perturb", True, 0.5)]


class Zero:
    def predict(self, obs, deterministic=True):
        return np.zeros(1, dtype="float32"), None


def episodes(model, base_R, w_res, dr, perturb, n):
    env = ResidualBalanceEnv(mode=GateMode.MAXIMUM, domain_randomization=dr,
                             perturbation_probability=perturb, base_R=base_R,
                             residual_penalty=w_res)
    out = []
    for ep in range(n):
        obs, _ = env.reset(seed=EVAL_SEED0 + ep)
        done = term = False
        th_sq, x_sq, en = [], [], []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            th_sq.append((env.state[1] - np.pi) ** 2)
            x_sq.append(env.state[0] ** 2)
            en.append(info["u_control"] ** 2)
        out.append({
            "episode": ep,
            "rmse_theta": float(np.sqrt(np.mean(th_sq))),
            "rmse_x": float(np.sqrt(np.mean(x_sq))),
            "energy": float(np.mean(en)),
            "success": 0 if term else 1,
        })
    return out


def cell_dir(r, w):
    if r == 0.01 and w == 0.05:
        return MODEL_ROOT  # canonico: a3/seed_*
    return MODEL_ROOT / f"baseR_{r:g}_wres_{w:g}"


agg, eprows = [], []
for r in BASE_R:
    for w in W_RES:
        cdir = cell_dir(r, w)
        for s in SEEDS:
            mpath = cdir / f"seed_{s}" / "model"
            if not (cdir / f"seed_{s}" / "model.zip").exists():
                print(f"  SKIP R={r} w={w} s={s}", flush=True)
                continue
            model = SAC.load(str(mpath), device="cpu")
            base = Zero()
            for scn, dr, pb in SCENARIOS:
                te = episodes(model, r, w, dr, pb, N_EVAL)
                be = episodes(base, r, w, dr, pb, N_EVAL)
                for t, b in zip(te, be):
                    d = b["rmse_theta"] - t["rmse_theta"]
                    eprows.append({
                        "base_R": r, "w_res": w, "seed": s, "scenario": scn,
                        "episode": t["episode"],
                        "rmse_theta_trained": t["rmse_theta"],
                        "rmse_theta_base": b["rmse_theta"],
                        "delta_rmse_theta": d,
                        "rmse_x_trained": t["rmse_x"], "rmse_x_base": b["rmse_x"],
                        "energy_trained": t["energy"], "energy_base": b["energy"],
                        "success_trained": t["success"], "success_base": b["success"],
                    })
                dd = np.array([b["rmse_theta"] - x["rmse_theta"] for x, b in zip(te, be)])
                tt = np.array([x["rmse_theta"] for x in te])
                bb = np.array([x["rmse_theta"] for x in be])
                agg.append({
                    "base_R": r, "w_res": w, "seed": s, "scenario": scn,
                    "rmse_theta_trained": float(tt.mean()),
                    "rmse_theta_base": float(bb.mean()),
                    "delta_mean": float(dd.mean()),
                    "delta_median": float(np.median(dd)),
                    "delta_frac_positive": float(np.mean(dd > 0)),
                    "success_trained": float(np.mean([x["success"] for x in te])),
                    "success_base": float(np.mean([x["success"] for x in be])),
                })
                print(f"  R={r:6g} w={w:4g} s={s}  {scn:14s}  "
                      f"base={bb.mean():.4f} trained={tt.mean():.4f} "
                      f"delta={dd.mean():+.4f} (+{np.mean(dd>0)*100:.0f}%)", flush=True)

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(agg[0]))
    w.writeheader(); w.writerows(agg)
with OUT_EP.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(eprows[0]))
    w.writeheader(); w.writerows(eprows)
print(f"\nGuardado: {OUT}  ({len(agg)} filas)", flush=True)
print(f"Guardado: {OUT_EP}  ({len(eprows)} filas)", flush=True)

# Resumen: delta medio por (base_R, scenario), promediado sobre seeds y w_res
print("\n=== GANANCIA DEL RESIDUO (delta rmse_theta, + = residuo ayuda) ===", flush=True)
for scn, _, _ in SCENARIOS:
    print(f"\n{scn}:", flush=True)
    for r in BASE_R:
        ds = [a["delta_mean"] for a in agg if a["base_R"] == r and a["scenario"] == scn]
        bs = [a["rmse_theta_base"] for a in agg if a["base_R"] == r and a["scenario"] == scn]
        if ds:
            print(f"  base_R={r:6g}  base_rmse={np.mean(bs):.4f}  "
                  f"delta={np.mean(ds):+.4f} +- {np.std(ds):.4f}", flush=True)
