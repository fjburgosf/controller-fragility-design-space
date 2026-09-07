"""Figura 3 de P2. Fragilidad del control aprendido en su propio espacio de configuracion.

Tres paneles.
 (a) sensibilidad a hiperparametros de los dos algoritmos
 (b) variabilidad entre semillas del mejor checkpoint, y diferencia frente al final
 (c) desempeno fisico dentro y fuera del rango de perturbacion de entrenamiento
"""
from pathlib import Path
import glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
PR = ROOT / "results" / "processed"
AZUL, NARANJA, VERDE, ROJO = "#1565C0", "#EF6C00", "#2E7D32", "#B71C1C"

fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.8))

# ---- (a) sensibilidad a hiperparametros ----
h = pd.read_csv(PR / "rl_hyperparam_sweep.csv")
a = ax[0]
for k, (algo, c) in enumerate((("sac", AZUL), ("ddpg", NARANJA))):
    s = h[h.algo == algo].sort_values("eval_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(s)) + k * 0.42 - 0.21
    a.bar(x, s.eval_mean, width=0.38, color=c, edgecolor="k", lw=0.4,
          label=algo.upper(), yerr=s.eval_std, capsize=2, error_kw={"lw": 0.8})
    peor, mejor = s.eval_mean.min(), s.eval_mean.max()
    a.text(0.97, 0.16 - k * 0.09, f"{algo.upper()} spread {peor/mejor:.1f}x",
           transform=a.transAxes, ha="right", fontsize=8.5, color=c, weight="bold")
a.set_xticks(range(len(h[h.algo == 'sac'])))
a.set_xticklabels([f"c{i+1}" for i in range(len(h[h.algo == 'sac']))], fontsize=7.5)
a.set_xlabel("hyperparameter configuration, ranked", fontsize=9)
a.set_ylabel("evaluation return  (0 or below, higher is better)", fontsize=8.5)
a.set_title("(a) configuration sensitivity of the\nlearned controllers", fontsize=9)
a.legend(fontsize=8, loc="lower left"); a.grid(axis="y", alpha=0.3)
a.tick_params(labelsize=8)

# ---- (b) variabilidad entre semillas ----
a = ax[1]
datos = {}
for algo, c in (("sac", AZUL), ("ddpg", NARANJA)):
    ms = [json.load(open(f)) for f in sorted(glob.glob(
        str(ROOT / "results" / "training" / f"standalone_{algo}" / "seed_*" / "meta.json")))]
    if not ms:
        continue
    datos[algo] = (c, [m["best_eval"] for m in ms], [m["final_eval"] for m in ms])
pos = 0; etiq = []
for algo, (c, best, final) in datos.items():
    a.scatter([pos] * len(best), best, color=c, s=55, zorder=3, label=f"{algo.upper()} best")
    a.scatter([pos + 0.32] * len(final), final, color=c, s=55, marker="x", zorder=3,
              label=f"{algo.upper()} final")
    for b, f in zip(best, final):
        a.plot([pos, pos + 0.32], [b, f], color=c, lw=0.7, alpha=0.55, zorder=2)
    a.text(pos + 0.16, -0.155, f"{algo.upper()}, n={len(best)}", ha="center", fontsize=8.5,
           color=c, weight="bold", transform=a.get_xaxis_transform())
    etiq += [pos, pos + 0.32]; pos += 1.0
a.set_xticks(etiq); a.set_xticklabels(["best", "final"] * len(datos), fontsize=7.5)
a.set_ylabel("evaluation return  (0 or below, higher is better)", fontsize=8.5)
a.set_title("(b) seed variability, and why the\ncheckpoint criterion matters", fontsize=9)
a.grid(axis="y", alpha=0.3); a.tick_params(labelsize=8)
a.legend(fontsize=7.5, loc="lower center", ncol=2, framealpha=0.95, borderpad=0.4)
a.margins(y=0.16)

# ---- (c) dentro y fuera de distribucion ----
a = ax[2]
f = pd.read_csv(PR / "final_summary.csv")
f["fam"] = f.method.str.replace(r"_s\d+", "", regex=True)
g = f.groupby(["dv", "fam"]).fell_frac.mean().unstack()
a.axvspan(3.4, 6, color=VERDE, alpha=0.11)
a.axvspan(6, 10.6, color=ROJO, alpha=0.11)
a.text(4.7, 0.97, "inside the training" + chr(10) + "disturbance range", ha="center", va="top",
       fontsize=8, color=VERDE, weight="bold", transform=a.get_xaxis_transform())
a.text(8.4, 0.97, "outside", ha="center", va="top", fontsize=8.5, color=ROJO,
       weight="bold", transform=a.get_xaxis_transform())
for fam, c, mk in (("LQR", VERDE, "s"), ("MPC", "#6A1B9A", "^"),
                   ("SAC", AZUL, "o"), ("DDPG", NARANJA, "D")):
    if fam in g:
        a.plot(g.index, g[fam], marker=mk, color=c, lw=1.9, ms=7, label=fam)
a.set_xlabel("sustained disturbance $d_v$ [V]", fontsize=9)
a.set_ylabel("pendulum fall rate", fontsize=9)
a.set_title("(c) each family has a boundary,\nand they sit at different places", fontsize=9)
a.set_ylim(-0.06, 1.1); a.set_xlim(3.4, 10.6)
a.legend(fontsize=8, loc="lower right", framealpha=0.95); a.grid(alpha=0.3)
a.tick_params(labelsize=8)

fig.suptitle("Fragility of learned control lives in its configuration and in its training distribution",
             fontsize=10.5, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig4_rl_fragility.{ext}", bbox_inches="tight", dpi=600)
print("fig4_rl_fragility guardada")
for algo, (_, b, f) in datos.items():
    print(f"  {algo}: {len(b)} semillas  best media {np.mean(b):.1f}  final media {np.mean(f):.1f}")
