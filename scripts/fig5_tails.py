"""Figura 5. Distribucion de colas del error angular a traves de la frontera.

Una media oculta exactamente los episodios que importan en un sistema inestable.
Esta figura muestra la distribucion completa por familia y nivel de perturbacion,
con la mediana, el percentil 95 y el peor caso marcados.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
e = pd.read_csv(ROOT / "results" / "processed" / "final_episodes.csv")
e["fam"] = e.method.str.replace(r"_s\d+", "", regex=True)

FAMS = ["LQR", "MPC", "SAC", "DDPG"]
COL = {"LQR": "#2E7D32", "MPC": "#6A1B9A", "SAC": "#1565C0", "DDPG": "#EF6C00"}
DVS = sorted(e.dv.unique())
FALL = 0.30

fig, ax = plt.subplots(1, len(DVS), figsize=(13.2, 3.9), sharey=True)
for j, dv in enumerate(DVS):
    a = ax[j]
    datos, pos, etiq = [], [], []
    for i, fam in enumerate(FAMS):
        s = e[(e.dv == dv) & (e.fam == fam)].rmse_theta.values
        if len(s) == 0:
            continue
        datos.append(s); pos.append(i); etiq.append(fam)
    bp = a.boxplot(datos, positions=pos, widths=0.6, patch_artist=True,
                   showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.5),
                   medianprops=dict(color="black", lw=1.4))
    for parche, fam in zip(bp["boxes"], etiq):
        parche.set_facecolor(COL[fam]); parche.set_alpha(0.55); parche.set_edgecolor("black")
    for i, fam in enumerate(etiq):
        s = e[(e.dv == dv) & (e.fam == fam)].rmse_theta.values
        a.plot(i, np.percentile(s, 95), marker=6, color="#212121", ms=11, mew=1.6,
               ls="none", zorder=5)
    a.axhline(FALL, color="#B71C1C", ls="--", lw=1.4)
    if j == 0:
        a.text(3.35, FALL * 1.25, "fall threshold", fontsize=6.8, color="#B71C1C",
               ha="right", va="bottom")
    a.set_yscale("log")
    a.set_xticks(range(len(etiq))); a.set_xticklabels(etiq, fontsize=8, rotation=20)
    dentro = dv <= 6
    a.set_title(f"$d_v$ = {dv:.0f} V\n{'inside' if dentro else 'outside'} training range",
                fontsize=9, color="#2E7D32" if dentro else "#B71C1C")
    a.grid(axis="y", alpha=0.3, which="both"); a.tick_params(labelsize=8)
ax[0].set_ylabel("RMSE $\\theta$ [rad], log scale", fontsize=9)

fig.suptitle("Distribution of angular error by family. The triangle marks the ninety fifth "
             "percentile and the dashed line the fall threshold", fontsize=10, y=1.03)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig5_tails.{ext}", bbox_inches="tight", dpi=600)
print("fig5_tails guardada")
for dv in DVS:
    fila = " | ".join(
        f"{f} p95 {np.percentile(e[(e.dv==dv)&(e.fam==f)].rmse_theta,95):.3f}"
        for f in FAMS if len(e[(e.dv == dv) & (e.fam == f)]))
    print(f"  dv={dv:4.1f}  {fila}")
