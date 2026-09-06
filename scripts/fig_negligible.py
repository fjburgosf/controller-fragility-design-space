"""Figura decisiva del estudio negativo P2: diferencia relativa variante-LQR
con banda de negligibilidad +-0.5%. Consume el analisis pareado formal.

Para cada (escenario, metrica en {rmse_theta,rmse_x,energy}) toma el
tamano de efecto en unidades fisicas: (variant_mean-lqr_mean)/lqr_mean*100,
usando p2_wilcoxon_results.csv (ya pareado 64 realizaciones). Marca con * las
diferencias con Holm<0.05. La tesis del paper: todo cae dentro de +-0.5%.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
w = pd.read_csv(ROOT / "tables" / "P2" / "p2_wilcoxon_results.csv")

METRICS = ["rmse_theta", "rmse_x", "energy"]
MLAB = {"rmse_theta": r"RMSE $\theta$", "rmse_x": "RMSE x", "energy": "Energy"}
SC = ["SC1_nominal", "SC2_dr_only", "SC3_dr_perturb"]
SCLAB = {"SC1_nominal": "Nominal", "SC2_dr_only": "DR", "SC3_dr_perturb": "DR+perturb"}
VAR = ["a1", "a2", "a3", "p2"]
COL = {"a1": "#4C72B0", "a2": "#55A868", "a3": "#C44E52", "p2": "#8172B3"}

w = w[w.metric.isin(METRICS)].copy()
w["rel"] = (w.variant_mean - w.lqr_mean) / w.lqr_mean * 100.0

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, m in zip(axes, METRICS):
    sub = w[w.metric == m]
    xt, xl = [], []
    pos = 0
    for sc in SC:
        for v in VAR:
            r = sub[(sub.scenario == sc) & (sub.variant == v)]
            if r.empty:
                pos += 1; continue
            val = r.rel.iloc[0]; sig = r.significativo.iloc[0] == "si"
            ax.bar(pos, val, color=COL[v], edgecolor="k", linewidth=0.4,
                   hatch="//" if not sig else None, alpha=0.9)
            if sig:
                ax.text(pos, val + (0.02 if val >= 0 else -0.02), "*",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
            pos += 1
        xt.append(pos - 2.5); xl.append(SCLAB[sc]); pos += 0.8
    ax.axhspan(-0.5, 0.5, color="0.85", zorder=0, label="±0.5% (negligible)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(MLAB[m]); ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Rel. diff. vs LQR (%)\n(<0 = better)")
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[v]) for v in VAR]
handles.append(plt.Rectangle((0, 0), 1, 1, color="0.85"))
axes[-1].legend(handles, VAR + ["±0.5% band"], fontsize=7, loc="upper right", ncol=1)
fig.suptitle("Residual effect sizes are statistically resolvable but <0.5% (negligible); * = Holm<0.05",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig7_negligible_effects.{ext}", bbox_inches="tight", dpi=300)
print("Guardado: fig7_negligible_effects.{png,pdf}")
print(f"rango rel% observado: [{w.rel.min():.3f}, {w.rel.max():.3f}]")
