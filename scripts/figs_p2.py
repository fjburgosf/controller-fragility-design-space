"""Genera las figuras del paper P2.

Fig 1 - Arquitectura IG-RRL v3
Fig 2 - Curvas de aprendizaje (reward vs pasos) por variante, 5 seeds
Fig 3 - Comparación de métricas físicas en los 3 escenarios (barras con error)
Fig 4 - Boxplot RMSE_theta por variante y escenario
Fig 5 - Scatter RMSE_theta vs energy (tradeoff performance/energía)
Fig 6 - Success rate por variante y escenario
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "P2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = ROOT / "tables" / "P2"

VARIANTS  = ["lqr", "a1", "a2", "a3", "p2"]
SEEDS     = [1000, 2000, 3000, 4000, 5000]
COLORS    = {"lqr": "#555555", "a1": "#4878CF", "a2": "#6ACC65",
             "a3": "#D65F5F", "p2": "#B47CC7"}
LABELS    = {"lqr": "LQR", "a1": "A1\n(min,no-DR)", "a2": "A2\n(min,DR)",
             "a3": "A3\n(max,DR)", "p2": "IG-RRL\n(P2)"}
SC_LABELS = {"SC1_nominal": "Nominal", "SC2_dr_only": "Incertidumbre", "SC3_dr_perturb": "DR+Perturb"}
TRAIN_DIR = ROOT / "results" / "training"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str):
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {name}", flush=True)


# ── Fig 1: Arquitectura IG-RRL v3 ────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    def box(x, y, w, h, label, color="#DDEEFF"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", fc=color, ec="#555", lw=1.2))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=9, wrap=True)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha="center", fontsize=8, color="#555")

    box(0.2, 1.5, 1.8, 1.0, "Planta\n(no lineal)", "#FFE4B5")
    box(2.5, 2.2, 1.8, 1.0, "EKF\n(nominal)", "#B0E0E6")
    box(2.5, 0.6, 1.8, 1.0, "LQR\n(nominal)", "#C1FFC1")
    box(5.0, 1.5, 1.8, 1.0, "Gate NIS\nρ(η)", "#FFD700")
    box(7.2, 2.2, 1.8, 1.0, "SAC residual\n(dominio random.)", "#E6B0E6")
    box(7.2, 0.6, 1.8, 1.0, "Σ  u = u_base\n  + ρ·ā·tanh(a)", "#FFDAB9")

    arrow(2.0, 2.0, 2.5, 2.5, "y")
    arrow(2.0, 2.0, 2.5, 1.0, "x̂")
    arrow(4.3, 2.5, 5.0, 2.0, "NIS")
    arrow(4.3, 1.0, 7.2, 1.1, "u_base")
    arrow(5.0, 2.0, 7.2, 2.5, "ρ")
    arrow(9.0, 2.5, 9.0, 1.6)
    arrow(9.0, 1.3, 0.2, 2.0, "u")
    ax.text(5.9, 2.9, "observación o", ha="center", fontsize=8, color="#555")
    arrow(5.9, 2.0, 7.2, 2.5)

    ax.set_title("Fig. 1 — Arquitectura IG-RRL v3", fontsize=11, pad=8)
    return fig

save(fig_architecture(), "fig1_architecture")

# ── Fig 2: Curvas de aprendizaje ─────────────────────────────────────────────
def load_monitor(variant, seed):
    p = TRAIN_DIR / variant / f"seed_{seed}" / "monitor.csv"
    if not p.exists():
        return None, None
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        try:
            rows.append((float(parts[0]), float(parts[2])))
        except (ValueError, IndexError):
            pass
    if not rows:
        return None, None
    rew, t = zip(*rows)
    return np.array(t), np.array(rew)

def smooth(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def fig_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
    for ax, variant in zip(axes.flat, ["a1", "a2", "a3", "p2"]):
        for seed in SEEDS:
            t, rew = load_monitor(variant, seed)
            if t is None:
                continue
            sr = smooth(rew)
            ep = np.arange(len(sr))
            ax.plot(ep, sr, alpha=0.6, lw=0.8, color=COLORS[variant])
        ax.set_title(LABELS[variant].replace("\n", " "))
        ax.set_xlabel("Episodios"); ax.set_ylabel("Reward (suavizado)")
        ax.axhline(-20, ls="--", color="#999", lw=0.8, label="-20")
    fig.suptitle("Fig. 2 — Curvas de aprendizaje (5 semillas por variante)", fontsize=11)
    fig.tight_layout()
    return fig

save(fig_learning_curves(), "fig2_learning_curves")

# ── Fig 3–6 requieren resultados de evaluación ────────────────────────────────
eval_csv = ROOT / "results" / "processed" / "formal_evaluation.csv"
if not eval_csv.exists():
    print("  [AVISO] formal_evaluation.csv no disponible aún. Figuras 3-6 omitidas.")
else:
    df = pd.read_csv(eval_csv)
    SCENARIOS = ["SC1_nominal", "SC2_dr_only", "SC3_dr_perturb"]

    # ── Fig 3: barras métricas físicas ───────────────────────────────────────
    def fig_metrics_bars():
        metrics = [("rmse_theta", "RMSE θ (rad)"), ("rmse_x", "RMSE x (m)"),
                   ("energy_mean", "Energía media (V²)")]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (metric, ylabel) in zip(axes, metrics):
            x = np.arange(len(SCENARIOS))
            width = 0.13
            for i, variant in enumerate(VARIANTS):
                sub = df[df["variant"] == variant]
                means, errs = [], []
                for sc in SCENARIOS:
                    sc_sub = sub[sub["scenario"] == sc]
                    means.append(sc_sub[metric].mean())
                    errs.append(sc_sub[metric].std(ddof=1) if len(sc_sub) > 1 else 0)
                ax.bar(x + (i - 2)*width, means, width, yerr=errs,
                       label=LABELS[variant].replace("\n", " "),
                       color=COLORS[variant], alpha=0.85, capsize=3)
            ax.set_ylabel(ylabel); ax.set_xticks(x)
            ax.set_xticklabels([SC_LABELS[s] for s in SCENARIOS], rotation=15)
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle("Fig. 3 — Métricas físicas por variante y escenario", fontsize=11)
        fig.tight_layout()
        return fig
    save(fig_metrics_bars(), "fig3_metrics_bars")

    # ── Fig 4: boxplot RMSE_theta ────────────────────────────────────────────
    def fig_boxplot():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, sc in zip(axes, SCENARIOS):
            sc_df = df[df["scenario"] == sc]
            data = [sc_df[sc_df["variant"] == v]["rmse_theta"].values for v in VARIANTS]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", lw=1.5))
            for patch, v in zip(bp["boxes"], VARIANTS):
                patch.set_facecolor(COLORS[v]); patch.set_alpha(0.8)
            ax.set_xticklabels([LABELS[v].replace("\n", " ") for v in VARIANTS], rotation=15, fontsize=8)
            ax.set_title(SC_LABELS[sc]); ax.set_ylabel("RMSE θ (rad)")
        fig.suptitle("Fig. 4 — Distribución RMSE θ por variante y escenario", fontsize=11)
        fig.tight_layout()
        return fig
    save(fig_boxplot(), "fig4_boxplot_rmse")

    # ── Fig 5: scatter RMSE vs energía ───────────────────────────────────────
    def fig_scatter():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, sc in zip(axes, SCENARIOS):
            sc_df = df[df["scenario"] == sc]
            for v in VARIANTS:
                sub = sc_df[sc_df["variant"] == v]
                ax.scatter(sub["energy_mean"], sub["rmse_theta"],
                           color=COLORS[v], label=LABELS[v].replace("\n", " "),
                           s=60, alpha=0.8, edgecolors="white", lw=0.5)
            ax.set_xlabel("Energía (V²)"); ax.set_ylabel("RMSE θ")
            ax.set_title(SC_LABELS[sc])
        axes[0].legend(fontsize=8)
        fig.suptitle("Fig. 5 — Trade-off desempeño / energía", fontsize=11)
        fig.tight_layout()
        return fig
    save(fig_scatter(), "fig5_scatter_tradeoff")

    # ── Fig 6: success rate ───────────────────────────────────────────────────
    def fig_success():
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(SCENARIOS))
        width = 0.15
        for i, variant in enumerate(VARIANTS):
            sub = df[df["variant"] == variant]
            means = [sub[sub["scenario"] == sc]["success_rate"].mean() for sc in SCENARIOS]
            ax.bar(x + (i-2)*width, means, width,
                   label=LABELS[variant].replace("\n", " "),
                   color=COLORS[variant], alpha=0.85)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Tasa de éxito")
        ax.set_xticks(x)
        ax.set_xticklabels([SC_LABELS[s] for s in SCENARIOS])
        ax.legend(fontsize=9)
        fig.suptitle("Fig. 6 — Tasa de éxito por variante y escenario", fontsize=11)
        fig.tight_layout()
        return fig
    save(fig_success(), "fig6_success_rate")

print("\nFiguras generadas en:", FIG_DIR, flush=True)
