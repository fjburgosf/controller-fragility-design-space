"""Analisis estadistico formal (pareado por realizacion).

Consume results/processed/formal_evaluation_episodes.csv (64 episodios por
variante-seed-escenario, EVAL_SEED0=9000: LQR y todas las variantes ven la
MISMA realizacion en cada episodio).

Diseno pareado: para cada (escenario, metrica, variante en {a1,a2,a3,p2}) se
promedian las 5 seeds de entrenamiento de la variante en cada eval_seed,
obteniendo 64 valores; se comparan contra los 64 valores del LQR en las mismas
realizaciones. Wilcoxon de rangos con signo pareado (n=64), tamano de efecto
rango biserial r, correccion Holm por familia (4 variantes) en cada
escenario-metrica. Para exito: McNemar pareado.

Convencion: en rmse/energia MENOR es mejor. r<0 => variante < LQR => mejor.

Guarda en tables/P2/:
  p2_performance_summary.csv   (mean +- std por variante-escenario-metrica)
  p2_wilcoxon_results.csv      (pareado 64 ep)
  p2_mcnemar_success.csv
  p2_final_comparison.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
IN_EP = ROOT / "results" / "processed" / "formal_evaluation_episodes.csv"
TAB = ROOT / "tables" / "P2"
TAB.mkdir(parents=True, exist_ok=True)

METRICS = ["rmse_theta", "rmse_x", "energy", "return"]
LOWER_BETTER = {"rmse_theta": True, "rmse_x": True, "energy": True, "return": False}
VARIANTS = ["a1", "a2", "a3", "p2"]


def rank_biserial(x, y):
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    n = len(d)
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / (n * (n + 1) / 2))


def holm(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    corr = np.array(pvals, dtype=float)
    for i, idx in enumerate(order):
        corr[idx] = min(1.0, pvals[idx] * (n - i))
    sc = corr[order]
    for i in range(1, len(sc)):
        sc[i] = max(sc[i], sc[i - 1])
    corr[order] = sc
    return corr.tolist()


df = pd.read_csv(IN_EP)
SCENARIOS = df["scenario"].unique().tolist()


def paired_vectors(sc_df, variant, metric):
    lqr = sc_df[sc_df["variant"] == "lqr"].set_index("episode")[metric].sort_index()
    var = sc_df[sc_df["variant"] == variant].groupby("episode")[metric].mean().sort_index()
    idx = lqr.index.intersection(var.index)
    return lqr.loc[idx].to_numpy(), var.loc[idx].to_numpy()


# --- Tabla resumen mean +- std ---
summary = []
for sc in SCENARIOS:
    scd = df[df["scenario"] == sc]
    for v in ["lqr"] + VARIANTS:
        for m in METRICS:
            vals = scd[scd["variant"] == v].groupby("episode")[m].mean().to_numpy()
            if len(vals) == 0:
                continue
            summary.append({"scenario": sc, "variant": v, "metric": m,
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "n_realizaciones": int(len(vals))})
sdf = pd.DataFrame(summary)
sdf.to_csv(TAB / "p2_performance_summary.csv", index=False)
print("Guardado: p2_performance_summary.csv", flush=True)

# --- Wilcoxon pareado 64 ep ---
wil = []
for sc in SCENARIOS:
    scd = df[df["scenario"] == sc]
    for m in METRICS:
        praw, comps = [], []
        for v in VARIANTS:
            lqr_v, var_v = paired_vectors(scd, v, m)
            if len(lqr_v) < 5:
                continue
            try:
                stat, p = stats.wilcoxon(var_v, lqr_v, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                stat, p = 0.0, 1.0
            r = rank_biserial(var_v, lqr_v)
            better = (r < 0) if LOWER_BETTER[m] else (r > 0)
            comps.append((v, stat, p, r, better, float(var_v.mean()), float(lqr_v.mean())))
            praw.append(p)
        for (v, stat, p, r, better, vm, lm), padj in zip(comps, holm(praw)):
            wil.append({"scenario": sc, "metric": m, "variant": v,
                        "variant_mean": round(vm, 5), "lqr_mean": round(lm, 5),
                        "wilcoxon_stat": round(stat, 2), "p_raw": round(p, 5),
                        "p_holm": round(padj, 5), "rank_biserial_r": round(r, 4),
                        "significativo": "si" if padj < 0.05 else "no",
                        "direccion": "mejor" if better else "peor"})
wdf = pd.DataFrame(wil)
wdf.to_csv(TAB / "p2_wilcoxon_results.csv", index=False)
print("Guardado: p2_wilcoxon_results.csv", flush=True)

# --- McNemar de exito pareado ---
mc = []
for sc in SCENARIOS:
    scd = df[df["scenario"] == sc]
    lqr_s = scd[scd["variant"] == "lqr"].set_index("episode")["success"].sort_index()
    for v in VARIANTS:
        var_s = scd[scd["variant"] == v].groupby("episode")["success"].min().sort_index()
        idx = lqr_s.index.intersection(var_s.index)
        L = lqr_s.loc[idx].to_numpy().astype(int)
        V = var_s.loc[idx].to_numpy().astype(int)
        b = int(np.sum((V == 0) & (L == 1)))
        c = int(np.sum((V == 1) & (L == 0)))
        p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
        mc.append({"scenario": sc, "variant": v, "var_solo_falla": b, "lqr_solo_falla": c,
                   "p_mcnemar": round(p, 4),
                   "lqr_success": round(float(L.mean()), 4),
                   "var_success": round(float(V.mean()), 4)})
pd.DataFrame(mc).to_csv(TAB / "p2_mcnemar_success.csv", index=False)
print("Guardado: p2_mcnemar_success.csv", flush=True)

# --- Comparacion final Markdown ---
lines = ["# Comparacion final P2 vs baselines (pareado 64 realizaciones)\n"]
for sc in SCENARIOS:
    lines.append("\n## " + sc + "\n\n")
    lines.append("| Variante | RMSE theta | RMSE x | Energia | Return |\n")
    lines.append("|---|---|---|---|---|\n")
    for v in ["lqr"] + VARIANTS:
        def cell(m):
            row = sdf[(sdf.scenario == sc) & (sdf.variant == v) & (sdf.metric == m)]
            return "-" if row.empty else f"{row['mean'].iloc[0]:.4f}±{row['std'].iloc[0]:.4f}"
        lines.append(f"| {v} | {cell('rmse_theta')} | {cell('rmse_x')} | {cell('energy')} | {cell('return')} |\n")
with (TAB / "p2_final_comparison.md").open("w", encoding="utf-8") as f:
    f.writelines(lines)
print("Guardado: p2_final_comparison.md", flush=True)

# --- Resumen a consola ---
print("\n=== RESUMEN (pareado 64 ep) ===", flush=True)
for sc in SCENARIOS:
    print("\n--- " + sc + " ---")
    for v in ["lqr"] + VARIANTS:
        r = sdf[(sdf.scenario == sc) & (sdf.variant == v)]
        if r.empty:
            continue
        g = lambda m: r[r.metric == m]["mean"].iloc[0]
        print(f"  {v:4s} rmseTh={g('rmse_theta'):.4f} rmseX={g('rmse_x'):.4f} "
              f"E={g('energy'):.3f} ret={g('return'):.1f}")
sig = wdf[wdf["significativo"] == "si"]
print(f"\nWilcoxon significativos (Holm<0.05): {len(sig)}/{len(wdf)}", flush=True)
if len(sig):
    print(sig[["scenario", "metric", "variant", "rank_biserial_r", "p_holm", "direccion"]].to_string(index=False))
