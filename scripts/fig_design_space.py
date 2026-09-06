"""Figura 2. Mapa del espacio de diseno del MPC y fallo de las dos guias de sintonia.

Cuatro paneles.
 (a) mapa de calor de la tasa de caida sobre (dt, T_pred), uno por combinacion de
     distribucion de bloques y coste terminal, con las zonas que cada guia
     recomienda marcadas encima.
 (b) interaccion entre horizonte y coste terminal.
 (c) muro de condicionamiento, cond(H) ~ e^(2*lambda*T).
 (d) contraste de las guias, cumplirlas frente a violarlas.

Todo el texto va en INGLES, que es el idioma del manuscrito, y toda cadena con
LaTeX es CRUDA, porque sin la r inicial "$\\tau$" se convierte en un tabulador
literal seguido de "au" y el error solo se ve al mirar la figura de cerca.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
d = pd.concat([pd.read_csv(ROOT / "results" / "processed" / f) for f in
               ("mpc_guidelines_full.csv", "mpc_grid_g1_extra.csv", "mpc_grid_g2_extra.csv")],
              ignore_index=True)
d = d[d.ctrl == "MPC"] if "ctrl" in d.columns else d

TAU, LAM = 0.725, 4.596
G1LO, G1HI, G2MIN = 0.10 * TAU, 0.25 * TAU, 4 * TAU
N_TOT = len(d)
N_SURV = int((d.fell_frac == 0).sum())
N_SURV_REGLA = int(((d.fell_frac == 0) & ((d.G1_ok == 1) | (d.G2_ok == 1))).sum())
CMAP = LinearSegmentedColormap.from_list("fall", ["#2E7D32", "#FFF59D", "#B71C1C"])

fig = plt.figure(figsize=(11.5, 7.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.35)

# --- (a) mapas de calor ---
combos = [("front", "riccati"), ("front", "stage"), ("uniform", "riccati"), ("uniform", "stage")]
dts = sorted(d.dt.unique()); tps = sorted(d.T_pred.unique())
for i, (bl, tm) in enumerate(combos):
    ax = fig.add_subplot(gs[0, i])
    M = np.full((len(tps), len(dts)), np.nan)
    for r in d[(d.blocking == bl) & (d.terminal == tm)].itertuples():
        M[tps.index(r.T_pred), dts.index(r.dt)] = r.fell_frac
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(dts))); ax.set_xticklabels([f"{v:g}" for v in dts], fontsize=7, rotation=45)
    ax.set_yticks(range(len(tps))); ax.set_yticklabels([f"{v:g}" for v in tps], fontsize=7)
    ax.set_title(f"{bl} + {tm}", fontsize=8.5)
    if i == 0:
        ax.set_ylabel(r"$T_{pred}$ [s]", fontsize=8)
    ax.set_xlabel(r"$T_s$ [s]", fontsize=8)
    for yy in range(len(tps)):
        for xx in range(len(dts)):
            if not np.isnan(M[yy, xx]):
                ax.text(xx, yy, f"{M[yy,xx]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[yy, xx] > 0.55 or M[yy, xx] < 0.12 else "black")
    # zonas que cada guia recomienda
    for xx, v in enumerate(dts):
        if G1LO <= v <= G1HI:
            ax.axvline(xx, color="#1565C0", lw=2.2, alpha=0.85)
    for yy, v in enumerate(tps):
        if v >= G2MIN:
            ax.axhline(yy, color="#6A1B9A", lw=2.2, alpha=0.85, ls="--")
cb = fig.colorbar(im, ax=[fig.axes[i] for i in range(4)], fraction=0.02, pad=0.012)
cb.set_label("pendulum fall rate", fontsize=8); cb.ax.tick_params(labelsize=7)

# --- (b) interaccion horizonte x coste terminal ---
ax = fig.add_subplot(gs[1, 0])
s = d[(d.dt <= 0.02) & (d.T_pred <= 1.5)]
for tm, c, mk in (("riccati", "#2E7D32", "o"), ("stage", "#B71C1C", "s")):
    g = s[s.terminal == tm].groupby("T_pred").fell_frac.mean()
    ax.plot(g.index, g.values, marker=mk, color=c, lw=2, label=f"terminal = {tm}")
ax.set_xlabel(r"$T_{pred}$ [s]", fontsize=8.5); ax.set_ylabel("fall rate", fontsize=8.5)
ax.set_title("(b) the best horizon depends on\nthe terminal weight", fontsize=9)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=7.5)

# --- (c) muro de condicionamiento ---
ax = fig.add_subplot(gs[1, 1])
Tg = np.linspace(0.2, 3.2, 100)
ax.semilogy(Tg, np.exp(2 * LAM * Tg) * 1e2, color="#37474F", lw=2, label=r"$\propto e^{2\lambda T}$")
meas = {0.4: 6.519e2, 0.8: 3.616e4, 1.5: 2.539e7, 3.0: 2.745e13}
ax.semilogy(list(meas), list(meas.values()), "o", color="#B71C1C", ms=7, label=r"measured cond($H$)")
ax.axhline(1e12, color="#F57F17", ls=":", lw=2)
ax.text(0.35, 2.5e12, "about three significant digits", fontsize=6.5, color="#F57F17")
ax.set_xlabel(r"$T_{pred}$ [s]", fontsize=8.5); ax.set_ylabel(r"cond($H$)", fontsize=8.5)
ax.set_title("(c) conditioning wall", fontsize=9)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both"); ax.tick_params(labelsize=7.5)

# --- (d) contraste de las guias ---
ax = fig.add_subplot(gs[1, 2])
lbl, val, col = [], [], []
for g, nm in (("G1_ok", "G1\nsampling"), ("G2_ok", "G2\nhorizon")):
    lbl += [f"{nm}\nfollowed", f"{nm}\nviolated"]
    val += [d[d[g] == 1].fell_frac.mean(), d[d[g] == 0].fell_frac.mean()]
    col += ["#B71C1C", "#2E7D32"]
b = ax.bar(range(4), val, color=col, edgecolor="k", lw=0.5)
ax.set_xticks(range(4)); ax.set_xticklabels(lbl, fontsize=6.5)
ax.set_ylabel("mean fall rate", fontsize=8.5); ax.set_ylim(0, 1.1)
ax.set_title("(d) following either rule is worse", fontsize=9)
for r, v in zip(b, val):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
ax.grid(axis="y", alpha=0.3); ax.tick_params(labelsize=7.5)

# --- (e) panel explicativo ---
ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
ax.text(0, 1.0, "The two rules, formalised", fontsize=9, weight="bold", va="top")
ax.text(0, 0.86, rf"G1  $T_s \in [{G1LO:.3f}, {G1HI:.3f}]$ s" "\n"
                 rf"     (10 to 25% of $\tau_{{dom}}$ = {TAU} s)" "\n"
                 r"     blue line in (a)", fontsize=7.2, va="top", color="#1565C0")
ax.text(0, 0.58, rf"G2  $T_{{pred}} \geq {G2MIN:.2f}$ s" "\n"
                 r"     (4$\tau$ of the dominant mode)" "\n"
                 r"     purple line in (a)", fontsize=7.2, va="top", color="#6A1B9A")
ax.text(0, 0.30, f"{N_SURV} of {N_TOT} configurations never fall.\n"
                 f"{'None' if N_SURV_REGLA == 0 else N_SURV_REGLA} of them satisfies G1 or G2.",
        fontsize=7.6, va="top", weight="bold")
ax.text(0, 0.12, "Dominant variable: terminal cost\n(0.05 riccati against 0.37 stage),\n"
                 "absent from both rules.", fontsize=7.2, va="top")

fig.suptitle("Design space of the predictive controller. Both tuning rules fail on an unstable "
             "plant with separated time scales", fontsize=10.5, y=0.985)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig2_design_space.{ext}", bbox_inches="tight", dpi=600)
print("fig2_design_space guardada")
print(f"  configuraciones {N_TOT}, supervivientes {N_SURV}, de ellas cumplen alguna regla {N_SURV_REGLA}")
