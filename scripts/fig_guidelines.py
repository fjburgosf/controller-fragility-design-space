"""Figura 2 de P2. Falsacion de las dos reglas de sintonia sobre 176 configuraciones.

Tres paneles.
 (a) tasa de caida por periodo de muestreo, marcando el rango que la regla
     recomienda, con los cuatro valores medidos dentro de el
 (b) tasa de caida por horizonte, marcando el umbral de la regla y los tres
     horizontes medidos por encima
 (c) muro de condicionamiento medido frente al crecimiento exponencial teorico
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
PR = ROOT / "results" / "processed"

TAU, LAM = 0.725, 4.596
G1LO, G1HI, G2MIN = 0.10 * TAU, 0.25 * TAU, 4 * TAU
ROJO, VERDE, GRIS = "#B71C1C", "#2E7D32", "#455A64"

d = pd.concat([pd.read_csv(PR / f) for f in
               ("mpc_guidelines_full.csv", "mpc_grid_g1_extra.csv", "mpc_grid_g2_extra.csv")],
              ignore_index=True)
d = d[d.ctrl == "MPC"]

fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7))

# ---- (a) muestreo ----
g = d.groupby("dt").agg(cae=("fell_frac", "mean"), n=("fell_frac", "size")).reset_index()
a = ax[0]
a.axvspan(G1LO, G1HI, color=ROJO, alpha=0.13, zorder=0)
a.text(np.sqrt(G1LO * G1HI), 0.45, "range the\nsampling rule\nrecommends",
       ha="center", va="center", fontsize=7.5, color=ROJO, weight="bold")
dentro = (g.dt >= G1LO) & (g.dt <= G1HI)
a.semilogx(g.dt[~dentro], g.cae[~dentro], "o-", color=VERDE, ms=7, lw=1.8, label="outside the range")
a.semilogx(g.dt[dentro], g.cae[dentro], "s", color=ROJO, ms=9, label="inside the range")
# el recuento por punto se indica en el pie, no sobre los datos
a.set_xlabel("sampling period $T_s$ [s]", fontsize=9)
a.set_ylabel("pendulum fall rate", fontsize=9)
a.set_title("(a) every configuration inside the\nrecommended band diverges", fontsize=9)
a.set_ylim(-0.05, 1.22); a.grid(alpha=0.3, which="both")
a.legend(fontsize=8, loc="lower left", framealpha=0.95)
a.tick_params(labelsize=8)

# ---- (b) horizonte ----
h = d[d.dt <= 0.02].groupby(["T_pred", "terminal"]).fell_frac.mean().unstack()
a = ax[1]
a.axvspan(G2MIN, h.index.max() * 1.08, color=ROJO, alpha=0.13, zorder=0)
a.text((G2MIN + h.index.max()) / 2, 0.28, "horizon rule\nsatisfied", ha="center",
       fontsize=7.5, color=ROJO, weight="bold")
for col, c, mk, lab in (("riccati", VERDE, "o", "terminal = Riccati"),
                        ("stage", "#EF6C00", "s", "terminal = stage $Q$")):
    if col in h:
        a.plot(h.index, h[col], marker=mk, color=c, lw=1.8, ms=6, label=lab)
a.set_xlabel("prediction horizon $T_{pred}$ [s]", fontsize=9)
a.set_ylabel("pendulum fall rate", fontsize=9)
a.set_title("(b) the optimal horizon depends on the\nterminal weight, which the rule omits", fontsize=9)
a.set_ylim(-0.05, 1.08); a.grid(alpha=0.3); a.legend(fontsize=7.5); a.tick_params(labelsize=8)

# ---- (c) condicionamiento ----
a = ax[2]
med = {0.4: 6.519e2, 0.8: 3.616e4, 1.5: 2.539e7, 3.0: 2.745e13}
T = np.linspace(0.3, 3.2, 120)
a.semilogy(T, np.exp(2 * LAM * T) * 1.6e2, color=GRIS, lw=1.8,
           label=r"$\propto e^{2\lambda_u T_{pred}}$")
a.semilogy(list(med), list(med.values()), "o", color=ROJO, ms=8, label="measured cond($H$)")
a.axhline(1e12, color="#F57F17", ls=":", lw=1.8)
a.text(0.35, 2.2e12, "about three significant digits", fontsize=6.8, color="#F57F17")
a.set_xlabel("prediction horizon $T_{pred}$ [s]", fontsize=9)
a.set_ylabel("Hessian condition number", fontsize=9)
a.set_title("(c) conditioning wall on an\nunstable plant", fontsize=9)
a.grid(alpha=0.3, which="both"); a.legend(fontsize=7.5, loc="lower right"); a.tick_params(labelsize=8)

fig.suptitle("Falsification of two model predictive control tuning rules over 176 configurations",
             fontsize=10.5, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig1_guidelines.{ext}", bbox_inches="tight", dpi=600)
print("fig1_guidelines guardada")
print(f"  configuraciones {len(d)} | dentro del rango G1 {int(dentro.sum())} valores de dt")
