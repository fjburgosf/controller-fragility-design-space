"""Figuras 2 y 3. Mapa del espacio de diseno del MPC, separado por coste terminal.

Antes era una sola figura con cuatro mapas de calor apretados en una fila y tres
paneles analiticos mas, con lineas azules y moradas dibujadas ENCIMA de las celdas
para senalar lo que cada guia recomienda. Esas lineas tapaban los numeros y la
figura era ilegible. Ahora hay dos figuras, una por coste terminal, con dos mapas
cada una y celdas grandes, y las zonas recomendadas se marcan FUERA del area de
datos, coloreando las etiquetas de los ejes y anadiendo una llave lateral.

La separacion por coste terminal no es cosmetica: es la variable que domina el
resultado, de modo que ver los dos mapas en figuras consecutivas es justamente la
comparacion que el articulo defiende.

Los paneles analiticos que antes acompanaban a los mapas se han retirado porque la
Figura 1 ya los muestra, y repetirlos era redundancia, no evidencia.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "P2"; FIG.mkdir(parents=True, exist_ok=True)
d = pd.concat([pd.read_csv(ROOT / "results" / "processed" / f) for f in
               ("mpc_guidelines_full.csv", "mpc_grid_g1_extra.csv", "mpc_grid_g2_extra.csv")],
              ignore_index=True)
d = d[d.ctrl == "MPC"] if "ctrl" in d.columns else d

TAU = 0.725
G1LO, G1HI, G2MIN = 0.10 * TAU, 0.25 * TAU, 4 * TAU
AZUL, MORADO = "#1565C0", "#6A1B9A"
CMAP = LinearSegmentedColormap.from_list("fall", ["#1B5E20", "#66BB6A", "#FFF176",
                                                  "#EF9A9A", "#B71C1C"])
DTS = sorted(d.dt.unique())
TPS = sorted(d.T_pred.unique())


def mapa(ax, bl, tm):
    """Un mapa de calor. Sin nada dibujado encima de las celdas."""
    M = np.full((len(TPS), len(DTS)), np.nan)
    for r in d[(d.blocking == bl) & (d.terminal == tm)].itertuples():
        M[TPS.index(r.T_pred), DTS.index(r.dt)] = r.fell_frac
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, origin="lower", aspect="auto")
    for y in range(len(TPS)):
        for x in range(len(DTS)):
            v = M[y, x]
            if np.isnan(v):
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#ECEFF1",
                                       edgecolor="white", lw=1.2, hatch="///"))
                continue
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    weight="bold" if v == 0 else "normal",
                    color="white" if v > 0.62 or v < 0.10 else "#212121")
    # rejilla suave entre celdas, sin ocultar el valor
    ax.set_xticks(np.arange(-.5, len(DTS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(TPS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(len(DTS)))
    ax.set_xticklabels([f"{v:g}" for v in DTS], fontsize=9)
    ax.set_yticks(range(len(TPS)))
    ax.set_yticklabels([f"{v:g}" for v in TPS], fontsize=9)
    # lo que cada guia recomienda se marca en las ETIQUETAS, no sobre los datos
    for et, v in zip(ax.get_xticklabels(), DTS):
        if G1LO <= v <= G1HI:
            et.set_color(AZUL); et.set_fontweight("bold")
    for et, v in zip(ax.get_yticklabels(), TPS):
        if v >= G2MIN:
            et.set_color(MORADO); et.set_fontweight("bold")
    ax.set_title(f"{bl} blocking", fontsize=11, pad=8)
    ax.set_xlabel(r"sampling period $T_s$ [s]", fontsize=10)
    return im


def llave(ax, x0, x1, y, color, texto):
    """Llave horizontal bajo el eje, en coordenadas de datos y de ejes."""
    ax.annotate("", xy=(x0 - .45, y), xytext=(x1 + .45, y),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color=color, lw=1.6))
    ax.text((x0 + x1) / 2, y - 0.045, texto, ha="center", va="top", fontsize=8.5,
            color=color, weight="bold", transform=ax.get_xaxis_transform())


def construir(tm, numero, etiqueta):
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 5.4), sharey=True)
    for a, bl in zip(ax, ("front", "uniform")):
        im = mapa(a, bl, tm)
    ax[0].set_ylabel(r"prediction horizon $T_{pred}$ [s]", fontsize=10)

    # llaves con lo que recomienda cada guia, fuera del area de datos
    ig1 = [i for i, v in enumerate(DTS) if G1LO <= v <= G1HI]
    for a in ax:
        llave(a, min(ig1), max(ig1), -0.14, AZUL, "sampling the rule recommends")
    ig2 = [i for i, v in enumerate(TPS) if v >= G2MIN]
    ax[1].annotate("", xy=(1.035, (min(ig2) - .45) / (len(TPS) - 1 + .9) + .02),
                   xytext=(1.035, (max(ig2) + .45) / (len(TPS) - 1 + .9) + .02),
                   xycoords="axes fraction", textcoords="axes fraction",
                   arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4",
                                   color=MORADO, lw=1.6))
    ax[1].text(1.055, 0.80, "horizon\nthe rule\nrecommends", transform=ax[1].transAxes,
               fontsize=8.5, color=MORADO, weight="bold", va="center")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.13)
    cb.set_label("pendulum fall rate", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax[0].text(-0.5, -0.30, "Hatched cells were not part of the grid.",
               transform=ax[0].transAxes, fontsize=8.5, color="#546E7A", va="top")
    n = int((d[d.terminal == tm].fell_frac == 0).sum())
    fig.suptitle(f"Design space with the {etiqueta} terminal weight. "
                 f"{n} of {len(d[d.terminal == tm])} configurations never lose the pendulum",
                 fontsize=12, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig{numero}_map_{tm}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  fig{numero}_map_{tm}: supervivientes {n} de {len(d[d.terminal == tm])}")


construir("riccati", 2, "Riccati")
construir("stage", 3, "stage")
print(f"configuraciones totales {len(d)}, supervivientes {int((d.fell_frac == 0).sum())}, "
      f"de ellas cumplen alguna guia "
      f"{int(((d.fell_frac == 0) & ((d.G1_ok == 1) | (d.G2_ok == 1))).sum())}")
