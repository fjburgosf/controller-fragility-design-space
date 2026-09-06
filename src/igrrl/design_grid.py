"""Carga unica del espacio de diseno del MPC, deduplicada.

El barrido se ejecuto en cuatro tandas y dos de ellas se solapan: el horizonte de
3.0 s con muestreo fino aparece tanto en la rejilla primaria como en la extension
construida para sondear la guia de horizonte. Concatenar los archivos sin mas
contaba doce configuraciones dos veces, de modo que el total parecia 176 cuando
las configuraciones distintas eran 164.

Las copias dan resultados identicos, cosa que este modulo comprueba en vez de
suponerla, asi que el solape sirve como prueba de reproducibilidad entre tandas.

Con la cuarta tanda, que cubre muestreo grueso con horizonte largo, la rejilla
queda completa: 8 muestreos x 7 horizontes x 2 distribuciones de bloques x 2
costes terminales = 224 configuraciones, sin celdas ausentes.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESADO = ROOT / "results" / "processed"

ARCHIVOS = (
    "mpc_guidelines_full.csv",      # rejilla primaria
    "mpc_grid_g1_extra.csv",        # extension de muestreo, para la guia G1
    "mpc_grid_g2_extra.csv",        # extension de horizonte, para la guia G2
    "mpc_grid_g3_coarse_long.csv",  # muestreo grueso con horizonte largo
)
CLAVE = ["dt", "T_pred", "blocking", "terminal"]


def cargar(verificar: bool = True) -> pd.DataFrame:
    """Devuelve el espacio de diseno con una fila por configuracion distinta."""
    d = pd.concat([pd.read_csv(PROCESADO / f) for f in ARCHIVOS], ignore_index=True)
    d = d[d.ctrl == "MPC"] if "ctrl" in d.columns else d

    if verificar:
        # las configuraciones repetidas entre tandas deben coincidir
        rep = d[d.duplicated(CLAVE, keep=False)]
        discrepa = rep.groupby(CLAVE).fell_frac.nunique()
        if (discrepa > 1).any():
            malas = discrepa[discrepa > 1].index.tolist()
            raise ValueError(f"tandas en desacuerdo en {len(malas)} configuraciones: {malas[:4]}")

    u = d.drop_duplicates(CLAVE).reset_index(drop=True)

    if verificar:
        dts, tps = sorted(u.dt.unique()), sorted(u.T_pred.unique())
        hechas = {tuple(r) for r in u[CLAVE].itertuples(index=False)}
        faltan = [c for c in itertools.product(dts, tps, ("front", "uniform"),
                                               ("riccati", "stage")) if c not in hechas]
        if faltan:
            raise ValueError(f"la rejilla tiene {len(faltan)} celdas sin ejecutar, "
                             f"por ejemplo {faltan[:3]}")
    return u


def resumen() -> dict:
    """Cifras del espacio de diseno que el manuscrito cita."""
    u = cargar()
    s = u[u.fell_frac == 0]
    return {
        "configuraciones": len(u),
        "muestreos": u.dt.nunique(),
        "horizontes": u.T_pred.nunique(),
        "G1_cumple": int((u.G1_ok == 1).sum()),
        "G2_cumple": int((u.G2_ok == 1).sum()),
        "G1_todas_caen": bool(u[u.G1_ok == 1].fell_frac.min() == 1.0),
        "G1_todas_divergen": bool(u[u.G1_ok == 1].diverged_frac.min() == 1.0),
        "G2_supervivientes": int((u[u.G2_ok == 1].fell_frac == 0).sum()),
        "supervivientes": len(s),
        "supervivientes_con_guia": int(((s.G1_ok == 1) | (s.G2_ok == 1)).sum()),
        "supervivientes_riccati": int((s.terminal == "riccati").sum()),
        "supervivientes_stage": int((s.terminal == "stage").sum()),
    }


if __name__ == "__main__":
    for k, v in resumen().items():
        print(f"  {k:26} {v}")
