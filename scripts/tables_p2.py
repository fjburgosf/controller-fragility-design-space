"""Tablas 1 a 4 del manuscrito P2, generadas desde datos reales (CSV, LaTeX y Markdown).

Notas de mantenimiento, porque cada una corresponde a un fallo real ya corregido.

1. Todo el contenido va en INGLES, que es el idioma del manuscrito. Una tabla en
   español dentro de un texto en inglés es un defecto de envío.
2. Las cadenas con LaTeX son CRUDAS. Sin la r inicial, "$\\tau$" se convierte en un
   tabulador literal seguido de "au", y el error es invisible al leer el CSV.
3. La Tabla 2 se calcula sobre el espacio de diseño COMPLETO y DEDUPLICADO que
   entrega igrrl.design_grid, porque su leyenda y el cuerpo del texto hablan de
   ese conjunto. El recuento se toma del propio dato, nunca escrito a mano. Las
   Tablas 3 y 4 restringen a la region de muestreo fino, tal como el texto
   declara, y esa restriccion deja fuera por construccion las tres ampliaciones.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.models.pendulum import equilibrium_up, linearize
from igrrl.mpc_design import Q_P2, R_P2
from igrrl.design_grid import cargar as cargar_espacio

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / "results" / "processed"
TAB = ROOT / "tables" / "P2"
TAB.mkdir(parents=True, exist_ok=True)


def save(df, name, caption):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(f"**{caption}**\n\n" + df.to_markdown(index=False),
                                    encoding="utf-8")
    (TAB / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=False, caption=caption, label=f"tab:{name}"),
        encoding="utf-8")
    print(f"  {name}: {len(df)} filas")


# --- Tabla 1: planta y analisis modal ---
p = load_pendulum_params()
A, B = linearize(equilibrium_up(0.0), 0.0, p)
lqr = LQRController(p, Q=Q_P2.copy(), R=R_P2)
ol = np.linalg.eigvals(A)
cl = np.linalg.eigvals(A - B @ lqr.K.reshape(1, -1))
FperV = p.kt / (p.Rm * p.r)
tau_lento, tau_rapido = 1 / min(abs(cl.real)), 1 / max(abs(cl.real))

t1 = pd.DataFrame([
    {"Quantity": r"Cart mass $M_c$", "Value": f"{p.M_c:.3f}", "Unit": "kg"},
    {"Quantity": r"Pendulum mass $M_p$", "Value": f"{p.Mp:.3f}", "Unit": "kg"},
    {"Quantity": r"Pivot to centre of mass distance $l$", "Value": f"{p.l:.3f}", "Unit": "m"},
    {"Quantity": r"Pendulum inertia $J_p$", "Value": f"{p.Jp:.5f}", "Unit": r"kg m$^2$"},
    {"Quantity": r"Torque constant $k_t$", "Value": f"{p.kt:.4f}", "Unit": "N m/A"},
    {"Quantity": r"Armature resistance $R_m$", "Value": f"{p.Rm:.2f}", "Unit": r"$\Omega$"},
    {"Quantity": r"Pulley radius $r$", "Value": f"{p.r:.5f}", "Unit": "m"},
    {"Quantity": r"Actuator limit", "Value": r"$\pm 12$", "Unit": "V"},
    {"Quantity": r"Force per volt $k_t/(R_m r)$", "Value": f"{FperV:.2f}", "Unit": "N/V"},
    {"Quantity": r"Unstable open loop pole", "Value": f"{max(ol.real):+.3f}", "Unit": "rad/s"},
    {"Quantity": r"Dominant closed loop mode $\tau_{slow}$", "Value": f"{tau_lento:.3f}", "Unit": "s"},
    {"Quantity": r"Fast closed loop mode $\tau_{fast}$", "Value": f"{tau_rapido:.4f}", "Unit": "s"},
    {"Quantity": r"Time scale ratio", "Value": f"{tau_lento/tau_rapido:.0f}", "Unit": "--"},
])
save(t1, "p2_tabla1_planta", "Plant parameters and closed loop modal structure.")

# --- Tabla 2: contraste de las guias sobre el espacio COMPLETO ---
d = cargar_espacio()   # deduplicado y con la rejilla completa
TAU = tau_lento
rows = []
for g, nm, rng in (("G1_ok", "G1 (sampling)", rf"$T_s \in [{0.10*TAU:.3f}, {0.25*TAU:.3f}]$ s"),
                   ("G2_ok", "G2 (horizon)", rf"$T_{{pred}} \geq {4*TAU:.2f}$ s")):
    for val, lab in ((1, "follows"), (0, "violates")):
        s = d[d[g] == val]
        rows.append({"Rule": nm, "Range": rng, "Configurations": lab, r"$n$": len(s),
                     "Mean fall rate": f"{s.fell_frac.mean():.3f}",
                     "Configs with no falls": int((s.fell_frac == 0).sum())})
save(pd.DataFrame(rows), "p2_tabla2_guias",
     f"Contrast of the two tuning rules over the full design space of {len(d)} configurations.")

# --- Tabla 3: jerarquia de variables de diseno, region de muestreo fino ---
s = d[(d.dt <= 0.02) & (d.T_pred <= 1.5)]
rows = []
for var, col in (("Terminal cost", "terminal"), ("Block distribution", "blocking")):
    for lv in sorted(s[col].unique()):
        q = s[s[col] == lv]
        rows.append({"Design variable": var, "Level": lv, r"$n$": len(q),
                     "Fall rate": f"{q.fell_frac.mean():.3f}",
                     r"P95 $|x|$ [m]": f"{q.maxx_p95.mean():.3f}",
                     "Named by the rules": "no"})
save(pd.DataFrame(rows), "p2_tabla3_jerarquia",
     "Ranking of the design variables of the predictive controller. "
     "Neither variable is named by the published rules.")

# --- Tabla 4: interaccion horizonte x coste terminal ---
piv = s.pivot_table(index="T_pred", columns="terminal", values="fell_frac", aggfunc="mean").round(3)
piv = piv.reset_index().rename(columns={"T_pred": r"$T_{pred}$ [s]",
                                        "riccati": "terminal = Riccati",
                                        "stage": r"terminal = stage $Q$"})
save(piv, "p2_tabla4_interaccion",
     "Interaction between prediction horizon and terminal weight. "
     "The two columns move in opposite directions, so the best horizon depends on the terminal choice.")

print(f"\nTablas en {TAB}")
