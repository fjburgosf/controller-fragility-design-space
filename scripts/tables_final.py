"""Tablas 5 y 6 del manuscrito P2, generadas desde final_summary.csv."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "tables" / "P2"; TAB.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "results" / "processed" / "final_summary.csv")
d["fam"] = d.method.str.replace(r"_s\d+", "", regex=True)


def guarda(df, nombre, pie):
    df.to_csv(TAB / f"{nombre}.csv", index=False)
    (TAB / f"{nombre}.md").write_text(f"**{pie}**\n\n" + df.to_markdown(index=False),
                                      encoding="utf-8")
    (TAB / f"{nombre}.tex").write_text(
        df.to_latex(index=False, escape=False, caption=pie, label=f"tab:{nombre}"),
        encoding="utf-8")
    print(f"  {nombre}  {len(df)} filas")


# ---------- Tabla 5. Comparacion final a traves de la frontera ----------
filas = []
for dv in sorted(d.dv.unique()):
    for fam in ("LQR", "MPC", "SAC", "DDPG"):
        s = d[(d.dv == dv) & (d.fam == fam)]
        if s.empty:
            continue
        filas.append({
            "$d_v$ [V]": f"{dv:.0f}",
            "Regime": "in" if dv <= 6 else "out",
            "Controller": fam,
            "$n$": int(s.n.sum()),
            "Fall rate": f"{s.fell_frac.mean():.2f}",
            "RMSE $\\theta$ [rad]": f"{s.rmse_theta.mean():.3f}",
            "P95 $|x|$ [m]": f"{s.maxx_p95.mean():.3f}",
            "Rail violation": f"{s.rail_viol_frac.mean():.2f}",
            "Energy": f"{s.energy.mean():.0f}",
        })
guarda(pd.DataFrame(filas), "p2_tabla5_comparacion_final",
       "Final comparison of the three controller families across the boundary of the "
       "learned agents training disturbance range, which sits at six volts. Each entry "
       "aggregates 48 paired realisations per controller.")

# ---------- Tabla 6. Coste computacional ----------
def mide(ctrl, n=200):
    import numpy as _np
    x = _np.array([0.0, _np.pi + 0.05, 0.0, 0.0])
    ctrl(0.0, x)                                    # descarta el primer solve
    t0 = time.perf_counter()
    for _ in range(n):
        ctrl(0.0, x)
    return (time.perf_counter() - t0) / n * 1000


import sys
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from igrrl.evaluate_common import make_lqr, make_policy, NOM
from igrrl.mpc_design import DesignMPC
from stable_baselines3 import SAC, DDPG

med = {"LQR": mide(make_lqr()),
       "MPC": mide(lambda t, x: DesignMPC(NOM, dt=0.01, T_pred=0.8, Nc=20,
                                          blocking="front", terminal="riccati").compute(t, x)
                   if False else None) or None}
mpc = DesignMPC(NOM, dt=0.01, T_pred=0.8, Nc=20, blocking="front", terminal="riccati")
med["MPC"] = mide(lambda t, x: mpc.compute(t, x))
for algo, cls in (("SAC", SAC), ("DDPG", DDPG)):
    f = ROOT / "results" / "training" / f"standalone_{algo.lower()}" / "seed_1" / "best_model.zip"
    if f.exists():
        med[algo] = mide(make_policy(cls.load(str(f.with_suffix("")))))

libre = {"LQR": "0", "MPC": "6", "SAC": "hyperparameters and seed",
         "DDPG": "hyperparameters and seed"}
filas6 = [{"Controller": k,
           "Time per control step [ms]": f"{v:.2f}",
           "Relative to LQR": f"{v/med['LQR']:.0f}x",
           "Design variables exposed": libre.get(k, "")}
          for k, v in med.items() if v is not None]
guarda(pd.DataFrame(filas6), "p2_tabla6_coste",
       "Computational cost per control step measured on the evaluation machine, together "
       "with the number of design decisions each family exposes to the engineer.")

print("\nTiempos medidos [ms/paso]")
for k, v in med.items():
    if v is not None:
        print(f"  {k:5} {v:7.3f}")
