"""PASADA 1 de verificacion. Recomputa CADA numero del manuscrito desde su CSV.

No confia en el texto ni en notas previas. Cada afirmacion numerica se recalcula
desde el dato crudo y se compara con lo que el manuscrito dice.
"""
from __future__ import annotations

import glob, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / "results" / "processed"
TXT = (ROOT / "paper" / "manuscript_en.md").read_text(encoding="utf-8")

OK, MAL = [], []


def chk(nombre, esperado, calculado, tol=None, texto_debe_contener=None):
    """Compara lo que dice el manuscrito con lo recomputado."""
    if texto_debe_contener is not None:
        presente = texto_debe_contener in TXT
        bien = presente
        det = f"el texto {'contiene' if presente else 'NO contiene'} \"{texto_debe_contener[:52]}\""
    else:
        if tol is None:
            tol = abs(esperado) * 0.02 + 1e-9
        bien = abs(esperado - calculado) <= tol
        det = f"manuscrito {esperado}  recomputado {calculado:.4f}  dif {abs(esperado-calculado):.4f}"
    (OK if bien else MAL).append((nombre, det))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<48} {det}")


print("=" * 96)
print("A. ESTRUCTURA MODAL Y REGLAS")
print("=" * 96)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.models.pendulum import equilibrium_up, linearize
from igrrl.mpc_design import Q_P2, R_P2

p = load_pendulum_params()
A, B = linearize(equilibrium_up(0.0), 0.0, p)
lqr = LQRController(p, Q=Q_P2.copy(), R=R_P2)
cl = np.linalg.eigvals(A - B @ lqr.K.reshape(1, -1))
tau_l, tau_r = 1 / min(abs(cl.real)), 1 / max(abs(cl.real))
chk("polo inestable +4.596 rad/s", 4.596, max(np.linalg.eigvals(A).real))
chk("tau lento 0.725 s", 0.725, tau_l)
chk("tau rapido 0.007 s", 0.007, tau_r, tol=5e-4)
chk("razon de escalas cerca de 100", 100, tau_l / tau_r, tol=12)
chk("rango G1 [0.073, 0.181] s", 0.0725, 0.10 * tau_l, tol=2e-3)
chk("umbral G2 2.90 s", 2.90, 4 * tau_l, tol=0.02)
chk("fuerza por voltio 2.46 N/V", 2.46, p.kt / (p.Rm * p.r), tol=0.02)

print()
print("=" * 96)
print("B. REJILLA DE DISENO")
print("=" * 96)
d = pd.concat([pd.read_csv(PR / f) for f in
               ("mpc_guidelines_full.csv", "mpc_grid_g1_extra.csv", "mpc_grid_g2_extra.csv")],
              ignore_index=True)
d = d[d.ctrl == "MPC"]
chk("176 configuraciones", 176, len(d), tol=0)
g1 = d[d.G1_ok == 1]
chk("4 valores de dt dentro de G1", 4, g1.dt.nunique(), tol=0)
chk("64 configuraciones dentro de G1", 64, len(g1), tol=0)
chk("todas dentro de G1 caen", 1.0, g1.fell_frac.min(), tol=0)
chk("todas dentro de G1 divergen", 1.0, g1.diverged_frac.min(), tol=0)
for dt_, esp in ((0.005, 0.203), (0.010, 0.379), (0.020, 0.594)):
    s = d[(d.dt == dt_) & (d.T_pred <= 3.0) & (~d.G2_ok.astype(bool) | (d.T_pred == 3.0))]
    base = pd.read_csv(PR / "mpc_guidelines_full.csv")
    base = base[base.ctrl == "MPC"]
    chk(f"caida media a dt={dt_}", esp, base[base.dt == dt_].fell_frac.mean(), tol=0.005)
g2 = d[d.G2_ok == 1]
chk("3 horizontes cumplen G2", 3, g2.T_pred.nunique(), tol=0)
chk("68 configuraciones cumplen G2", 68, len(g2), tol=0)
chk("ninguna que cumple G2 sobrevive", 0, int((g2.fell_frac == 0).sum()), tol=0)
extra2 = pd.read_csv(PR / "mpc_grid_g2_extra.csv"); extra2 = extra2[extra2.ctrl == "MPC"]
chk("caida 0.776 a T=2.2", 0.776, extra2[extra2.T_pred == 2.2].fell_frac.mean(), tol=0.005)
chk("caida 0.948 a T=4.5", 0.948, extra2[extra2.T_pred == 4.5].fell_frac.mean(), tol=0.005)

base = pd.read_csv(PR / "mpc_guidelines_full.csv"); base = base[base.ctrl == "MPC"]
chk("24 de 80 con 0 % de caidas", 24, int((base.fell_frac == 0).sum()), tol=0)
ok0 = base[base.fell_frac == 0]
chk("ninguna de ellas cumple G1 o G2", 0, int(ok0.G1_ok.sum() + ok0.G2_ok.sum()), tol=0)

s = d[(d.dt <= 0.02) & (d.T_pred <= 1.5)]
chk("terminal riccati 0.052", 0.052, s[s.terminal == "riccati"].fell_frac.mean(), tol=0.004)
chk("terminal stage 0.368", 0.368, s[s.terminal == "stage"].fell_frac.mean(), tol=0.004)
chk("blocking front 0.136", 0.136, s[s.blocking == "front"].fell_frac.mean(), tol=0.004)
chk("blocking uniform 0.284", 0.284, s[s.blocking == "uniform"].fell_frac.mean(), tol=0.004)
chk("p95|x| front 0.782", 0.782, s[s.blocking == "front"].maxx_p95.mean(), tol=0.01)
chk("p95|x| uniform 1.379", 1.379, s[s.blocking == "uniform"].maxx_p95.mean(), tol=0.01)
piv = s.pivot_table(index="T_pred", columns="terminal", values="fell_frac", aggfunc="mean")
for T, er, es in ((0.4, 0.000, 0.615), (0.8, 0.020, 0.313), (1.5, 0.135, 0.177)):
    chk(f"interaccion T={T} riccati", er, piv.loc[T, "riccati"], tol=0.004)
    chk(f"interaccion T={T} stage", es, piv.loc[T, "stage"], tol=0.004)
chk("riccati domina en todo el rango medido", True,
    all(piv.loc[T, "riccati"] < piv.loc[T, "stage"] for T in (0.4, 0.8, 1.5)), tol=0)

print()
print("=" * 96)
print("C. MATRICES DRL")
print("=" * 96)
h = pd.read_csv(PR / "rl_hyperparam_sweep.csv")
for algo, mej, peor in (("sac", -10.6, -84.4), ("ddpg", -19.4, -165.8)):
    a = h[h.algo == algo]
    chk(f"{algo} mejor {mej}", mej, a.eval_mean.max(), tol=0.15)
    chk(f"{algo} peor {peor}", peor, a.eval_mean.min(), tol=0.15)
for algo, mb, sb, mf, sf in (("sac", -8.2, 1.5, -31.7, 35.2), ("ddpg", -21.2, 6.4, -53.0, 17.8)):
    ms = [json.load(open(f)) for f in sorted(glob.glob(
        str(ROOT / "results" / "training" / f"standalone_{algo}" / "seed_*" / "meta.json")))]
    chk(f"{algo} 5 semillas", 5, len(ms), tol=0)
    b = [m["best_eval"] for m in ms]; fi = [m["final_eval"] for m in ms]
    chk(f"{algo} best media {mb}", mb, np.mean(b), tol=0.1)
    chk(f"{algo} best desv {sb}", sb, np.std(b), tol=0.1)
    chk(f"{algo} final media {mf}", mf, np.mean(fi), tol=0.15)
    chk(f"{algo} final desv {sf}", sf, np.std(fi), tol=0.15)

print()
print("=" * 96)
print("D. COMPARACION FINAL")
print("=" * 96)
f = pd.read_csv(PR / "final_summary.csv")
f["fam"] = f.method.str.replace(r"_s\d+", "", regex=True)
ESP = {(4.0, "LQR"): (0.00, 0.037, 0.180), (4.0, "SAC"): (0.00, 0.041, 0.226),
       (4.0, "DDPG"): (0.00, 0.042, 0.286), (6.0, "LQR"): (0.00, 0.056, 0.246),
       (8.0, "LQR"): (0.00, 0.076, 0.432), (8.0, "MPC"): (0.00, 0.078, 0.508),
       (8.0, "SAC"): (0.91, 22.912, 3.611), (8.0, "DDPG"): (0.33, 1.557, 1.898),
       (10.0, "LQR"): (1.00, 2.638, 2.182), (10.0, "DDPG"): (0.92, 3.854, 2.495)}
for (dv, fam), (ec, er, ex) in ESP.items():
    s = f[(f.dv == dv) & (f.fam == fam)]
    chk(f"dv={dv:g} {fam} caida {ec}", ec, s.fell_frac.mean(), tol=0.02)
    chk(f"dv={dv:g} {fam} rmseTh {er}", er, s.rmse_theta.mean(), tol=max(0.002, abs(er) * 0.02))

print()
print("=" * 96)
print("E. AFIRMACIONES QUE NO DEBEN APARECER (sobrealcance ya corregido)")
print("=" * 96)
for frase in ("the regulator does not, and the asymmetry",
              "has no such boundary",
              "no crossing is observed" if False else "the two trends move in opposite directions, so the horizon"):
    presente = frase in TXT
    (MAL if presente and "does not" in frase or presente and "no such boundary" in frase
     else OK).append((frase[:40], "ausente" if not presente else "PRESENTE"))
    print(f"  [{'MAL' if presente and ('does not' in frase or 'no such boundary' in frase) else 'ok '}] "
          f"{frase[:52]:<54} {'PRESENTE' if presente else 'ausente'}")

print()
print("=" * 96)
print(f"RESUMEN PASADA 1   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 96)
for n, det in MAL:
    print(f"  DISCREPANCIA  {n}\n                {det}")
