"""Mini-experimento de diseño para IG-RRL (B9) — docs/IG_RRL_DESIGN.md §9, §11.

Pregunta central: cuando la planta tiene incertidumbre paramétrica (M_c, Mp, l),
¿el NIS del EKF —que corre con parámetros NOMINALES— sube por encima de su valor
nominal lo suficiente para servir de señal de gating?

- Si el NIS separa {nominal} de {Mp ±20%} con un umbral limpio -> el gating por
  NIS tiene piso -> IG-RRL viable.
- Si no separa -> probar eta = traza(P) o entropia(P), o repensar el gate.

Controlador: LQR (B2) + EKF (B4) en realimentacion de salida (identico a B4),
planta no lineal perturbada. NO interviene ningun residuo: solo se mide la senal
de incertidumbre que IG-RRL usaria.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.evaluation.metrics import compute_metrics
from src.models.pendulum import equilibrium_up
from src.simulation.closed_loop import simulate_output_feedback

DT = 0.01
T_FINAL = 6.0
X0 = np.array([0.0, np.pi - 0.3, 0.0, 0.0])
P0 = np.eye(4) * 1e-3
MEAS_NOISE = np.array([0.001, 0.001])  # igual que el benchmark nominal
SEEDS = [0, 1, 2, 3, 4]
SETTLE_T = 1.5  # s; se miden estadisticos del NIS despues de este tiempo
M = 2  # dim(y); NIS ~ chi2(2) bajo modelo nominal, E=2, p95=5.99, p99=9.21

PARAMS = ["M_c", "Mp", "l"]
DELTAS = [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]


def entropy_gauss(P: np.ndarray) -> float:
    n = P.shape[0]
    sign, logdet = np.linalg.slogdet(P)
    return 0.5 * (n * np.log(2 * np.pi * np.e) + logdet)


EKF_R = None   # None = R por defecto del EKF; si no, np.diag(...)
EKF_Q = None


def run_one(plant_p, seed: int):
    """Una simulacion LQR+EKF; devuelve series NIS, traza(P), entropia(P) y metricas."""
    lqr = LQRController(load_pendulum_params())          # disenado en NOMINAL
    kw = {}
    if EKF_R is not None:
        kw["R"] = EKF_R
    if EKF_Q is not None:
        kw["Q"] = EKF_Q
    ekf = ExtendedKalmanFilter(load_pendulum_params(), **kw)   # corre con NOMINAL
    rng = np.random.default_rng(seed)

    nis_series, trP_series, ent_series = [], [], []

    # se envuelve el EKF para capturar sus estadisticos cada paso
    orig_step = ekf.step

    def capturing_step(xhat, P, u, y, dt):
        xhat_n, P_n, K = orig_step(xhat, P, u, y, dt)
        nis_series.append(ekf.last_nis)
        trP_series.append(float(np.trace(P_n)))
        ent_series.append(entropy_gauss(P_n))
        return xhat_n, P_n, K

    ekf.step = capturing_step

    res = simulate_output_feedback(
        X0, X0.copy(), plant_p, DT, T_FINAL, lqr, ekf,
        P0=P0, measurement_noise_std=MEAS_NOISE, rng=rng,
    )
    m = compute_metrics(res.t, res.X_true, res.U, equilibrium_up())
    t = res.t[:-1]
    mask = t >= SETTLE_T
    return {
        "nis": np.array(nis_series),
        "trP": np.array(trP_series),
        "ent": np.array(ent_series),
        "mask": mask,
        "rmse_theta": m["rmse_theta"],
        "stable": m["stable"],
    }


def agg(plant_p):
    runs = [run_one(plant_p, s) for s in SEEDS]
    nis = np.concatenate([r["nis"][r["mask"]] for r in runs])
    trP = np.concatenate([r["trP"][r["mask"]] for r in runs])
    ent = np.concatenate([r["ent"][r["mask"]] for r in runs])
    # eta filtrado (EWMA) por corrida, se toma el maximo sostenido (p95 del filtrado)
    eta_max = []
    for lam in (0.05, 0.1):
        for r in runs:
            e = 2.0
            series = []
            for v in r["nis"]:
                e = (1 - lam) * e + lam * v
                series.append(e)
            series = np.array(series)[r["mask"]]
            eta_max.append(np.percentile(series, 95))
    return {
        "nis_mean": nis.mean(), "nis_p95": np.percentile(nis, 95),
        "trP_mean": trP.mean(), "ent_mean": ent.mean(),
        "eta_p95_filt": np.mean(eta_max),
        "rmse_theta": np.mean([r["rmse_theta"] for r in runs]),
        "stable_pct": 100 * np.mean([r["stable"] for r in runs]),
    }


def main():
    global EKF_R
    import sys
    p = load_pendulum_params()

    if "--recal" in sys.argv:
        # R que coincide con el ruido de medicion real (std=0.001 -> var=1e-6)
        EKF_R = np.diag([1e-6, 1e-6])
        print(">>> EKF recalibrado: R = diag([1e-6, 1e-6]) (coincide con el ruido real)\n")

    print("chi2(2): E=2.00  p90=4.61  p95=5.99  p99=9.21\n")
    base = agg(p)
    print(f"{'case':>16} | {'NIS mean':>8} {'NIS p95':>8} {'eta_filt p95':>12} "
          f"{'trace P':>10} {'entropy':>9} | {'RMSEθ':>7} {'stable%':>7}")
    print("-" * 100)
    print(f"{'NOMINAL':>16} | {base['nis_mean']:>8.2f} {base['nis_p95']:>8.2f} "
          f"{base['eta_p95_filt']:>12.2f} {base['trP_mean']:>10.2e} "
          f"{base['ent_mean']:>9.2f} | {base['rmse_theta']:>7.4f} {base['stable_pct']:>6.0f}%")
    print()

    rows = {}
    for name in PARAMS:
        for d in DELTAS:
            pert = replace(p, **{name: getattr(p, name) * (1.0 + d)})
            r = agg(pert)
            rows[(name, d)] = r
            tag = f"{name} {d:+.0%}"
            print(f"{tag:>16} | {r['nis_mean']:>8.2f} {r['nis_p95']:>8.2f} "
                  f"{r['eta_p95_filt']:>12.2f} {r['trP_mean']:>10.2e} "
                  f"{r['ent_mean']:>9.2f} | {r['rmse_theta']:>7.4f} {r['stable_pct']:>6.0f}%")
        print()

    # vertices combinados duros (Mp-, Mc-, l-) y (Mp-, Mc+, l-)
    for signs in [(-1, -1, -1), (-1, +1, -1)]:
        pert = replace(p, **{n: getattr(p, n) * (1 + s * 0.20) for n, s in zip(PARAMS, signs)})
        r = agg(pert)
        tag = "vtx " + ",".join(f"{n}{'+' if s > 0 else '-'}" for n, s in zip(PARAMS, signs))
        print(f"{tag:>16} | {r['nis_mean']:>8.2f} {r['nis_p95']:>8.2f} "
              f"{r['eta_p95_filt']:>12.2f} {r['trP_mean']:>10.2e} "
              f"{r['ent_mean']:>9.2f} | {r['rmse_theta']:>7.4f} {r['stable_pct']:>6.0f}%")

    # veredicto: ratio NIS medio de los casos Mp+-20% vs nominal
    print("\n" + "=" * 100)
    print("VEREDICTO")
    for name in PARAMS:
        worst = max(abs(rows[(name, d)]["nis_mean"] - base["nis_mean"]) for d in DELTAS)
        ratio = max(rows[(name, d)]["nis_mean"] for d in DELTAS) / base["nis_mean"]
        print(f"  {name}: NIS medio max/nominal = {ratio:.1f}x   (delta abs max = {worst:.2f})")
    sep_ok = (min(rows[("Mp", d)]["nis_mean"] for d in (-0.20, 0.20))
              > base["nis_p95"])
    print(f"\n  Mp +-20% separa del p95 nominal ({base['nis_p95']:.2f})? "
          f"{'SI -> gating por NIS viable' if sep_ok else 'NO -> revisar eta alternativo'}")


if __name__ == "__main__":
    main()
