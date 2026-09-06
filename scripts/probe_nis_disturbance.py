"""Sonda 2 de diseno para B9 (docs/DECISIONS.md D15, opcion C).

La sonda anterior (probe_nis_gating.py) mostro que el NIS del EKF NO responde a
la incertidumbre parametrica cerca del equilibrio. Esta sonda comprueba el otro
regimen: ¿el NIS SI salta ante PERTURBACIONES (impulso, sostenida, ruido de
proceso)? Si si, B9 puede usar el NIS como disparador de robustificacion frente
a disturbios (y la robustez parametrica se cubre por otra via).

Montaje: LQR + EKF en realimentacion de salida. La planta recibe u_aplicado =
u_controlador + d(t); el EKF y el controlador solo ven u_controlador. Asi el
disturbio es genuinamente "no modelado".
"""
from __future__ import annotations

import numpy as np

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.estimators.ekf import ExtendedKalmanFilter
from src.models.pendulum import equilibrium_up, rk4_step

DT = 0.01
T_FINAL = 6.0
X0 = np.array([0.0, np.pi - 0.3, 0.0, 0.0])
P0 = np.eye(4) * 1e-3
MEAS_NOISE = np.array([0.001, 0.001])
EKF_R = np.diag([1e-6, 1e-6])   # recalibrado (coincide con el ruido real)
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
SEEDS = [0, 1, 2, 3, 4]
NIS_NOMINAL_P95 = 6.31  # de probe_nis_gating.py --recal


def dist_impulse(amp, t0=2.5, dur=0.1):
    return lambda t: amp if t0 <= t < t0 + dur else 0.0


def dist_sustained(amp, t0=2.0):
    return lambda t: amp if t >= t0 else 0.0


def dist_procnoise(std, rng):
    return lambda t: rng.normal(0.0, std)


def run(scenario, seed, plant_p=None):
    p = plant_p if plant_p is not None else load_pendulum_params()
    lqr = LQRController(load_pendulum_params())
    ekf = ExtendedKalmanFilter(load_pendulum_params(), R=EKF_R)
    rng = np.random.default_rng(seed)

    d_fn = {
        "nominal": lambda t: 0.0,
        "impulse_8V_0.1s": dist_impulse(8.0),
        "impulse_12V_0.1s": dist_impulse(12.0),
        "impulse_5V_0.05s": dist_impulse(5.0, dur=0.05),
        "sustained_3V": dist_sustained(3.0),
        "sustained_1V": dist_sustained(1.0),
        "procnoise_2V": dist_procnoise(2.0, rng),
        "procnoise_0.5V": dist_procnoise(0.5, rng),
    }[scenario]

    n = int(round(T_FINAL / DT))
    x = X0.copy()
    xhat = X0.copy()
    P = P0.copy()
    nis_series = []
    theta_max = 0.0
    for k in range(n):
        t = k * DT
        u_ctrl = float(np.clip(lqr(t, xhat), -12.0, 12.0))
        y = H @ x + rng.normal(0.0, MEAS_NOISE)
        xhat, P, _ = ekf.step(xhat, P, u_ctrl, y, DT)
        nis_series.append(ekf.last_nis)
        u_plant = float(np.clip(u_ctrl + d_fn(t), -12.0, 12.0))
        x = rk4_step(x, u_plant, p, DT)
        theta_max = max(theta_max, abs(np.arctan2(np.sin(x[1] - np.pi), np.cos(x[1] - np.pi))))
    return np.array(nis_series), theta_max, np.all(np.isfinite(x))


def eta_filt(nis, lam=0.1):
    e = 2.0
    out = []
    for v in nis:
        e = (1 - lam) * e + lam * v
        out.append(e)
    return np.array(out)


def agg(scenario, plant_p=None):
    pre, win, etawin, t_above, tmaxs, oks = [], [], [], [], [], []
    THR = 10.0  # umbral candidato para el gate (NIS filtrado)
    for s in SEEDS:
        nis, tmax, ok = run(scenario, s, plant_p)
        t = np.arange(len(nis)) * DT
        eta = eta_filt(nis)
        pre.append(nis[(t >= 1.5) & (t < 2.0)].mean())     # settled, antes del disturbio
        win.append(nis[(t >= 2.0) & (t <= 3.5)].mean())    # ventana del disturbio
        etawin.append(eta[(t >= 2.0) & (t <= 3.5)].max())
        t_above.append(DT * np.sum(eta[t >= 2.0] > THR))   # s con eta_filt > THR tras t=2
        tmaxs.append(tmax)
        oks.append(ok)
    return dict(
        nis_pre=np.mean(pre), nis_win=np.mean(win), eta_win=np.mean(etawin),
        t_above=np.mean(t_above), theta_max=np.mean(tmaxs), stable=100 * np.mean(oks),
    )


def main():
    print("EKF recalibrado. NIS settled nominal ~ 2 (chi2(2): E=2, p95=5.99). "
          "Gate candidato: eta_filt (EWMA lam=0.1) > 10\n")
    print(f"{'scenario':>20} | {'NIS pre(1.5-2s)':>15} {'NIS win(2-3.5s)':>15} {'eta_filt win max':>16} "
          f"{'s con eta>10':>12} | {'|theta|max':>10} {'finite%':>8}")
    print("-" * 116)
    scenarios = ["nominal", "impulse_5V_0.05s", "impulse_8V_0.1s", "impulse_12V_0.1s",
                 "sustained_1V", "sustained_3V", "procnoise_0.5V", "procnoise_2V"]
    for sc in scenarios:
        r = agg(sc)
        print(f"{sc:>20} | {r['nis_pre']:>15.2f} {r['nis_win']:>15.2f} {r['eta_win']:>16.2f} "
              f"{r['t_above']:>12.2f} | {r['theta_max']:>10.4f} {r['stable']:>7.0f}%")

    from dataclasses import replace
    p = load_pendulum_params()
    rp = agg("nominal", replace(p, Mp=p.Mp * 0.8))
    print(f"{'[ref] Mp-20% param':>20} | {rp['nis_pre']:>15.2f} {rp['nis_win']:>15.2f} "
          f"{rp['eta_win']:>16.2f} {rp['t_above']:>12.2f} | {rp['theta_max']:>10.4f} {rp['stable']:>7.0f}%")

    print("\n" + "=" * 116)
    print("VEREDICTO: si 'NIS win' >> 'NIS pre' para los disturbios y 'Mp-20% param'")
    print("queda en ~2 como nominal -> el NIS es buen disparador de PERTURBACION,")
    print("no de incertidumbre parametrica (opcion C de D15 viable).")


if __name__ == "__main__":
    main()
