"""Métricas físicas obligatorias por controlador.

Todas las funciones operan sobre una trayectoria ya simulada
(t, X, U) devuelta por `src.models.pendulum.simulate` o
`src.simulation.closed_loop.simulate_output_feedback` (usar `.X_true`).
No fabrica ninguna métrica: todo se calcula directamente de los arrays.
"""
from __future__ import annotations

import numpy as np


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_metrics(
    t: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    x_eq: np.ndarray,
    u_max: float = 12.0,
    settle_band: float = 0.02,
    wall_clock_seconds: float | None = None,
    x_settle_abs: float = 0.01,
) -> dict:
    """x_eq = [x_ref, theta_ref, 0, 0] (equilibrio invertido, típicamente
    theta_ref=pi). `wall_clock_seconds`: tiempo real de cómputo de la
    simulación, si se midió (métrica de "Computación").
    `x_settle_abs`: banda absoluta en metros para el establecimiento de la
    posición del carro (D16); 0.01 m equivale a 1 cm."""
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    x_err = X[:, 0] - x_eq[0]
    theta_err = _wrap(X[:, 1] - x_eq[1])

    metrics = {}

    # --- Error ---
    metrics["rmse_x"] = float(np.sqrt(np.mean(x_err**2)))
    metrics["rmse_theta"] = float(np.sqrt(np.mean(theta_err**2)))
    metrics["mae_x"] = float(np.mean(np.abs(x_err)))
    metrics["mae_theta"] = float(np.mean(np.abs(theta_err)))
    metrics["max_error_x"] = float(np.max(np.abs(x_err)))
    metrics["max_error_theta"] = float(np.max(np.abs(theta_err)))
    n_tail = max(1, len(t) // 10)  # último 10% de la simulación
    metrics["steady_state_error_x"] = float(np.mean(np.abs(x_err[-n_tail:])))
    metrics["steady_state_error_theta"] = float(np.mean(np.abs(theta_err[-n_tail:])))

    # --- Respuesta temporal (sobre theta_err, la variable de control primaria) ---
    e0 = theta_err[0]
    if abs(e0) > 1e-9:
        target_10 = 0.9 * e0
        target_90 = 0.1 * e0
        # cruces hacia 0 desde e0 (rise time 10%->90% del camino recorrido)
        crossed_10 = np.where(np.abs(theta_err) <= np.abs(target_10))[0]
        crossed_90 = np.where(np.abs(theta_err) <= np.abs(target_90))[0]
        t10 = t[crossed_10[0]] if len(crossed_10) else np.nan
        t90 = t[crossed_90[0]] if len(crossed_90) else np.nan
        metrics["rise_time"] = float(t90 - t10) if np.isfinite(t90) and np.isfinite(t10) else np.nan
    else:
        metrics["rise_time"] = 0.0

    settle_thresh = settle_band * max(abs(e0), 1e-9)
    outside = np.where(np.abs(theta_err) > settle_thresh)[0]
    metrics["settling_time"] = float(t[outside[-1]]) if len(outside) else 0.0

    # Establecimiento de la POSICION del carro (D16). Las metricas temporales
    # historicas se calculan solo sobre theta, lo que oculta que un controlador
    # puede haber verticalizado el pendulo y seguir con el carro en transitorio.
    # Se usa banda relativa a la excursion maxima de x y tambien banda absoluta,
    # porque x arranca en la referencia y una banda relativa a x[0] no aplica.
    x_span = float(np.max(np.abs(x_err)))
    outside_x = np.where(np.abs(x_err) > settle_band * max(x_span, 1e-9))[0]
    metrics["settling_time_x"] = float(t[outside_x[-1]]) if len(outside_x) else 0.0
    outside_x_abs = np.where(np.abs(x_err) > x_settle_abs)[0]
    metrics["settling_time_x_abs"] = (
        float(t[outside_x_abs[-1]]) if len(outside_x_abs) else 0.0
    )
    # True solo si x se establecio DENTRO del horizonte simulado
    metrics["settled_x_in_horizon"] = bool(
        len(outside_x_abs) == 0 or t[outside_x_abs[-1]] < t[-1] - 1e-9
    )

    peak_idx = int(np.argmax(np.abs(theta_err)))
    metrics["peak_time"] = float(t[peak_idx])
    peak_val = theta_err[peak_idx]
    if abs(e0) > 1e-9 and np.sign(peak_val) == np.sign(e0):
        metrics["overshoot_pct"] = 0.0
        metrics["undershoot_pct"] = float(max(0.0, (abs(peak_val) - abs(e0)) / abs(e0) * 100))
    elif abs(e0) > 1e-9:
        metrics["overshoot_pct"] = float(abs(peak_val) / abs(e0) * 100)
        metrics["undershoot_pct"] = 0.0
    else:
        metrics["overshoot_pct"] = 0.0
        metrics["undershoot_pct"] = 0.0

    # --- Control ---
    metrics["iae"] = float(np.trapezoid(np.abs(theta_err), t))
    metrics["ise"] = float(np.trapezoid(theta_err**2, t))
    metrics["control_effort_abs"] = float(np.trapezoid(np.abs(U), t[: len(U)]))
    metrics["control_rms"] = float(np.sqrt(np.mean(U**2)))
    # Con una entrada de tensión, integral(u^2 dt) es esfuerzo cuadrático
    # [V^2 s], no energía física en julios. Se conserva la clave histórica
    # ``control_energy`` para compatibilidad y se expone el nombre correcto.
    quadratic_effort = float(np.trapezoid(U**2, t[: len(U)]))
    metrics["quadratic_control_effort"] = quadratic_effort
    metrics["control_energy"] = quadratic_effort
    metrics["max_u"] = float(np.max(np.abs(U)))
    metrics["saturation_pct"] = float(np.mean(np.abs(U) >= u_max - 1e-6) * 100)

    # --- Criterio de regulación en horizonte finito ---
    # Una simulación finita no demuestra estabilidad asintótica. El indicador
    # exige que las dos salidas y las dos velocidades permanezcan próximas al
    # equilibrio durante el 10 % final del ensayo.
    finite = bool(np.all(np.isfinite(X)))
    tail = X[-n_tail:]
    mean_abs_xdot = float(np.mean(np.abs(tail[:, 2] - x_eq[2])))
    mean_abs_thetadot = float(np.mean(np.abs(tail[:, 3] - x_eq[3])))
    success = bool(
        finite
        and metrics["steady_state_error_theta"] < 0.05
        and metrics["steady_state_error_x"] < 0.02
        and mean_abs_xdot < 0.05
        and mean_abs_thetadot < 0.10
    )
    metrics["tail_mean_abs_xdot"] = mean_abs_xdot
    metrics["tail_mean_abs_thetadot"] = mean_abs_thetadot
    metrics["success"] = success
    metrics["stable"] = success  # alias conservado para scripts previos

    # --- Computación ---
    if wall_clock_seconds is not None:
        metrics["wall_clock_total_s"] = float(wall_clock_seconds)
        metrics["wall_clock_per_step_s"] = float(wall_clock_seconds / max(1, len(U)))

    return metrics
