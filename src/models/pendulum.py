"""Modelo dinámico del péndulo invertido sobre carrito.

Reconstrucción fiel (misma física, sin alterar signos ni ecuaciones) de
`PendulumNonlinear.m`, auditado en AUDIT_REPORT.md §3.6 (idéntico en las
4 ubicaciones donde aparece en el material fuente). La definición formal
del modelo, la convención angular y la justificación de cada parámetro
están en docs/CANONICAL_MODEL.md — cualquier cambio a las ecuaciones debe
reflejarse en ambos lugares.

Convención: theta = 0 -> péndulo abajo (estable). theta = pi -> péndulo
arriba (inestable, punto de operación de control).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from src.config import PendulumParams

State = np.ndarray  # [x, theta, xdot, thetadot]


def dynamics(x: State, u: float, p: PendulumParams) -> np.ndarray:
    """xdot = f(x, u) — ecuaciones no lineales de PendulumNonlinear.m."""
    x1, x2, x3, x4 = x

    Mp, Mt, Jp, l, g = p.Mp, p.Mt, p.Jp, p.l, p.g
    c, b = p.c, p.b
    kt, Ra, r = p.kt, p.Ra, p.r
    alpha = p.alpha

    sin2, cos2 = np.sin(x2), np.cos(x2)

    Delta = alpha + Mp**2 * l**2 * sin2**2
    motor_force = kt * u / (Ra * r)
    back_emf = kt**2 * x3 / (Ra * r**2)

    xdot1 = x3
    xdot2 = x4

    xdot3 = (
        Mp * l * b * x4 * cos2
        + Mp**2 * l**2 * g * sin2 * cos2
        + (Jp + Mp * l**2) * (motor_force - back_emf - c * x3 + Mp * l * x4**2 * sin2)
    ) / Delta

    xdot4 = (
        -motor_force * Mp * l * cos2
        + back_emf * Mp * l * cos2
        + c * Mp * l * x3 * cos2
        - Mp**2 * l**2 * x4**2 * sin2 * cos2
        + (Mt + Mp) * (-b * x4 - Mp * g * l * sin2)
    ) / Delta

    return np.array([xdot1, xdot2, xdot3, xdot4])


def rk4_step(x: State, u: float, p: PendulumParams, dt: float) -> np.ndarray:
    """Un paso de integración RK4 de paso fijo."""
    k1 = dynamics(x, u, p)
    k2 = dynamics(x + dt / 2 * k1, u, p)
    k3 = dynamics(x + dt / 2 * k2, u, p)
    k4 = dynamics(x + dt * k3, u, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    x0: State,
    p: PendulumParams,
    dt: float,
    t_final: float,
    controller=None,
    u_sequence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula la planta no lineal con RK4.

    Exactamente uno de `controller` (callable(t, x) -> u) o `u_sequence`
    (array de longitud N) debe proporcionarse.
    """
    if (controller is None) == (u_sequence is None):
        raise ValueError("Proveer exactamente uno de: controller, u_sequence.")

    n_steps = int(round(t_final / dt))
    t = np.arange(n_steps + 1) * dt
    X = np.zeros((n_steps + 1, 4))
    U = np.zeros(n_steps)
    X[0] = x0

    x = np.array(x0, dtype=float)
    for k in range(n_steps):
        u = controller(t[k], x) if controller is not None else u_sequence[k]
        u = float(np.clip(u, -12.0, 12.0))
        U[k] = u
        x = rk4_step(x, u, p, dt)
        X[k + 1] = x

    return t, X, U


def equilibrium_up(x_ref: float = 0.0) -> np.ndarray:
    """Punto de operación de control: péndulo arriba, carro en x_ref."""
    return np.array([x_ref, np.pi, 0.0, 0.0])


def equilibrium_down(x_ref: float = 0.0) -> np.ndarray:
    """Equilibrio estable en lazo abierto: péndulo abajo."""
    return np.array([x_ref, 0.0, 0.0, 0.0])


def linearize(x_eq: State, u_eq: float, p: PendulumParams) -> tuple[np.ndarray, np.ndarray]:
    """Linealización simbólica exacta de las ecuaciones no lineales en (x_eq, u_eq).

    Replica el procedimiento de GenerateJacobians.m (Jacobianos simbólicos
    A = df/dx, B = df/du), evaluados numéricamente en el punto de
    operación dado. Por defecto se usa para linealizar alrededor del
    equilibrio invertido (theta=pi), que es el punto de control.
    """
    x1s, x2s, x3s, x4s, us = sp.symbols("x1 x2 x3 x4 u", real=True)
    Mp, Mt, Jp, l, g = sp.symbols("Mp Mt Jp l g", positive=True)
    c, b = sp.symbols("c b", positive=True)
    kt, Ra, r = sp.symbols("kt Ra r", positive=True)
    alpha = sp.symbols("alpha", positive=True)

    sin2, cos2 = sp.sin(x2s), sp.cos(x2s)
    Delta = alpha + Mp**2 * l**2 * sin2**2
    motor_force = kt * us / (Ra * r)
    back_emf = kt**2 * x3s / (Ra * r**2)

    xdot1 = x3s
    xdot2 = x4s
    xdot3 = (
        Mp * l * b * x4s * cos2
        + Mp**2 * l**2 * g * sin2 * cos2
        + (Jp + Mp * l**2) * (motor_force - back_emf - c * x3s + Mp * l * x4s**2 * sin2)
    ) / Delta
    xdot4 = (
        -motor_force * Mp * l * cos2
        + back_emf * Mp * l * cos2
        + c * Mp * l * x3s * cos2
        - Mp**2 * l**2 * x4s**2 * sin2 * cos2
        + (Mt + Mp) * (-b * x4s - Mp * g * l * sin2)
    ) / Delta

    f = sp.Matrix([xdot1, xdot2, xdot3, xdot4])
    state_vars = sp.Matrix([x1s, x2s, x3s, x4s])

    A_sym = f.jacobian(state_vars)
    B_sym = f.jacobian(sp.Matrix([us]))

    subs = {
        Mp: p.Mp,
        Mt: p.Mt,
        Jp: p.Jp,
        l: p.l,
        g: p.g,
        c: p.c,
        b: p.b,
        kt: p.kt,
        Ra: p.Ra,
        r: p.r,
        alpha: p.alpha,
        x1s: x_eq[0],
        x2s: x_eq[1],
        x3s: x_eq[2],
        x4s: x_eq[3],
        us: u_eq,
    }

    A = np.array(A_sym.subs(subs)).astype(np.float64)
    B = np.array(B_sym.subs(subs)).astype(np.float64)
    return A, B


def make_jacobian_fn(p: PendulumParams):
    """Compila (una sola vez, vía sympy.lambdify) A(x,u) = df/dx, B(x,u) = df/du
    evaluables en cualquier punto de trabajo -- usado por el EKF para
    relinealizar en cada paso (como PendulumJacobian.m), sin pagar el costo
    de sustitución simbólica repetida de `linearize()`.
    """
    x1s, x2s, x3s, x4s, us = sp.symbols("x1 x2 x3 x4 u", real=True)

    Mp, Mt, Jp, l, g = p.Mp, p.Mt, p.Jp, p.l, p.g
    c, b = p.c, p.b
    kt, Ra, r = p.kt, p.Ra, p.r
    alpha = p.alpha

    sin2, cos2 = sp.sin(x2s), sp.cos(x2s)
    Delta = alpha + Mp**2 * l**2 * sin2**2
    motor_force = kt * us / (Ra * r)
    back_emf = kt**2 * x3s / (Ra * r**2)

    xdot1 = x3s
    xdot2 = x4s
    xdot3 = (
        Mp * l * b * x4s * cos2
        + Mp**2 * l**2 * g * sin2 * cos2
        + (Jp + Mp * l**2) * (motor_force - back_emf - c * x3s + Mp * l * x4s**2 * sin2)
    ) / Delta
    xdot4 = (
        -motor_force * Mp * l * cos2
        + back_emf * Mp * l * cos2
        + c * Mp * l * x3s * cos2
        - Mp**2 * l**2 * x4s**2 * sin2 * cos2
        + (Mt + Mp) * (-b * x4s - Mp * g * l * sin2)
    ) / Delta

    f = sp.Matrix([xdot1, xdot2, xdot3, xdot4])
    state_vars = sp.Matrix([x1s, x2s, x3s, x4s])

    A_sym = f.jacobian(state_vars)
    B_sym = f.jacobian(sp.Matrix([us]))

    A_fn = sp.lambdify((x1s, x2s, x3s, x4s, us), A_sym, modules="numpy")
    B_fn = sp.lambdify((x1s, x2s, x3s, x4s, us), B_sym, modules="numpy")

    def jacobian(x: State, u: float) -> tuple[np.ndarray, np.ndarray]:
        A = np.array(A_fn(x[0], x[1], x[2], x[3], u), dtype=np.float64).reshape(4, 4)
        B = np.array(B_fn(x[0], x[1], x[2], x[3], u), dtype=np.float64).reshape(4, 1)
        return A, B

    return jacobian
