"""Swing-up + estabilización — controlador compuesto para arrancar en
theta=0 (péndulo abajo, condición inicial física real) y llevarlo hasta
el equilibrio invertido theta=pi, donde se conmuta a un estabilizador
lineal (LQR o Pole Placement).

A diferencia del resto de `src/controllers/`, la ley de swing-up NO se
extrajo directamente del material fuente auditado con valores numéricos
reutilizables: los desarrollos originales (`Clase 5/.../Lqrplusswingup.slx`,
`IP_SwingUp_Design.slx`) implementan swing-up en Simulink sin una ley
analítica en texto auditable. El usuario sí compartió el código Python
real (`IPFramework/Files/LQR.py`, clase `LQRVisualEnvironment`) que
controló el péndulo físico de PRIMBIO con éxito; sus ganancias están
tuneadas en PWM/centímetros sobre ese hardware específico y no se
reutilizan numéricamente, pero su ESTRATEGIA (kick inicial, bombeo de
energía, conmutación a LQR con voltaje mínimo) informó el diseño de este
módulo.

Historial de diseño (tres controladores de swing-up quedan implementados
para trazabilidad, aunque solo `FeedbackLinearizedSwingUp` es el que
funciona de forma robusta — ver `SwingUpStabilizeController`):

1. `EnergySwingUpController` — bombeo de energía por voltaje directo
   (Åström & Furuta 2000). Validado por simulación: converge a un CICLO
   LÍMITE ESTABLE (~85-90% de la energía objetivo) y nunca cruza a
   theta=pi con los parámetros físicos canónicos, incluso tras 400s de
   simulación y grid search de ganancia. Se investigó subir `kt` o bajar
   `Rm` (motor más fuerte): ambos empeoran el resultado, porque
   BackEMF∝kt²/Rm crece más rápido que MotorForce∝kt/Rm. Se conserva como
   resultado negativo documentado.
2. `GatedEnergySwingUp` — bombeo solo cerca de theta=0 (como el algoritmo
   real de PRIMBIO). Con los parámetros físicos canónicos (motor débil)
   da peor resultado que el bombeo continuo (menor duty cycle efectivo).
3. `FeedbackLinearizedSwingUp` — bombeo de energía + partial feedback
   linearization del canal del carro (Åström & Furuta 2000 / Spong 1995).
   Éste SÍ funciona: valida el 100% de una batería de 40 perturbaciones
   iniciales aleatorias en tests/test_swingup.py. Ver su docstring y el
   de `SwingUpStabilizeController` para el bug de envolvente angular que
   causaba fallos aparentemente aleatorios antes de esta solución.
"""
from __future__ import annotations

import numpy as np

from src.config import PendulumParams
from src.controllers.base import Controller, StateFeedbackController
from src.models.pendulum import dynamics


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class EnergySwingUpController(Controller):
    """Control basado en energía: bombea energía mecánica del péndulo
    hasta la del equilibrio invertido. Energía de referencia (theta=0
    abajo): E(theta,thetadot) = 0.5*J_eff*thetadot^2 - Mp*g*l*cos(theta).
    """

    def __init__(
        self,
        params: PendulumParams,
        k_swing: float = 40.0,
        j_eff: float | None = None,
        bang_bang: bool = False,
        u_max: float = 12.0,
    ):
        self.p = params
        self.k_swing = k_swing
        self.bang_bang = bang_bang
        self.u_max = u_max
        # J_eff: inercia efectiva del péndulo respecto al pivote usada en el
        # término de energía cinética. Se adopta Jp + Mp*l^2, consistente con
        # el término compuesto (Jp+Mp*l^2) que aparece repetidamente en
        # PendulumNonlinear.m (ver docs/CANONICAL_MODEL.md §2).
        self.j_eff = j_eff if j_eff is not None else (params.Jp + params.Mp * params.l**2)

    def energy(self, x: np.ndarray) -> float:
        theta, thetadot = x[1], x[3]
        return 0.5 * self.j_eff * thetadot**2 - self.p.Mp * self.p.g * self.p.l * np.cos(theta)

    def energy_top(self) -> float:
        return self.p.Mp * self.p.g * self.p.l  # E(theta=pi, thetadot=0)

    def compute(self, t: float, x: np.ndarray) -> float:
        theta, thetadot = x[1], x[3]
        e_err = self.energy(x) - self.energy_top()
        # Signo derivado de dE/dt bajo la ley u=k*thetadot*cos(theta)*e_err:
        # en PendulumNonlinear.m el motor entra a xdot4 como
        # -MotorForce*Mp*l*cos(theta) (signo negativo), a diferencia de la
        # convención de texto clásica; el signo de k aquí se deriva de esa
        # ecuación (ver docs/CANONICAL_MODEL.md), no es un valor de texto.
        if self.bang_bang:
            u = self._greedy_bang_bang(x, e_err)
        else:
            u = self.k_swing * thetadot * np.cos(theta) * e_err
        return float(np.clip(u, -self.u_max, self.u_max))

    def _greedy_bang_bang(self, x: np.ndarray, e_err: float) -> float:
        """Elige u en {-u_max, +u_max} evaluando numéricamente dE/dt de la
        dinámica NO lineal completa para cada opción (sin aproximación
        analítica de la ley de swing-up), y toma la que empuja E hacia
        E_top. Es más robusto que una ley analítica linealizada cerca de
        theta=0 (la ley proporcional de este mismo módulo mostró un techo
        de energía en amplitudes grandes al validarla por simulación).
        """
        theta, thetadot = x[1], x[3]

        def d_energy(u_try: float) -> float:
            xdot = dynamics(x, u_try, self.p)
            thetaddot = xdot[3]
            return self.p.Mp * self.p.g * self.p.l * np.sin(theta) * thetadot + self.j_eff * thetadot * thetaddot

        dE_plus = d_energy(self.u_max)
        dE_minus = d_energy(-self.u_max)

        # Si E<E_top (e_err<0) queremos maximizar dE/dt; si E>E_top, minimizarla.
        if e_err < 0:
            return self.u_max if dE_plus > dE_minus else -self.u_max
        return self.u_max if dE_plus < dE_minus else -self.u_max


class GatedEnergySwingUp(Controller):
    """Swing-up con bombeo de energía activo SOLO cerca del punto más bajo
    del arco (theta≈0), y coast (u=0) el resto del ciclo.

    Reconstruido a partir del algoritmo REAL que el usuario confirma que
    funcionó en el péndulo físico de PRIMBIO (`IPFramework/Files/LQR.py`,
    clase `LQRVisualEnvironment.compute_control`, rama de swing-up: pump
    solo si `theta` está dentro de una ventana angular alrededor de 0 y
    la energía actual no ha superado ya un techo; en cualquier otro punto
    del ciclo retorna voltaje cero). La convención de ángulo/signos de esa
    fuente está invertida respecto a la de este proyecto (su hardware
    reconstruye theta con un signo negativo explícito, `w =
    radians(-raw_angular_speed)`) y sus ganancias están tuneadas en PWM
    (0-255) y centímetros, no en voltios/metros SI — por eso NO se copian
    los valores numéricos (k_swingup=1.5, K=[1600,140,-13,-7.5], factor
    300, umbral 0.85) tal cual: se reimplementa la ESTRATEGIA (bombeo
    gateado por proximidad al fondo, coast fuera de esa ventana, kick
    inicial para romper el reposo exacto) sobre las unidades SI y la
    convención angular de este proyecto, con el signo de bombeo ya
    verificado empíricamente en docs/CANONICAL_MODEL.md /
    EnergySwingUpController.
    """

    def __init__(
        self,
        params: PendulumParams,
        k_swing: float = 40.0,
        bottom_window: float = 0.28,     # rad, ventana alrededor de theta=0 (mismo orden que la fuente real)
        energy_cap_ratio: float = 1.05,   # fracción de E_top por encima de la cual se deja de bombear
        kick_voltage: float = 2.2,        # V, mismo valor que la fuente real (startup_kick_voltage)
        kick_max_steps: int = 12,
        rest_omega_threshold: float = 0.08,  # rad/s, mismo valor que la fuente real
    ):
        self.p = params
        self.j_eff = params.Jp + params.Mp * params.l**2
        self.k_swing = k_swing
        self.bottom_window = bottom_window
        self.energy_cap = energy_cap_ratio * params.Mp * params.g * params.l
        self.kick_voltage = kick_voltage
        self.kick_max_steps = kick_max_steps
        self.rest_omega_threshold = rest_omega_threshold
        self._kick_steps_used = 0

    def energy(self, x: np.ndarray) -> float:
        theta, thetadot = x[1], x[3]
        return 0.5 * self.j_eff * thetadot**2 - self.p.Mp * self.p.g * self.p.l * np.cos(theta)

    def energy_top(self) -> float:
        return self.p.Mp * self.p.g * self.p.l

    def compute(self, t: float, x: np.ndarray) -> float:
        theta, thetadot, cart_x = x[1], x[3], x[0]
        theta_wrapped = wrap_to_pi(theta)  # error respecto a theta=0 (fondo)
        near_bottom = abs(theta_wrapped) < self.bottom_window

        # Kick inicial: rompe el equilibrio singular u=0 en reposo exacto
        # (thetadot=0 anula la ley de bombeo por construcción), igual que
        # el "startup_kick" de la fuente real.
        if near_bottom and abs(thetadot) < self.rest_omega_threshold and self._kick_steps_used < self.kick_max_steps:
            self._kick_steps_used += 1
            direction = -1.0 if cart_x >= 0 else 1.0
            return direction * self.kick_voltage

        if not near_bottom:
            return 0.0  # coast: sin control fuera de la ventana del fondo

        e_err = self.energy(x) - self.energy_top()
        if self.energy(x) > self.energy_cap:
            return 0.0  # ya se alcanzó suficiente energía, dejar de bombear

        u = self.k_swing * thetadot * np.cos(theta) * e_err
        return float(np.clip(u, -12.0, 12.0))


class FeedbackLinearizedSwingUp(Controller):
    """Swing-up por bombeo de energía + partial feedback linearization
    del canal del carro (Åström & Furuta 2000; también descrito como
    "collocated linearization" en la literatura de cart-pole, p.ej.
    Spong 1995).

    A diferencia de `EnergySwingUpController` (que comanda voltaje
    directamente y demostró, por simulación, converger a un ciclo límite
    estable sin alcanzar el equilibrio invertido — ver docs de diseño),
    aquí se comanda la ACELERACIÓN deseada del carro:

        a_des = k_energy * thetadot*cos(theta)*(E - E_top)   [bombeo]
                - k_x*(x - x_ref) - k_v*xdot                  [centrado del carro]

    y se invierte exactamente la ecuación de aceleración del carro del
    modelo no lineal (`PendulumNonlinear.m`, término xdot(3)) para
    hallar el voltaje Va que produce esa aceleración -- cancelando así
    los términos de acoplamiento/fricción/back-EMF del canal del carro en
    vez de dejar que compitan con el bombeo de energía. Esto aborda
    directamente la hipótesis de que, en la ley por voltaje directo,
    parte de la energía inyectada se iba a energía cinética del carro
    (disipada luego por fricción `c`) en vez de a energía del péndulo.
    """

    def __init__(
        self,
        params: PendulumParams,
        k_energy: float = 10.0,
        k_x: float = 1.0,
        k_v: float = 0.5,
        x_ref: float = 0.0,
        energy_cap_ratio: float = 1.05,
        kick_voltage: float = 1.5,
        kick_steps: int = 15,
    ):
        self.p = params
        self.j_eff = params.Jp + params.Mp * params.l**2
        self.k_energy = k_energy
        self.k_x = k_x
        self.k_v = k_v
        self.x_ref = x_ref
        self.energy_cap = energy_cap_ratio * params.Mp * params.g * params.l
        # Kick inicial: en reposo EXACTO (theta=0, thetadot=0, x=0, xdot=0)
        # tanto el término de energía como el de centrado del carro se
        # anulan (a_des=0), un equilibrio singular de la propia ley de
        # control. Se rompe con un pulso de voltaje corto, igual en
        # espíritu al "startup_kick" de IPFramework/Files/LQR.py. (Un
        # análisis previo atribuyó erróneamente a este kick una asimetría
        # direccional del swing-up; la causa real era un bug de
        # envolvente angular en `StateFeedbackController.compute`
        # — corregido en src/controllers/base.py — no la dirección del
        # kick, que es irrelevante una vez que hay CUALQUIER perturbación
        # inicial no nula, como confirma la validación en tests/.)
        self.kick_voltage = kick_voltage
        self.kick_steps = kick_steps
        self._kick_used = 0

    def energy(self, x: np.ndarray) -> float:
        theta, thetadot = x[1], x[3]
        return 0.5 * self.j_eff * thetadot**2 - self.p.Mp * self.p.g * self.p.l * np.cos(theta)

    def energy_top(self) -> float:
        return self.p.Mp * self.p.g * self.p.l

    def compute(self, t: float, x: np.ndarray) -> float:
        x1, theta, x3, thetadot = x
        p = self.p

        theta_wrapped = wrap_to_pi(theta)
        if abs(theta_wrapped) < 0.2 and abs(thetadot) < 0.05 and self._kick_used < self.kick_steps:
            self._kick_used += 1
            return -self.kick_voltage

        e_val = self.energy(x)
        e_err = e_val - self.energy_top()
        a_energy = 0.0 if e_val > self.energy_cap else self.k_energy * thetadot * np.cos(theta) * e_err
        a_des = a_energy - self.k_x * (x1 - self.x_ref) - self.k_v * x3

        sin2, cos2 = np.sin(theta), np.cos(theta)
        Delta = p.alpha + p.Mp**2 * p.l**2 * sin2**2
        back_emf = p.kt**2 * x3 / (p.Ra * p.r**2)
        j_pend = p.Jp + p.Mp * p.l**2

        # Inversión exacta de xdot(3) = a_des respecto a motor_force
        # (ver PendulumNonlinear.m / docs/CANONICAL_MODEL.md §2).
        rhs = (
            a_des * Delta
            - p.Mp * p.l * p.b * thetadot * cos2
            - p.Mp**2 * p.l**2 * p.g * sin2 * cos2
            + j_pend * (back_emf + p.c * x3 - p.Mp * p.l * thetadot**2 * sin2)
        )
        motor_force = rhs / j_pend
        Va = motor_force * p.Ra * p.r / p.kt
        return float(np.clip(Va, -12.0, 12.0))


class SwingUpStabilizeController(Controller):
    """Conmuta de swing-up a un estabilizador lineal (LQR/PolePlacement)
    al entrar en la región de captura alrededor de theta=pi.

    Se intentó primero usar como criterio de conmutación la función de
    Lyapunov cuadrática del LQR, V(x)=(x-x_eq)^T S (x-x_eq) — la práctica
    habitual en swing-up (Åström & Furuta 2000). Se verificó por
    simulación que V NO predice de forma monótona la cuenca de atracción
    real en este modelo (el diseño LQR usado tiene Q con peso cero en las
    velocidades, lo que produce una forma cuadrática mal condicionada
    para este propósito) — se descarta ese criterio por evidencia
    empírica, no se usa pese a ser la práctica estándar en la literatura,
    porque en ESTE caso concreto no funciona.

    Además de un umbral en (ángulo, velocidad angular), la captura exige
    un CRUCE POR CERO de thetadot dentro de esa ventana (`_is_turning_point`):
    conmutar en un punto de retorno real, no en un cruce rápido de paso,
    evita dejar al LQR en un estado con velocidad angular alta fuera de su
    cuenca real de atracción. Una vez capturado, el modo queda fijo
    (latch) y no vuelve a swing-up.

    NOTA IMPORTANTE DE DISEÑO: durante la validación se observó una
    aparente "cuenca de atracción asimétrica" del LQR (funcionaba para
    perturbaciones iniciales de un signo pero no del otro). La causa raíz
    NO era física: `StateFeedbackController.compute` (src/controllers/base.py)
    no envolvía el error angular a [-pi,pi], así que un estado con theta
    fuera de [0,2*pi) (p.ej. tras un swing-up que gira en sentido
    negativo) producía un error de varios radianes en vez del error real
    (~0.1-0.2 rad), generando un esfuerzo de control erróneo. Corregido
    en base.py; confirmado con una batería de 40 perturbaciones iniciales
    aleatorias (0 fallas) en tests/test_swingup.py.
    """

    def __init__(
        self,
        params: PendulumParams,
        stabilizer: StateFeedbackController,
        swingup: Controller | None = None,
        theta_capture: float = 0.35,   # rad, calibrado por simulación (ver docstring)
        omega_capture: float = 1.5,     # rad/s
    ):
        self.swingup = swingup if swingup is not None else FeedbackLinearizedSwingUp(params)
        self.stabilizer = stabilizer
        self.theta_capture = theta_capture
        self.omega_capture = omega_capture
        self.mode_log: list[str] = []
        self._captured = False  # latch: una vez en modo estabilización, no regresa a swing-up
        self._prev_thetadot: float | None = None

    def in_capture_region(self, x: np.ndarray) -> bool:
        angle_error = wrap_to_pi(x[1] - np.pi)
        return abs(angle_error) < self.theta_capture and abs(x[3]) < self.omega_capture

    def _is_turning_point(self, x: np.ndarray) -> bool:
        """Cruce por cero de thetadot dentro de la ventana angular: garantiza
        que la captura ocurre en un punto de retorno real (thetadot≈0 por
        el teorema del valor intermedio), no en un cruce rápido de paso.
        Evita el fallo observado con un umbral puramente instantáneo: una
        conmutación en pleno "molinete" con velocidad alta deja al LQR
        fuera de su cuenca real de atracción (ver docs de diseño)."""
        angle_error = wrap_to_pi(x[1] - np.pi)
        if abs(angle_error) >= self.theta_capture:
            self._prev_thetadot = x[3]
            return False
        crossed = self._prev_thetadot is not None and np.sign(x[3]) != np.sign(self._prev_thetadot)
        self._prev_thetadot = x[3]
        return crossed and abs(x[3]) < self.omega_capture

    def compute(self, t: float, x: np.ndarray) -> float:
        if not self._captured and self._is_turning_point(x):
            self._captured = True
        if self._captured:
            self.mode_log.append("stabilize")
            return self.stabilizer.compute(t, x)
        self.mode_log.append("swingup")
        return self.swingup.compute(t, x)
