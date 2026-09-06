"""Invariantes de P2. Cada test corresponde a un bug real cometido en el proyecto.

Su propósito no es cubrir código sino impedir que reaparezcan estos fallos.
"""
import numpy as np
import pytest

from src.config import load_pendulum_params
from src.controllers.lqr import LQRController
from src.models.pendulum import equilibrium_up, linearize
from igrrl.mpc_design import DesignMPC, block_lengths, Q_P2, R_P2
from igrrl.env_standalone import StandaloneBalanceEnv
from igrrl.env import ResidualBalanceEnv

NOM = load_pendulum_params()


# ---------- BUG D33: la aleatorización de dominio estaba muerta ----------

@pytest.mark.parametrize("cls", [StandaloneBalanceEnv, ResidualBalanceEnv])
def test_domain_randomization_realmente_varia(cls):
    """reset() consecutivos DEBEN dar plantas distintas. Bug D33: re-sembraba."""
    env = cls(domain_randomization=True, seed=1)
    mps = []
    for _ in range(5):
        _, info = env.reset()
        mps.append(info["parameters"].Mp)
    assert len(set(mps)) > 1, f"DR muerta: {cls.__name__} repite Mp={mps[0]} en todo episodio"


@pytest.mark.parametrize("cls", [StandaloneBalanceEnv, ResidualBalanceEnv])
def test_reproducibilidad_con_semilla_explicita(cls):
    """reset(seed=X) DEBE ser reproducible pese a la corrección del bug D33."""
    a = cls(domain_randomization=True); _, ia = a.reset(seed=777)
    b = cls(domain_randomization=True); _, ib = b.reset(seed=777)
    assert ia["parameters"].Mp == ib["parameters"].Mp


def test_sin_dr_los_parametros_son_nominales():
    env = StandaloneBalanceEnv(domain_randomization=False, seed=1)
    for _ in range(3):
        _, info = env.reset()
        assert info["parameters"].Mp == NOM.Mp


# ---------- BUG de formulación: el MPC no reproducía al LQR ----------

def test_mpc_sin_restriccion_reproduce_al_lqr():
    """Prueba de subsunción: MPC con mismo coste, terminal Riccati y horizonte
    suficiente DEBE dar una acción cercana a la del LQR lejos de restricciones."""
    lqr = LQRController(NOM, Q=Q_P2.copy(), R=R_P2, x_ref=0.0)
    mpc = DesignMPC(NOM, dt=0.005, T_pred=1.5, Nc=20,
                    blocking="front", terminal="riccati", x_limit=None)
    rng = np.random.default_rng(0)
    difs = []
    for _ in range(12):
        x = np.array([rng.uniform(-0.05, 0.05), np.pi + rng.uniform(-0.05, 0.05),
                      rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2)])
        difs.append(abs(mpc.compute(0.0, x) - lqr.compute(0.0, x)))
    assert np.median(difs) < 1.0, f"MPC no reproduce al LQR: |du| mediana={np.median(difs):.3f} V"


# ---------- Muro de condicionamiento: horizonte largo en planta inestable ----------

def test_condicionamiento_crece_con_el_horizonte():
    """Documenta el muro exponencial: horizonte corto bien condicionado,
    horizonte largo catastrófico. Si esto deja de cumplirse, revisar mpc_design."""
    c_corto = np.linalg.cond(DesignMPC(NOM, dt=0.01, T_pred=0.4, terminal="riccati")._H)
    c_largo = np.linalg.cond(DesignMPC(NOM, dt=0.01, T_pred=3.0, terminal="riccati")._H)
    assert c_corto < 1e5, f"horizonte corto mal condicionado: {c_corto:.2e}"
    assert c_largo > 1e10, f"el muro de condicionamiento no aparece: {c_largo:.2e}"


def test_planta_es_inestable_y_con_escalas_separadas():
    """Premisa del paper: polo inestable y razón de escalas grande."""
    A, B = linearize(equilibrium_up(0.0), 0.0, NOM)
    assert max(np.linalg.eigvals(A).real) > 1.0, "la planta debería ser inestable"
    lqr = LQRController(NOM, Q=Q_P2.copy(), R=R_P2)
    cl = np.linalg.eigvals(A - B @ lqr.K.reshape(1, -1))
    razon = (1 / min(abs(cl.real))) / (1 / max(abs(cl.real)))
    assert razon > 50, f"escalas no separadas: razón={razon:.1f}"


# ---------- Move blocking ----------

def test_blocking_suma_el_horizonte():
    for scheme in ("front", "uniform"):
        for Np, Nc in ((40, 10), (150, 20), (600, 20), (7, 20)):
            L = block_lengths(Np, Nc, scheme)
            assert sum(L) == Np, f"{scheme} Np={Np} Nc={Nc}: suma {sum(L)}"


def test_blocking_front_es_fino_al_inicio():
    L = block_lengths(150, 20, "front")
    assert L[0] <= L[-1] / 4, f"'front' no concentra resolución al inicio: {L}"


# ---------- Guarda de no-finito ----------

def test_mpc_no_revienta_con_estado_no_finito():
    mpc = DesignMPC(NOM, dt=0.01, T_pred=0.8)
    assert mpc.compute(0.0, np.array([np.inf, np.pi, 0.0, 0.0])) == 0.0
    assert mpc.compute(0.0, np.array([np.nan, np.pi, 0.0, 0.0])) == 0.0


# ---------- Entorno standalone: acción de control completa ----------

def test_accion_mapea_al_rango_completo_del_actuador():
    env = StandaloneBalanceEnv(domain_randomization=False, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([5.0]))    # saturado por tanh
    assert abs(info["u_control"]) > 11.0, f"u={info['u_control']}: no alcanza el actuador"


def test_recompensa_no_penaliza_residuo():
    """El entorno standalone NO debe llevar el término de penalización del residuo
    (artefacto D23 que favorecía estructuralmente al control base)."""
    import inspect
    src = inspect.getsource(StandaloneBalanceEnv.step)
    assert "residual_penalty" not in src
