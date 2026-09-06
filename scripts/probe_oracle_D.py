"""Sonda E aplicada a D: puede un residuo acotado capturar la conciencia
de restriccion de via del MPC?

Base: LQR con el mismo coste cuadratico que el MPC (Q_P2, R_P2), sin restriccion.
Oraculo: r*(x) = u_MPC_restringido(x) - u_LQR(x), inyectado como accion con el
mapa del entorno invertido (residual = residual_limit * tanh(clip(action,-1,1))).

VERDE (fijado antes de ver datos): LQR+oraculo reduce la fraccion de
realizaciones que violan L en >= 50 % respecto a LQR puro, con |need| mediana
<= residual_limit. Si ROJO -> un residuo acotado no basta; D se reduce a
comparacion LQR vs MPC (sin residuo aprendido como contribucion central).

Salida: results/processed/oracle_D_probe.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv
from igrrl.mpc_constrained import ConstrainedMPC, Q_P2, R_P2
from src.controllers.lqr import LQRController

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / "oracle_D_probe.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL = 48
X_LIMIT = 0.30
RESID_LIMS = [4.0, 8.0, 12.0, 20.0]
TANH1 = float(np.tanh(1.0))


class DEnv(ResidualBalanceEnv):
    """Regimen de D: offset de carro + disturbio sostenido moderado."""

    x0_hw: float = 0.15

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.state[0] = self.np_random.uniform(-self.x0_hw, self.x0_hw)
        self.xhat = self.state.copy()
        return self._observation(self.gate.authority()), info

    def _sample_disturbance(self):
        start = int(self.np_random.integers(int(1.5 / self.dt), int(3.5 / self.dt)))
        return {"kind": "sustained", "start": start,
                "duration": int(1.5 / self.dt),
                "amplitude": float(self.np_random.uniform(1.0, 3.0))}


def make_env():
    e = DEnv(mode=GateMode.MAXIMUM, domain_randomization=True,
             perturbation_probability=1.0, base_R=0.01, gate_warmup_s=0.0)  # rho_max=1.0 por defecto
    return e


def run(kind, resid_lim=None, n=N_EVAL):
    """kind in {'lqr','mpc','oracle'}."""
    env = make_env()
    lqr = LQRController(env.nominal_params, Q=Q_P2.copy(), R=R_P2)
    mpc = ConstrainedMPC(env.nominal_params, dt=env.dt, x_limit=X_LIMIT)
    viol, th_rms, energy, needs, sat = [], [], [], [], []
    for ep in range(n):
        env.reset(seed=EVAL_SEED0 + ep)
        done = term = False
        xmax, th_sq, e_sum, ep_need, ep_sat, steps = 0.0, [], 0.0, [], 0, 0
        while not done:
            xhat = env.xhat.copy()
            t = env.steps * env.dt
            ub = lqr.compute(t, xhat)
            if kind == "lqr":
                target = 0.0
            else:
                um = mpc.compute(t, xhat)
                target = um - ub
            ep_need.append(abs(target))
            if kind == "mpc":
                # entregar u_MPC exacto: rho*residual = target, rho=1, sin recorte
                # de residuo (solo el actuador de 12 V recorta u_control)
                d = np.clip(target / env.residual_limit, -0.999, 0.999)
                action = np.array([np.arctanh(d)], dtype="float32")
            elif kind == "oracle":
                cap = TANH1 * resid_lim
                if abs(target) > cap:
                    ep_sat += 1
                d = np.clip(target, -0.999 * cap, 0.999 * cap) / resid_lim
                action = np.array([np.arctanh(np.clip(d, -0.999, 0.999))], dtype="float32")
                env.residual_limit = resid_lim
            else:
                action = np.zeros(1, dtype="float32")
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            xmax = max(xmax, abs(env.state[0]))
            th_sq.append((env.state[1] - np.pi) ** 2)
            e_sum += info["u_control"] ** 2 * env.dt
        viol.append(int(xmax > X_LIMIT))
        needs.append(float(np.mean(ep_need)))
        sat.append(ep_sat / max(1, steps))
        if not term:
            th_rms.append(np.sqrt(np.mean(th_sq)))
            energy.append(e_sum)
    return {
        "frac_violate": float(np.mean(viol)),
        "rmse_theta": float(np.mean(th_rms)) if th_rms else float("nan"),
        "energy": float(np.mean(energy)) if energy else float("nan"),
        "need_abs_mean": float(np.mean(needs)),
        "need_abs_p50": float(np.percentile(needs, 50)),
        "need_abs_p90": float(np.percentile(needs, 90)),
        "satur_frac": float(np.mean(sat)),
        "n_term": int(n - len(th_rms)),
    }


rows = []
print(f"X_LIMIT={X_LIMIT}  N={N_EVAL}  (base LQR coste-igualado, MPC restringido)\n", flush=True)

m = run("lqr")
rows.append({"method": "LQR_puro", "resid_lim": "", **m})
lqr_v = m["frac_violate"]
print(f"LQR puro       viol={m['frac_violate']:.3f}  rmseTh={m['rmse_theta']:.4f}  "
      f"E={m['energy']:.1f}  |need|={m['need_abs_mean']:.2f}  term={m['n_term']}", flush=True)

m = run("mpc")
rows.append({"method": "MPC_restringido", "resid_lim": "", **m})
print(f"MPC restring.  viol={m['frac_violate']:.3f}  rmseTh={m['rmse_theta']:.4f}  "
      f"E={m['energy']:.1f}  |need|={m['need_abs_mean']:.2f} (p90 {m['need_abs_p90']:.2f})  "
      f"term={m['n_term']}", flush=True)

for rl in RESID_LIMS:
    m = run("oracle", resid_lim=rl)
    rows.append({"method": "LQR+oraculo", "resid_lim": rl, **m})
    print(f"LQR+orac {rl:4.0f}V viol={m['frac_violate']:.3f}  rmseTh={m['rmse_theta']:.4f}  "
          f"E={m['energy']:.1f}  |need|={m['need_abs_mean']:.2f}  sat={m['satur_frac']:.2f}  "
          f"term={m['n_term']}", flush=True)

keys = ["method", "resid_lim", "frac_violate", "rmse_theta", "energy",
        "need_abs_mean", "need_abs_p50", "need_abs_p90", "satur_frac", "n_term"]
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"\nGuardado: {OUT}", flush=True)

orac = [r for r in rows if r["method"] == "LQR+oraculo"]
best = min(orac, key=lambda r: r["frac_violate"])
red = 1 - best["frac_violate"] / lqr_v if lqr_v > 0 else float("nan")
print(f"\nLQR viola {lqr_v:.2f} -> mejor LQR+oraculo {best['frac_violate']:.2f} "
      f"(reduccion {red*100:.0f} %, limite {best['resid_lim']:.0f} V, "
      f"|need| p50 {best['need_abs_p50']:.1f} V)", flush=True)
verde = red >= 0.50 and best["need_abs_p50"] <= best["resid_lim"]
print(f">>> {'VERDE: entrenar D' if verde else 'ROJO: residuo acotado no basta para D'}", flush=True)
