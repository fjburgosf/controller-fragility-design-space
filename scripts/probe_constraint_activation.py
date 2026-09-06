"""Paso 0 de D: ¿la restriccion de via realmente muerde?

No se inventa un limite. Se mide la excursion natural del carro |x| bajo el
LQR base (R=0.01 canonico y R=10 bien sintonizado) en escenarios que empujan
al carro, y se elige el limite donde la restriccion se activa en una fraccion
util de realizaciones.

Salida: results/processed/constraint_activation.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from igrrl.controller import GateMode
from igrrl.env import ResidualBalanceEnv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "processed" / "constraint_activation.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

EVAL_SEED0 = 9000
N_EVAL = 60


class PushEnv(ResidualBalanceEnv):
    """Variantes de condicion inicial / disturbio que empujan el carro."""

    x0_halfwidth: float = 0.0
    force_sustained: bool = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.x0_halfwidth > 0.0:
            self.state[0] = self.np_random.uniform(-self.x0_halfwidth, self.x0_halfwidth)
            self.xhat = self.state.copy()
        return self._observation(self.gate.authority()), info

    def _sample_disturbance(self):
        if not self.force_sustained:
            return super()._sample_disturbance()
        start = int(self.np_random.integers(int(1.5 / self.dt), int(3.0 / self.dt)))
        return {"kind": "sustained", "start": start,
                "duration": int(2.0 / self.dt),
                "amplitude": float(self.np_random.uniform(2.0, 6.0))}


def run(base_R, x0_hw, forced, dr, n=N_EVAL):
    env = PushEnv(mode=GateMode.MINIMUM, domain_randomization=dr,
                  perturbation_probability=1.0 if forced else 0.5,
                  base_R=base_R)
    env.x0_halfwidth = x0_hw
    env.force_sustained = forced
    peaks, th_rms, term_n = [], [], 0
    zero = np.zeros(1, dtype="float32")
    for ep in range(n):
        env.reset(seed=EVAL_SEED0 + ep)
        done = term = False
        xs, th = [], []
        while not done:
            _, _, term, trunc, _ = env.step(zero)
            done = term or trunc
            xs.append(abs(env.state[0]))
            th.append((env.state[1] - np.pi) ** 2)
        peaks.append(max(xs))
        term_n += int(term)
        if not term:
            th_rms.append(np.sqrt(np.mean(th)))
    p = np.array(peaks)
    return {
        "peak_x_mean": float(p.mean()), "peak_x_p50": float(np.percentile(p, 50)),
        "peak_x_p75": float(np.percentile(p, 75)), "peak_x_p90": float(np.percentile(p, 90)),
        "peak_x_max": float(p.max()),
        "rmse_theta": float(np.mean(th_rms)) if th_rms else float("nan"),
        "n_term": term_n,
        "_peaks": p,
    }


CASES = [
    ("A_init_only",   0.00, False, True),
    ("B_x0_offset",   0.25, False, True),
    ("C_sustained",   0.00, True,  True),
    ("D_x0_sustain",  0.25, True,  True),
]
LIMITS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

rows = []
print(f"{'caso':14s} {'R':>6s} {'mean':>7s} {'p50':>7s} {'p75':>7s} {'p90':>7s} {'max':>7s} {'rmseTh':>7s} {'term':>5s}", flush=True)
store = {}
for name, x0hw, forced, dr in CASES:
    for R in (0.01, 10.0):
        m = run(R, x0hw, forced, dr)
        store[(name, R)] = m.pop("_peaks")
        rows.append({"case": name, "base_R": R, "x0_halfwidth": x0hw,
                     "forced_sustained": forced, **m})
        print(f"{name:14s} {R:6g} {m['peak_x_mean']:7.3f} {m['peak_x_p50']:7.3f} "
              f"{m['peak_x_p75']:7.3f} {m['peak_x_p90']:7.3f} {m['peak_x_max']:7.3f} "
              f"{m['rmse_theta']:7.4f} {m['n_term']:5d}", flush=True)

print("\n=== fraccion de realizaciones que EXCEDEN el limite ===", flush=True)
hdr = "caso           R      " + "".join(f"{L:>7.2f}" for L in LIMITS)
print(hdr, flush=True)
for (name, R), p in store.items():
    frac = [float(np.mean(p > L)) for L in LIMITS]
    print(f"{name:14s} {R:6g} " + "".join(f"{f:7.2f}" for f in frac), flush=True)
    for L, f in zip(LIMITS, frac):
        rows.append({"case": name, "base_R": R, "x0_halfwidth": None,
                     "forced_sustained": None, "limit": L, "frac_exceed": f})

keys = sorted({k for r in rows for k in r})
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"\nGuardado: {OUT}", flush=True)
