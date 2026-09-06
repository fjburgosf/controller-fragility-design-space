"""Verificacion independiente de la matriz SAC. No confia en meta.json."""
import glob, json, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from stable_baselines3 import SAC
from igrrl.env_standalone import StandaloneBalanceEnv
from igrrl.evaluate_common import make_policy, make_lqr, run_episode, EVAL_SEED0

print("1. El entorno de EVALUACION realmente aleatoriza?")
e = StandaloneBalanceEnv(domain_randomization=True, seed=51000)
mps = [e.reset()[1]['parameters'].Mp for _ in range(5)]
print(f"   Mp en 5 episodios: {[round(m,5) for m in mps]}")
print(f"   -> {'VARIA (ok)' if len(set(mps))>1 else 'CONGELADO (BUG)'}\n")

print("2. Los 5 modelos son DISTINTOS entre si?")
ws = {}
for s in range(1, 6):
    m = SAC.load(f"results/training/standalone_sac/seed_{s}/best_model")
    w = np.concatenate([p.detach().numpy().ravel() for p in m.policy.actor.parameters()])
    ws[s] = float(np.abs(w).sum())
    print(f"   s{s}  |W|_1 actor = {ws[s]:.4f}")
u = len({round(v, 3) for v in ws.values()})
print(f"   -> {'5 modelos distintos (ok)' if u == 5 else f'solo {u} distintos (SOSPECHOSO)'}\n")

print("3. El best_model REALMENTE supera al final? (evaluacion propia, n=24)")
for s in range(1, 6):
    row = []
    for tag in ("best_model", "model_final"):
        try:
            mdl = SAC.load(f"results/training/standalone_sac/seed_{s}/{tag}")
        except Exception:
            row.append(None); continue
        eps = [run_episode(make_policy(mdl), EVAL_SEED0 + i, dv=5.0) for i in range(24)]
        row.append((np.mean([e['fell'] for e in eps]),
                    np.mean([e['rmse_theta'] for e in eps])))
    b, f = row
    if b and f:
        ok = "best mejor" if b[1] <= f[1] else "FINAL MEJOR (contradice meta.json)"
        print(f"   s{s}  best cae={b[0]:.2f} rmseTh={b[1]:.4f} | final cae={f[0]:.2f} rmseTh={f[1]:.4f}  -> {ok}")

print("\n4. La politica realmente controla? (vs LQR, mismas realizaciones, dv=5)")
lq = [run_episode(make_lqr(), EVAL_SEED0 + i, dv=5.0) for i in range(24)]
print(f"   LQR      cae={np.mean([e['fell'] for e in lq]):.2f}  rmseTh={np.mean([e['rmse_theta'] for e in lq]):.4f}")
for s in range(1, 6):
    m = SAC.load(f"results/training/standalone_sac/seed_{s}/best_model")
    eps = [run_episode(make_policy(m), EVAL_SEED0 + i, dv=5.0) for i in range(24)]
    print(f"   SAC s{s}  cae={np.mean([e['fell'] for e in eps]):.2f}  rmseTh={np.mean([e['rmse_theta'] for e in eps]):.4f}")

print("\n5. Mapeo de accion coherente entre entorno y arnes?")
m = SAC.load("results/training/standalone_sac/seed_1/best_model")
env = StandaloneBalanceEnv(domain_randomization=False, seed=0)
obs, _ = env.reset(seed=0)
a, _ = m.predict(obs, deterministic=True)
_, _, _, _, info = env.step(a)
u_harness = make_policy(m)(0.0, env.xhat)
print(f"   u del entorno   = {info['u_control']:.4f} V")
print(f"   u del arnes     = {u_harness:.4f} V  (estados distintos, solo se compara el rango)")
print(f"   ambos dentro de +-12 V: {abs(info['u_control'])<=12.001 and abs(u_harness)<=12.001}")
