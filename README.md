# Controller fragility design space

Simulation code for the study *Design space analysis of predictive, optimal and
learned control for an unstable underactuated plant*.

This repository is the simulation layer only: the plant equations, every
numerical setting, and the scripts that run the experiments. It does not build
the figures, tables or manuscript. Each run writes raw results under `results/`,
from which the paper artefacts are produced elsewhere.

The study is exclusively computational. There is no experimental or hardware
validation, and the conclusions describe the simulated plant and the specific
envelope examined.

---

## 1. Plant

A pendulum on a cart driven by a DC motor through a pulley. One actuator (the
armature voltage) governs two degrees of freedom.

State and input:

$$
x = \begin{bmatrix} x_c & \theta & \dot{x}_c & \dot{\theta} \end{bmatrix}^{\top},
\qquad u = V_a \in [-12,\ 12]\ \text{V}.
$$

Angle convention: $\theta = 0$ is the stable hanging equilibrium, $\theta = \pi$
is the inverted equilibrium, and every controller here operates around
$\theta = \pi$. Measured outputs are $x_c$ and $\theta$.

### 1.1 Nonlinear model

With $s = \sin\theta$, $c = \cos\theta$, and the abbreviations

$$
M_t = M_c + \frac{J_m}{r^2}, \qquad
\alpha = J_p (M_t + M_p) + M_t M_p l^2, \qquad
\Delta = \alpha + M_p^2 l^2 s^2,
$$

$$
F_m = \frac{k_t\, u}{R_m\, r}, \qquad
F_b = \frac{k_t^2\, \dot{x}_c}{R_m\, r^2},
$$

the equations of motion are

$$
\ddot{x}_c =
\frac{M_p l\, b\, \dot\theta\, c
\;+\; M_p^2 l^2 g\, s\, c
\;+\; (J_p + M_p l^2)\bigl(F_m - F_b - c_{\text{cart}}\dot{x}_c + M_p l\, \dot\theta^{2} s\bigr)}
{\Delta},
$$

$$
\ddot{\theta} =
\frac{-F_m M_p l\, c
\;+\; F_b M_p l\, c
\;+\; c_{\text{cart}} M_p l\, \dot{x}_c\, c
\;-\; M_p^2 l^2 \dot\theta^{2} s\, c
\;+\; (M_t + M_p)\bigl(-b\,\dot\theta - M_p g l\, s\bigr)}
{\Delta},
$$

where $c_{\text{cart}}$ is the cart viscous friction (`c` in the parameter file)
and $b$ the pivot viscous friction. Source: `src/models/pendulum.py`.

Integration is fixed-step **RK4**. In every experiment the plant advances by one
RK4 step per control interval, so the integration step equals the controller
sample time $T_s$.

### 1.2 Physical parameters

Single source of truth: [`configs/p1_canonical_parameters.yaml`](configs/p1_canonical_parameters.yaml).
No module redefines these; all code builds `PendulumParams` from that file.

| Symbol | Value | Unit | Meaning |
|---|---|---|---|
| $r$ | 0.02075 | m | motor pulley radius |
| $J_m$ | 8.458354293×10⁻⁹ | kg·m² | rotor inertia |
| $R_m$ | 4.3 | Ω | armature resistance |
| $k_t$ | 0.219848705 | N·m/A | torque / back-EMF constant ($k_t = k_b$) |
| $M_c$ | 0.190 | kg | cart mass |
| $c_{\text{cart}}$ | 0.7 | N·s/m | cart viscous friction |
| $M_p$ | 0.097 | kg | pendulum mass |
| $J_p$ | 0.00517333 | kg·m² | pendulum inertia about the pivot |
| $l$ | 0.2 | m | pivot to centre of mass |
| $b$ | 7.892×10⁻⁵ | N·m·s/rad | pivot viscous friction |
| $g$ | 9.81 | m/s² | gravity |

### 1.3 Linearisation and discretisation

`linearize()` builds the Jacobian of the nonlinear model symbolically (SymPy) at
$x_{eq} = [x_{\text{ref}}, \pi, 0, 0]$, $u_{eq} = 0$, giving continuous
$(A, B)$. The linearised plant has an unstable pole at approximately
$\lambda_u \approx +4.596$ rad/s. Once the regulator closes the loop the modes
span two time scales, a dominant one settling in $\tau_{\text{dom}} \approx
0.725$ s and the fastest decaying in $\approx 0.007$ s.

For the predictive controller the pair is discretised by exact zero-order hold,

$$
\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix}
= \exp\!\left( \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} T_s \right).
$$

---

## 2. Controllers

### 2.1 Cost matched LQR

Continuous-time infinite-horizon regulator on the error $e = x - x_{eq}$, gain
from the continuous algebraic Riccati equation (`control.lqr`):

$$
J = \int_0^\infty \bigl( e^{\top} Q\, e + R\, u^2 \bigr)\, dt,
\qquad
K = R^{-1} B^{\top} P,
\qquad
A^{\top} P + P A - P B R^{-1} B^{\top} P + Q = 0.
$$

The same $Q$, $R$ are reused as the MPC stage cost, so the two designs are
cost matched.

$$
Q = \operatorname{diag}(1.0,\ 12.0,\ 0.05,\ 0.02),
\qquad R = 0.002.
$$

Source: `src/controllers/lqr.py`, weights in `src/igrrl/mpc_design.py`
(`Q_P2`, `R_P2`).

### 2.2 Condensed predictive controller

Finite-horizon problem on the discretised linear model, solved at each step with
only the free control moves as decision variables (dense / condensed form, as
used in embedded implementations):

$$
\min_{v}\ \sum_{k=0}^{N_p - 1}\bigl( x_k^{\top} Q x_k + R\, u_k^2 \bigr)
+ x_{N_p}^{\top} P_f\, x_{N_p}
\quad\text{s.t.}\quad
x_{k+1} = A_d x_k + B_d u_k,\ \ |u_k| \le 12\ \text{V},
$$

with $N_p = \operatorname{round}(T_{\text{pred}}/T_s)$, $N_c = 20$ free moves, and
an index map $u_k = v_{b(k)}$.

* **Block distribution.** `uniform` splits the horizon into equal blocks;
  `front` uses geometrically growing block lengths ($1.35^{\,i}$), fine early and
  coarse late.
* **Terminal weight.** `stage` sets $P_f = Q$; `riccati` sets $P_f$ to the
  solution of the discrete algebraic Riccati equation
  $P_f = A_d^{\top} P_f A_d - A_d^{\top} P_f B_d (R + B_d^{\top} P_f B_d)^{-1} B_d^{\top} P_f A_d + Q$.
* **Optional cart constraint.** $|x_c| \le x_{\text{limit}}$ enforced with an
  $\ell_1$ slack of weight $10^4$; disabled ($x_{\text{limit}} = \text{None}$)
  unless a sweep sets it.
* **Conditioning.** Condensation on an unstable plant makes the Hessian
  ill-conditioned as the horizon grows,
  $\kappa(H) = \mathcal{O}\!\bigl( \exp(2 \lambda_u T_{\text{pred}}) \bigr)$;
  measured from $2.3\times10^{2}$ at $T_{\text{pred}} = 0.4$ s to
  $1.1\times10^{13}$ at $3.0$ s.

Solver: CVXPY with CLARABEL, warm started. Source: `src/igrrl/mpc_design.py`
(design-space sweep) and `src/igrrl/mpc_constrained.py` (final comparison).

### 2.3 Extended Kalman filter

Output feedback from $(x_c, \theta)$; no controller sees the true state. Joseph
form update. Covariances:

$$
Q_{\text{EKF}} = \operatorname{diag}(10^{-9}, 10^{-9}, 10^{-8}, 10^{-8}),
\qquad
R_{\text{EKF}} = \operatorname{diag}(10^{-6}, 10^{-6}),
\qquad
P_0 = 10^{-3}\, I_4 .
$$

$R_{\text{EKF}}$ matches the measurement noise variance ($\sigma_{\text{meas}} =
10^{-3}$ on both channels) used by the evaluation harness. `src/estimators/ekf.py`
ships a different default $R = \operatorname{diag}(10^{-8}, 10^{-6})$; the P2
harness overrides it with the value above. Source: `src/estimators/ekf.py`,
`src/igrrl/evaluate_common.py`.

### 2.4 Learned agents (SAC, DDPG)

Stable-Baselines3, environment `src/igrrl/env_standalone.py`
(`StandaloneBalanceEnv`), stabilisation only.

* **Observation (5).** $[\,x_c,\ \sin\theta_e,\ \cos\theta_e,\ \dot{x}_c,\ \dot\theta\,]$
  with $\theta_e = \operatorname{wrap}(\theta - \pi)$, from the EKF estimate.
* **Action (1).** $a \in [-1, 1]$, mapped linearly to the full actuator range,
  $u = 12 \cdot \operatorname{clip}(a, -1, 1)$ V.
* **Reward.**
  $r = -\bigl( 12\,\theta_e^2 + 1.0\, x_c^2 + 0.05\, \dot{x}_c^2 + 0.02\, \dot\theta^2 + 0.002\, u^2 \bigr)$,
  i.e. the negated LQR stage cost on the realised state and action, with an
  added $-100$ when an episode terminates early.
* **Termination.** $|x_c| > 1.5$ m or $|\theta_e| > 1.4$ rad. The return is a
  sum of negated stage costs and is therefore bounded above by zero.

---

## 3. Numerical settings to reproduce the experiments

### 3.1 Shared evaluation protocol (`src/igrrl/evaluate_common.py`)

| Setting | Value |
|---|---|
| Integrator | RK4, one step per control interval |
| Control / integration step $T_s$ | 0.02 s (final comparison); swept in the design grid |
| Simulation horizon | 6.0 s |
| Initial condition | $x_c = \dot{x}_c = \dot\theta = 0$, $\theta_0 = \pi + \mathcal{U}(-0.02, 0.02)$ rad |
| Disturbance | additive input voltage $d_v$ over $t \in [1.0,\ 1.4)$ s (a 0.4 s window) |
| Disturbance levels (final comparison) | $d_v \in \{4,\ 6,\ 8,\ 10\}$ V |
| Measurement noise | Gaussian, $\sigma = 10^{-3}$ on $x_c$ and $\theta$ |
| Actuator limit | $|u| \le 12$ V |
| Fall criterion | RMS angular error $> 0.30$ rad |
| Rail limit | $|x_c| > 0.30$ m counts as a rail violation |
| Realisations per condition | 16 (design grid), 48 (final comparison) |
| Pairing | base seed 9000; realisation $i$ gives every controller the same sampled plant, disturbance and noise |
| Domain randomisation (per realisation, multiplicative on nominal) | $M_p \sim \mathcal{U}(0.80, 1.20)$, $J_p \sim \mathcal{U}(0.75, 1.25)$, $l \sim \mathcal{U}(0.85, 1.15)$, $k_t \sim \mathcal{U}(0.90, 1.10)$ |

### 3.2 Design-space grid (`scripts/grid_mpc_guidelines.py` and batch overrides)

224 predictive configurations, the full Cartesian product

| Variable | Values |
|---|---|
| Sampling period $T_s$ [s] | 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15 |
| Prediction horizon $T_{\text{pred}}$ [s] | 0.4, 0.8, 1.5, 2.2, 3.0, 3.5, 4.5 |
| Block distribution | uniform, front |
| Terminal weight | stage, riccati |

$8 \times 7 \times 2 \times 2 = 224$. Control horizon $N_c = 20$ throughout.
Horizon 6.0 s, 16 paired realisations per configuration, sustained disturbance
$d_v = 8$ V over $[1.0, 1.4)$ s. `GRID_DTS`, `GRID_TPREDS` and `GRID_OUT`
environment variables select the sub-grid written by each batch.

The two tuning rules under test, stated as falsifiable inequalities on
$\tau_{\text{dom}} = 0.725$ s:

$$
\textbf{G1 (sampling): } T_s \in [0.10\,\tau_{\text{dom}},\ 0.25\,\tau_{\text{dom}}]
= [0.073,\ 0.181]\ \text{s},
\qquad
\textbf{G2 (horizon): } T_{\text{pred}} \ge 4\,\tau_{\text{dom}} = 2.90\ \text{s}.
$$

### 3.3 Final comparison (`scripts/evaluate_final.py`)

MPC configuration $T_s = 0.01$ s, $T_{\text{pred}} = 0.8$ s, $N_c = 20$,
`front`, `riccati`. 48 paired realisations at each of $d_v \in \{4, 6, 8, 10\}$ V.
The learned entries are the five best checkpoints per algorithm.

### 3.4 RL training (`scripts/train_standalone_matrix.py`)

| Setting | Value |
|---|---|
| Training environment step | 0.02 s |
| Episode length | 8.0 s (400 steps) |
| Reset angle | $\theta_0 = \pi + \mathcal{U}(-0.30, 0.30)$ rad |
| Training-time disturbance | present with probability 0.5; start step $\sim \mathcal{U}(2.0/T_s,\ N_{\max}-10)$; then 50/50 a single-step impulse of amplitude $\mathcal{U}(-40, 40)$ or a sustained voltage $\mathcal{U}(-6, 6)$ V |
| Training-time domain randomisation | same ranges as the evaluation protocol |
| Budget | 300 000 timesteps per seed |
| Evaluation callback | every 10 000 steps, 10 episodes, best checkpoint kept; eval-env seed = train seed + 50 000 |
| Replay buffer | 150 000 |
| Learning starts | 5 000 |
| Batch size | 256 |
| $\tau$ (target smoothing) | 0.005 |
| $\gamma$ | 0.99 |
| `train_freq` | 1 |
| DDPG exploration noise | $\mathcal{N}(0,\ 0.1)$ per action dimension |
| Selected hyperparameters | SAC: lr $3\times10^{-4}$, net $[64, 64]$. DDPG: lr $10^{-4}$, net $[256, 256]$ |

### 3.5 RL hyperparameter search (`scripts/sweep_rl_hyperparams.py`)

| Setting | Value |
|---|---|
| Learning rate | $\{10^{-4},\ 3\times10^{-4},\ 10^{-3}\}$ |
| Network width | $\{[64, 64],\ [256, 256]\}$ |
| Configurations per algorithm | 6 |
| Budget per configuration | 150 000 timesteps |
| Evaluation | 30 episodes |
| Tuning seed | 101 (disjoint from the validation seeds and from the evaluation seed base) |

### 3.6 Random seeds

| Purpose | Seed(s) |
|---|---|
| Global | 0 |
| RL training / validation | 1, 2, 3, 4, 5 |
| RL hyperparameter tuning | 101 |
| RL eval environment | train seed + 50 000 |
| Paired evaluation (all experiments) | base 9000, realisation $i$ → seed $9000 + i$ |

### 3.7 Computation environment

Computation times were measured on an Intel Core i5-10400T (6 cores, 12 threads,
2.00 GHz), with 16 GB RAM, running Windows 11 Pro (build 26200) and Python
3.12.10, using CVXPY 1.9.2 and CLARABEL 0.11.1.

---

## 4. Running the simulations

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

The sweep and the agent training are the expensive steps. Each script writes its
raw output under `results/`.

```bash
python scripts/grid_mpc_guidelines.py        # 224-configuration design space
python scripts/train_standalone_matrix.py --algo sac  --seed 1   # repeat seeds 1..5
python scripts/train_standalone_matrix.py --algo ddpg --seed 1   # repeat seeds 1..5
python scripts/sweep_rl_hyperparams.py --algo sac                # and --algo ddpg
python scripts/evaluate_final.py             # paired 12-controller comparison
python scripts/compare_checkpoints.py        # best vs final checkpoint, on the plant
python scripts/measure_cost.py               # per-step control cost, closed loop
python scripts/measure_conditioning.py       # Hessian condition number and solver status
```

The design grid is completed by re-running `grid_mpc_guidelines.py` with the
`GRID_DTS`, `GRID_TPREDS` and `GRID_OUT` environment variables set to the missing
sub-grids.

### Checks

```bash
python scripts/verify_controllers.py    # each controller against a property it must satisfy by definition
python -m pytest tests/ -q              # invariants, each encoding a defect found during the study
```

`verify_controllers.py` checks the regulator against the continuous Riccati
equation and the condensed predictive matrices against a direct propagation of
the same control sequence, so it does not depend on any second implementation.
The pytest suite checks, among other invariants, that domain randomisation
actually varies between episodes and that the action mapping reaches the full
actuator range.

---

## 5. Layout

| Path | Contents |
|---|---|
| `src/models/` | Nonlinear plant, RK4, linearisation |
| `src/controllers/` | Cost matched LQR |
| `src/estimators/` | Extended Kalman filter |
| `src/igrrl/` | Condensed predictive controller, training environment, evaluation harness, design-grid loader |
| `scripts/` | Every simulation, sweep, training and check script |
| `configs/` | Central plant parameters (`p1_canonical_parameters.yaml`) |
| `tests/` | Invariant tests |

Simulation outputs, trained checkpoints, figures and tables are not tracked; they
are rebuilt by running the scripts above.

## License

MIT. See [`LICENSE`](LICENSE).
