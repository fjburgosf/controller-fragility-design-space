# Where controller fragility lives

Simulation code for the study *Where controller fragility lives. A design space
study of optimal, predictive and learned control for an unstable underactuated
plant*.

Every number in the manuscript is produced by a script in this repository. No
value was typed by hand: each table and each figure is generated from stored
results, and a set of verification scripts recomputes every numerical claim of
the text directly from those results.

## What the study does

A cart and pendulum plant whose closed loop modes differ by a factor near one
hundred is simulated under parametric uncertainty and an unmeasured input
disturbance. Two published tuning rules for predictive control are treated as
falsifiable statements and tested over a design space of 176 configurations,
alongside a cost matched linear quadratic regulator and two learned agents
trained with soft actor critic and deep deterministic policy gradient.

The study is exclusively computational. There is no experimental or hardware
validation, and the conclusions describe the simulated plant and the specific
envelope examined.

## Contents

This repository holds the code only. Simulation outputs, trained checkpoints,
generated figures and generated tables are not tracked, since they are rebuilt by
running the pipeline.

| Path | Contents |
|---|---|
| `src/igrrl/` | Plant, cost matched regulator, condensed predictive controller, estimator, training environment and evaluation harness |
| `scripts/` | Every experiment, table, figure and verification script |
| `configs/` | Central experiment configuration |
| `tests/` | Invariant tests, each one encoding a defect found during the study |

## Environment

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Running the pipeline

The design space sweep and the agent training are the expensive steps and must
run before the tables and figures can be built.

```bash
python scripts/grid_mpc_guidelines.py
python scripts/train_standalone_matrix.py
python scripts/evaluate_final.py
python scripts/measure_cost.py
```

Tables and figures then rebuild in seconds from the stored results.

```bash
python scripts/tables_p2.py
python scripts/tables_final.py
python scripts/fig_guidelines.py
python scripts/fig_design_space.py
python scripts/fig_rl_fragility.py
python scripts/fig4_tails.py
python scripts/fig5_timeseries.py
```

## Verification

Seven scripts check different surfaces of the work. Each reports how many checks
passed and lists every discrepancy found.

```bash
python scripts/verify_controllers.py
python scripts/verify_pass1.py
python scripts/verify_pass2.py
python scripts/verify_figures.py
python scripts/verify_numbering.py
python scripts/verify_language.py
python scripts/verify_docx.py
```

`verify_controllers.py` checks each controller against a mathematical property it
must satisfy by definition rather than against another implementation, so the
regulator is checked against the continuous Riccati equation and the condensed
predictive matrices against a direct propagation of the same control sequence.
`verify_pass1.py` recomputes every numerical claim of the manuscript from its
source file. `verify_figures.py` recomputes the numbers attributed to each
figure. `verify_numbering.py` checks that equations, tables, figures and
references are numbered without gaps, cited in ascending order and cited at all.
`verify_language.py` checks that every table cell and every figure label reaches
the reader in English.

## Invariant tests

```bash
python -m pytest tests/ -q
```

Each test corresponds to a real defect found during the study, so a failure means
that defect has returned. Two are worth naming. One checks that domain
randomisation actually varies between episodes, after a reset that reseeded the
generator on every episode silently disabled it. Another checks that the action
mapping reaches the full actuator range, after a double hyperbolic tangent capped
the learned agents at seventy six percent of the authority available to the
classical controllers.

## Citation

The manuscript is under review. Citation details will be added once it is
published.

## License

MIT. See `LICENSE`.
