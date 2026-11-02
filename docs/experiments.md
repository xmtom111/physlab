# Experiments

An experiment is a protocol, a factorial run, a table, a fit, and a report.

## Protocol

`ExperimentProtocol` names the question, the apparatus, the independent
levels, the measurements, the controlled constants, and the procedure.
`repetitions` is at least 1. `safety_class` is `sim_only`, `low`, or
`needs_approval`. `seed` shuffles the trial order.

Each procedure step names a skill. `SkillRegistry` holds the arm skills
(`safe_transit`, `approach`, `retreat`, `navigate_and_reach`, `pick`,
`place`), `run_controller`, and the plants below. `validate_protocol`
returns `unknown skill <name>` when a step is not in that set. The runner
refuses to start when the list is not empty.

## Runner and storage

`ExperimentRunner(event_log, world=None, robot=None, devices=None, skills=None)`
builds the full grid of independent values times repetitions and shuffles it
with `numpy.random.default_rng(protocol.seed)`. Skills receive a
`TrialContext`. A skill that raises is recorded with its message. The trial
stays in the table.

Files land in `runs/<run_id>/`:

- `protocol.json`
- `trials.parquet` (failures included)
- `events.jsonl`
- `media/` for plots

## Fitting

`fit_models` tries linear, power, exponential, logarithmic, inverse, and
quadratic curves with `scipy.optimize.curve_fit`. The power law starts from
a log-log line. Each fit reports parameters, standard errors, R², AIC, BIC,
and the residual standard deviation. The list is sorted by AIC, lowest
first.

AIC is `n log(RSS / n) + 2k`. BIC uses `k log(n)` in place of `2k`.

`fit_custom` takes an expression. The parser allows numbers, `x`, the
parameter names, `+ - * / **`, parentheses, and `sin`, `cos`, `exp`, `log`,
`sqrt`. It does not call `eval`.

`report` writes Markdown with the trial table, the fit table, and a
matplotlib plot under `media/`.

## Library

All three use seed 0.

| Id | Apparatus | Sweep | Model | Bar |
| --- | --- | --- | --- | --- |
| E1 | `pendulum_kit` | length 0.2–0.8 m | `T = a L^b`, then `T = 2π √(L/g)` | exponent 0.5 ± 0.05, `g` within 5% |
| E2 | `incline_ramp` | 10–35 deg | `a = g (sin θ − μ cos θ)` | `μ` within 0.05 of 0.1 |
| E3 | `microscope` | object distance | `v = u f / (u − f)` | `f` within 3% of 0.05 m |

E1 is a hinge plus a slide joint held by a position servo. The bob is pulled
to 10 degrees and released. The period is the mean gap between positive-going
zero crossings of the bob's horizontal position.

E2 slides a block along the ramp. The joint friction loss is
`μ m g cos θ` with `μ = 0.1`. Acceleration is the quadratic coefficient of
the tracked position.

E3 hides the focal length. A search moves the image distance until the
geometric blur circle is smallest, then the thin-lens equation is fit.

```bash
uv run python examples/09_pendulum_law.py
```

The report is `runs/<run_id>/report.md`. The example exits 1 when the
pendulum bars are missed.
