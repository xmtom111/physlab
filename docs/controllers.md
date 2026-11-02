# Controllers

An agent can write a controller, run it against a simulated plant, and read a
short report when the run fails. The controller never receives the simulator.

## API

`physlab.controllers.api` is the only physlab module controller source may
import. `Controller.setup`, `step`, and `teardown` see a `ControllerContext`
(`time_s`, `dt_s`, `params`, `log`, `assert_that`) plus an observation mapping.
`step` returns an action mapping. Names come from a `DeviceManifest`: each
sensor has a unit, a noise standard deviation, and a range; each actuator has
a unit, `min`, `max`, and a `rate_limit` in command units per second.

`assert_that` records a failure and continues the step. The report lists those
failures. It does not turn them into an exception.

## Static check

`check_source` parses the program with `ast` before anything runs. Imports
must be `math`, `numpy` (and its submodules), `dataclasses`, `typing`,
`collections` (and `collections.abc`), or `physlab.controllers.api`. Names and
attributes `open`, `exec`, `eval`, `compile`, `__import__`, `globals`,
`locals`, `os`, `sys`, `subprocess`, `socket`, `threading`, and `time` are
rejected, as is any dunder attribute and `getattr` of a dunder string.
Violations carry line numbers.

This check is a guardrail. It catches accidents and the obvious escapes. It is
not a security boundary: a string can still hide a call the walk does not
name. The boundary is the process sandbox.

## Sandbox and runner

`run_controller` spawns a process with `multiprocessing` context `"spawn"`.
The child sets `RLIMIT_CPU` and `RLIMIT_AS` to 1 GiB, then loads the source.
The parent owns the plant and exchanges one observation dict and one action
dict over a `Pipe` at `control_hz` (default 100). Actions are clipped to the
manifest range and rate limit. Each clip is a `ClippingEvent` on the report.

A step that does not answer before `step_timeout_s`, or a run that exceeds
`wall_timeout_s`, ends as `TIMEOUT`. An exception during import, setup, step,
or teardown is reported as that phase, with a traceback whose `<controller>`
line numbers match the source. The child stdout and stderr are tailed.

`RunReport` also carries a parquet time series, one matplotlib PNG per sensor
and actuator, assertion failures, and `summary_text` of at most 2000
characters. That text is what an agent should read when it debugs the
controller.

## Virtual devices

`physlab.sim.virtual_devices` provides `StepperMotor`, `Servo`,
`DCMotorWithEncoder`, `LED`, `Photodiode`, `LoadCell`, `Thermometer`, and
`Encoder`. Each noisy reading uses a `numpy` generator seeded by the caller.
`LoadCell.read_mujoco` takes the Z component of an `mjSENS_FORCE` site sensor.
`Thermometer.step` is a first-order lag toward ambient plus heater gain.

## Scenarios

`examples/scenarios/` holds three plants:

- `focus_stage` — stepper rate moves a stage 1 mm per revolution. `defocus_m`
  is the signed distance from the focal plane. `focus` peaks at 1 there.
- `pendulum_release` — a servo latches the bob while `hold >= 0.5`. Releasing
  it integrates a 0.25 m pendulum.
- `line_follow` — wheel speeds in rad/s drive a 10 cm track with 3 cm wheels.
  Two photodiodes sit 1.5 cm either side of the chassis and read a bright
  line of half-width 1 cm.

## Firmware

`compile_arduino` and `upload_arduino` call `arduino-cli` when it is on
`PATH`. Otherwise they raise `DependencyMissing`. MicroPython generation is
out of scope. Tests use a fake `arduino-cli` on `PATH` and do not invoke a
real toolchain.

`examples/08_controller_debug_loop.py` runs a focus law with the wrong sign,
prints `summary_text`, then runs the corrected law.
