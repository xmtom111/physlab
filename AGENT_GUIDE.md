# Agent guide

This file tells an automated agent what physlab can do today and which files
the agent must leave alone. The MCP server is not implemented yet, and no
tools are exposed.

## What works now

- Build and compose poses in named frames. Quaternions are `(w, x, y, z)`.
  Lengths are meters and angles are radians. See `docs/core.md`.
- Load `configs/default.yaml` with `physlab.core.config.load_config`. Override
  values with `PHYSLAB_` environment variables (`PHYSLAB_SIM__TIMESTEP_S`).
- Write a JSONL run log with `physlab.core.events.EventLog`.
- Register backends by name with `physlab.core.registry.Registry`.
- Inspect the install with `physlab --version` and `physlab config show`.

## What does not work yet

Robot descriptions, MuJoCo scenes, motion planning, perception, CAD, printing,
assembly, controllers, experiments, chemistry, real drivers, and the MCP server
are packages only. Do not invent tool names or call hardware SDKs from an
agent turn.

## Files an agent must not change

- `configs/safety/real.yaml` — real-world safety policy.
- `configs/chem/reagents.yaml` — reagent allowlist.

An agent must not approve its own real-world actions. There is no approval
API in this release. Do not send commands to physical robots, printers, or
lab instruments. Examples and tests stay in simulation and do not open devices.

## Units and frames

Use SI units in names (`length_m`, `angle_rad`, `time_s`). The world frame is
`"world"`. A pose passed to `Pose.compose` on the right must use the child
frame `"body"`. Convert scipy quaternions only with
`quat_wxyz_to_xyzw` and `quat_xyzw_to_wxyz`.

## Logging

Use `physlab.core.logging.get_logger(__name__)`. Do not call `print` in library
code. Pass a seed or a `numpy.random.Generator` into any random process.
