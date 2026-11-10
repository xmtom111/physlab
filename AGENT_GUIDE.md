# Agent guide

This file is for an LLM agent operating physlab through the MCP server.
Read it before calling tools. One server process owns one scene.

## Do not do these things

- Do not edit `configs/safety/real.yaml` or `configs/chem/reagents.yaml`.
  No tool writes those files. `physlab approve` refuses them.
- Do not approve your own real-world action. A human runs
  `physlab approve <plan_or_protocol_file>` and pastes the token.
  The token is required in real mode. This release checks that it is
  present. It does not yet check that the token matches the file (that is
  a later safety gate).
- Do not invent tool names. Call `tools/list` or the resources below.
- Do not send a path to a scene. `load_scene` takes a file stem.

## Units and frames

| Quantity | Unit | Frame |
| --- | --- | --- |
| Robot and part poses | meters | `world`, unless a field says `body` |
| Quaternion | `(w, x, y, z)` | the pose's `frame` |
| Revolute joints | radians | joint |
| Gantry `q_rad` | meters | joint (the name is historical) |
| Yaw | radians | world Z |
| CAD parameters | millimeters | part schema |
| `bbox_m` and part `frames` | meters | part body frame |
| Chemistry concentration | molar | n/a |
| Chemistry volume | microliters | n/a |
| Chemistry temperature | Celsius | n/a |
| Controller channels | the manifest `unit` | the scenario |
| Timeouts and durations | seconds | n/a |

A pose passed into a body-frame composition uses `frame` `"body"`.
Tool poses are otherwise in `"world"`.

## How every tool answers

Inputs are one pydantic object in the `request` field. The schema is the
tool's JSON schema.

Outputs are `{"ok": true, "data": ..., "error": null}` or
`{"ok": false, "data": null, "error": {"type", "message", "details"}}`.
`capture_image` and `design_part` also attach a PNG image block.
Read `summary_text` before a traceback.

Calls that can take longer than two seconds return `{"job_id": "..."}`
immediately. Poll `get_job` until `status` is `done` or `error`.
`progress` is a fraction from 0 to 1. A finished controller job's `result`
is a RunReport: `summary_text`, `phase_failed`, and `plot_images`.

## Resources

- `physlab://robots` — catalog id, kind, dof, whether the mesh is on disk.
- `physlab://parts` — part names and parameter schemas (millimeters).
- `physlab://skills` — experiment skills, motion skills, assembly ops.
- `physlab://runs/{run_id}/report` — Markdown written by `get_report`.
- `physlab://docs/agent_guide` — this file.

## Scenes

`list_scenes` then `load_scene`. Shipped stems:

- `table_gantry` — procedural XYZ gantry. No Menagerie download. No gripper.
  Joint targets are meters.
- `table_arm` — xArm7, overhead stereo, wrist camera. Needs
  `physlab assets fetch`. This is the scene that can pick and place.

`load_scene` drops spawned parts. Designs already built stay on disk.
`reset_scene` restores the current scene and takes a `seed`.

## Build an apparatus

1. `list_parts` and read the schema.
2. `design_part`. Keep `design_id`. Look at the preview PNG and `frames`
   (`grasp`, `mount`, sometimes `optical_axis`) in the part body frame.
3. `check_part`. If `ok` is false, read `issues` and change millimeters,
   not the printer profile.
4. `print_part`, then poll `get_job`. Simulation finishes on the next
   status read and spawns the mesh.
5. `list_inventory`. Copy `aruco_id` into the assembly plan. The pose is
   the part origin in the world frame.
6. `validate_assembly_plan`. Fix every issue before running.
7. `run_assembly_plan` on `table_arm`, then poll. The report's
   `failed_step` says which step stopped. A perception failure means the
   marker was not seen: check the camera with `capture_image` and
   `locate_marker`.

Place targets are meters in the world frame. Do not place two parts within
2 cm of each other.

## Write a controller

1. `get_device_manifest`. Use only those sensor and actuator names.
   Obey each channel's unit and range.
2. Subclass `Controller` from `physlab.controllers.api`. Allowed imports are
   `math`, `numpy`, `dataclasses`, `typing`, `collections`, and that API
   module. `open`, `exec`, and `eval` are rejected.
3. `check_controller_code` before a run.
4. `run_controller`, then poll. Read `summary_text` first.
   - `phase_failed` `CHECK`: the source broke the allowlist. The line is in
     the summary. Fix the import or the call and run again.
   - `IMPORT` or `SETUP`: the class is missing or `setup` raised.
   - `STEP`: `step` raised. The traceback tail is in the summary.
   - `TIMEOUT`: a step or the process exceeded its wall clock.
   - `ok` false with assertions: `assert_that` messages name the time.
   - Clipping: the command left the manifest range. Lower the gain.
5. Plot images are PNG base64 on the job result. The sign of a feedback
   law is the usual bug: if the error grows, flip the sign and run again.

Scenarios: `focus_stage` (meters and a dimensionless sharpness),
`pendulum_release` (radians), `line_follow` (dimensionless photos and
rad/s wheel speeds).

## Run an experiment

1. `validate_protocol`. Skills must be names from `physlab://skills`.
   Unknown skills are issues.
2. `run_experiment`, then poll. Failures stay in the trial table.
3. `fit_models` with the independent column `x` and the measurement `y`.
   The list is sorted by AIC. The lowest AIC is the best of the six
   shapes on this sample. It is not a proof of the law.
4. `get_report`. Then read `physlab://runs/{run_id}/report`.
   If the exponent or the recovered constant misses the bar in the report,
   check units before changing the hypothesis. A column in millimeters
   fitted as meters will look like a bad law.

## Chemistry

1. Use only reagents from the allowlist the safety check names. Do not
   add reagents by editing the config.
2. Concentrations are molar. Volumes are microliters. Temperature is Celsius.
3. `check_chem_protocol`. Read `issues` and `safety.violations`.
   Real mode without a token is a denial you can read there.
4. `run_chem_protocol`, then poll. The result `run_id` is a table of
   readings. `fit_models` can take `concentration_m` and `absorbance`,
   or the pH column from a titration.
5. A denial for an unknown reagent, a concentration over the cap, or an
   incompatible mix is final. Change the protocol. Do not change the
   allowlist.

## Real mode

Hardware and liquid tools require `approval_token`: `reset_scene`,
`move_arm_joints`, `move_arm_pose`, `gripper`, `move_base`, `print_part`,
`run_assembly_plan`, `run_controller`, `run_experiment`, and
`run_chem_protocol`. `stop_all` does not. Call `stop_all` when a motion
should end.

This release has no real drivers. A present token still returns
`DependencyMissing` for those tools. Stay in `mode: sim` until a human
installs drivers.

## Failure recovery

- `ApprovalRequired`: ask a human for `physlab approve`. Do not mint a token.
- `PlanningFailed` or a motion `success` of false: read `reason`.
  `IK_FAILED` means the pose is out of reach. Move the target closer to
  the robot, in meters, in the world frame. `COLLISION` means a straight
  path hits something. Raise the pose and try again.
- `PerceptionFailed` or `found: false`: capture the camera. Confirm the
  marker id and that `marker_size_m` is 0.03 for a printed part.
- `ConfigError` on a scene: call `list_scenes` and use a stem.
- A job `status` of `error`: the `error` object has the same shape as a
  tool error. Fix the input and submit a new job. Job ids are not reused.
