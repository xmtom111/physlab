# Core

The core layer is the shared language for the rest of physlab: rigid poses,
configuration, structured logs, a run event log, and a name-to-factory
registry. Later tasks depend on these types and do not reimplement them.

## Units and frames

- Lengths are meters, angles are radians, time is seconds.
- Quaternions are `(w, x, y, z)`, the MuJoCo order.
- scipy's `Rotation` uses `(x, y, z, w)`. Convert only with
  `quat_wxyz_to_xyzw` and `quat_xyzw_to_wxyz`.
- Every pose names its reference frame. The default name is `world`.
- `Pose` stores the transform that maps a point in the body frame into
  `pose.frame`: `p_frame = T @ p_body`.

`quat_normalize` returns a unit quaternion whose first non-zero component is
positive, so `q` and `-q` become the same value. `rpy_to_quat` uses intrinsic
XYZ: roll about X, then pitch about the new Y, then yaw about the new Z.

## Child-frame convention

`pose.compose(other)` computes `T_pose @ T_other`.

- `pose` maps its body into `pose.frame`.
- `other` must map a further body into `pose`'s body. Mark that by setting
  `other.frame` to `CHILD_FRAME` (`"body"`).
- The result keeps `pose.frame`.
- `pose.inverse()` is tagged with `"body"`, so `pose.compose(pose.inverse())`
  is the identity in `pose.frame`.
- `with_offset(dx_m, dy_m, dz_m)` moves the origin by a vector expressed in
  the body frame. The orientation does not change.
- `error_to(other)` returns `(translation_m, rotation_rad)` and requires both
  poses to share a frame.

## Public API

Import from `physlab.core` or from the submodule that owns the name.

| Area | Names |
| --- | --- |
| Frames | `quat_wxyz_to_xyzw`, `quat_xyzw_to_wxyz`, `quat_normalize`, `quat_multiply`, `quat_to_matrix`, `matrix_to_quat`, `rpy_to_quat`, `quat_angle_between` |
| Types | `Pose`, `Pose2D`, `JointState`, `MotionResult`, `FailureReason`, `Mode`, `CHILD_FRAME` |
| Errors | `PhysLabError`, `ConfigError`, `DependencyMissing`, `AssetMissing`, `SafetyViolation`, `PlanningFailed`, `PerceptionFailed`, `HardwareError`, `ApprovalRequired`, `CodeRejected` |
| Config | `PhysLabConfig`, `SimConfig`, `GlBackend`, `load_config` |
| Logging | `configure_logging`, `get_logger` |
| Events | `Event`, `EventLog`, `Actor`, `new_run_id` |
| Registry | `Registry` |

Pydantic models are frozen and reject unknown fields. `Pose.quat_wxyz` is
normalized on input and a zero norm raises `PhysLabError`.

`load_config(path=None)` reads YAML, applies `PHYSLAB_` overrides, and returns
absolute paths. Nested overrides use `__`, for example
`PHYSLAB_SIM__TIMESTEP_S`. `sim.gl_backend` is `egl`, `osmesa`, or `glfw`.
`sim.viewer` defaults to false. A passive MuJoCo viewer is opened only when
that flag is true and `DISPLAY` is set.

`configure_logging(level, json)` configures structlog for the process. Pass
`json=True` for one JSON object per line. `get_logger(name)` binds `logger`
to `name`.

`EventLog(run_dir)` writes `run_dir/<run_id>/events.jsonl`. A run id is
`YYYYMMDD-HHMMSS-<6 hex>`. Pass `seed` for a reproducible suffix; otherwise
the suffix comes from OS entropy. `record` appends one event and `read_all`
returns the events in order. `actor` is `agent`, `human`, or `system`.

`Registry.register` rejects a duplicate name with `ConfigError`. `create`
forwards keyword arguments to the factory. `names` returns registration order.

The command line is `physlab --version` and `physlab config show`. The show
command writes YAML to stdout and does not go through the logger.

## Example

`examples/01_core_poses.py` builds a wrist pose from roll-pitch-yaw, composes
a 12 cm tool expressed in the body frame, shifts that tool by 1 cm along its
X axis, and checks that compose-then-inverse is the identity within `1e-9`.

```bash
uv run python examples/01_core_poses.py
```

A shorter fragment:

```python
from physlab.core.frames import rpy_to_quat
from physlab.core.types import CHILD_FRAME, Pose

wrist = Pose(
    position_m=(0.45, -0.10, 0.25),
    quat_wxyz=tuple(float(v) for v in rpy_to_quat(0.0, 0.15, 0.40)),
    frame="world",
)
tool = Pose(position_m=(0.0, 0.0, 0.12), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame=CHILD_FRAME)
tool_in_world = wrist.compose(tool)
```

## Limitations

- No global config object. Callers pass `PhysLabConfig` into later objects.
  Logging is the exception: structlog's configuration is process-global, and
  `configure_logging` is the only writer.
- `EventLog` run ids are not seeded unless the caller passes `seed`. The
  unseeded suffix is OS entropy so two runs in the same second stay distinct.
- Default config discovery walks from the working directory and then the
  source checkout (`src/physlab/core/config.py`). A non-editable install that
  is launched outside the repository needs an explicit path.
- `DependencyMissing` and the other unused error subclasses are part of the
  hierarchy. Nothing in this release raises them yet.
- `Registry.create` accepts arbitrary keyword arguments. The `Any` on
  `**kwargs` is intentional because the factories do not share one signature.
- Event `params` and `result` are open JSON objects (`dict[str, Any]`).
- Robot assets, MuJoCo, and real drivers are not loaded here. `sim.gl_backend`
  is stored only. Set `MUJOCO_GL` in the environment when rendering lands.
