# Simulation

The simulation package compiles a `SceneSpec` to MuJoCo, steps it, and exposes
the HAL devices. Units are SI. Quaternions are `(w, x, y, z)`.

## Scene

`SceneSpec` holds an optional `TableSpec` (default top 1.2 m by 0.8 m at
0.75 m), a floor size, robot placements, objects, and cameras.

`ObjectSpec` collision is primitives only: box, cylinder, sphere, or capsule.
`size_m` is the full box edge lengths, a sphere radius, or `(radius, height)`
for a cylinder or capsule. The builder converts those to MuJoCo half extents.
A `visual_mesh` is loaded with `contype` and `conaffinity` 0. MuJoCo replaces
a mesh collider with its convex hull, which fills holes, so meshes never
collide.

`build_mjspec(scene, catalog)` returns an `MjSpec`. Every element that comes
from a robot is prefixed with `"{instance}/"`. A differential-drive placement
also gets a free joint named `"{instance}_base_free"` on the anchor. That
joint is not part of the catalog.

`CameraSpec` is fixed in the world (`pose`) or bolted to a body
(`attached_to_body`). The offscreen framebuffer is at least 640 by 480 and at
least as large as the largest camera.

`RobotPlacement.arm_id`, when set, mounts that catalog arm on the base's
mount site before the combined model is placed. Joint names keep the
instance prefix.

## World

`SimWorld(scene, catalog, config)` owns the compiled `MjModel`, `MjData`, and
a reentrant lock.

| Method | Behavior |
| --- | --- |
| `step(n=1)` | Advance `n` integrator steps of `sim.timestep_s`. |
| `run_until(predicate, timeout_s)` | Step until the predicate is true. Returns whether it was. |
| `reset(seed=None)` | Restore the initial joint state. Replaces the numpy Generator when `seed` is set. |
| `time_s` | Simulation clock, seconds. |
| `add_object_at_runtime(obj)` | Recompile and keep qpos/qvel of joints that already exist, matched by name. |
| `weld(body_a, body_b)` / `unweld` | Equality weld of the current relative pose, then recompile. |
| `contacts()` | `Contact` rows for the live contact buffer. |
| `copy_data()` | A new `MjData`. Planning must not write the live buffer. |

`config.sim.viewer` opens `mujoco.viewer.launch_passive` only when it is true
and `DISPLAY` is set. A missing viewer extra is logged and ignored.

## Devices

`SimArm.move_joints` follows a minimum-jerk curve. Duration is the longest
joint travel divided by `joint_velocity_max_rad_s * speed_scale`. An empty
limit tuple means 1.0 rad/s. After the curve, the controller holds the target
and adds `qfrc_bias / kp` so gravity does not leave a steady-state error.
Success is `max |q - q_target| < 2e-3` rad and `max |qdot| < 1e-2`. A new
contact between the robot and a body that is not a finger-grasped object
returns `FailureReason.COLLISION`. Contacts already present when the motion
starts (the base resting on the table) are ignored. A target outside
`jnt_range` returns `JOINT_LIMIT` without stepping.

`SimGripper` interpolates `ctrl` between `open_ctrl` and `closed_ctrl`.
`is_grasping` is true when a left finger geom and a right finger geom both
touch one non-robot body.

`SimMobileBase` for `diff_drive` converts `(vx, wz)` to left and right wheel
rates and integrates the anchor free joint. `vy` is ignored. `move_to` is
rotate, drive, rotate with proportional clips, and it stops inside half of
the requested tolerance. Wheel actuator gains are set to zero at bind time:
the catalog gain that keeps a sphere-wheel contact stable still diverges when
the commanded rate jumps, and the acceptance check is the planar pose.
`holonomic_joints` servos the base joints to the integrated or absolute
target.

`SimCamera` renders RGB and depth in meters with MuJoCo's classic `Renderer`.
Intrinsics use a square pixel and the vertical field of view:
`fy = 0.5 * height / tan(fovy/2)`, `fx = fy`, principal point at the image
center.

`Recorder(world, camera_name, fps)` writes an H.264 MP4. The default path is
`run_dir / "{camera_name}.mp4"`. `capture` samples when the simulation clock
crosses the next frame.

## Headless GL

`physlab.sim.gl.renderer_class` tries `MUJOCO_GL` if it is set, otherwise
`egl`, then `osmesa`. The vendored PyOpenGL 3.1.0 wheel has no
`EGLDeviceEXT`, so EGL fails in this environment and OSMesa renders. Prefer
`MUJOCO_GL=egl` and leave the fallback in place when EGL is missing.

## Examples

```bash
uv run python examples/03_sim_arm_joints.py
uv run python examples/03b_sim_mobile_base.py
```

The arm example needs Menagerie (`physlab assets fetch`). Both write an MP4
under the configured `run_dir`.

## Limitations

- `move_pose` plans in joint space. See `docs/motion.md`.
- Differential drive is kinematic on the anchor free joint. Wheel `ctrl` is
  written, but the actuator gain is zeroed so the contact solver does not
  diverge.
- A passive viewer needs a display and the `mujoco.viewer` extra.
- Attaching a Menagerie arm warns that the child integrator differs. The
  parent value is kept.
