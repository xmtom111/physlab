# Motion

Motion turns a pose goal into a joint trajectory and runs it on the HAL.
Inverse kinematics uses mink on a copy of `MjData`. The live simulation is
only stepped while a trajectory is executing.

## Inverse kinematics

`solve_ik(world, instance, target, ...)` returns an `IKResult`.

The quadratic program is mink 1.4.0:

- `FrameTask` on the end-effector site (`frame_type="site"`).
- `ConfigurationLimit` on the model.
- `DofFreezingTask` as an equality so only the arm joints move.
- `CollisionAvoidanceLimit` when `avoid_collisions=True`, between named robot
  geoms and the rest of the scene.
- `solve_ik(..., solver="daqp")`. `daqp` is the QP solver installed with
  qpsolvers in this environment. The returned value is a tangent velocity;
  `Configuration.integrate_inplace` applies it.

The first attempt starts at `seed_q` or the current arm position. Up to five
more attempts draw seeds from `numpy.random.default_rng(world.config.seed)`.
Success is a world-frame translation under `pos_tol_m` (default 1 mm) and,
unless `position_only` is set, a rotation under `rot_tol_rad` (default 0.01).

## Planning

`check_path_collision` samples a joint path so the largest joint step is at
most one degree and reports the first waypoint index that touches a body
outside `allowed_bodies`. Checks use a copied `MjData`.

`plan_joint_path` returns the straight line when that check is clear. Bodies
already touching the robot at the start are allowed, so the table under the
base is not an obstacle. Otherwise it runs RRT-Connect in joint space: at
most 5000 samples, seeded from `config.seed`, with shortcut smoothing.
Failure raises `PlanningFailed`.

`plan_cartesian_line` walks a straight translation and a slerp in steps of
`step_m` (default 5 mm). Each step is an IK solve seeded from the previous
joints. A joint jump above 0.3 rad raises `PlanningFailed`.

## Time scaling and execution

`time_parameterize(waypoints, vel_limits, acc_limits, speed_scale)` builds a
`Trajectory`. Every joint shares one trapezoidal path parameter. The segment
time is set by the slowest joint after scaling both limits by `speed_scale`.
`sample(t)` clamps outside the duration.

`SimArm.move_pose` solves IK, plans a joint path, time-scales it with an
acceleration limit of 4 rad/s², and tracks it. A new contact between an arm
body and a non-allowed body aborts with `COLLISION`. Base bodies (chassis,
wheels, caster, anchor) do not abort an arm move.

## Skills

These call only the HAL.

- `safe_transit(robot, target, clearance_z_m)` lifts to just above the
  clearance, moves across, then descends. Waypoints are 2 cm apart. The
  commanded height is 2 mm above `clearance_z_m` so tracking error does not
  dip below the clearance.
- `approach(robot, target, standoff_m)` backs up along the tool -Z axis, then
  moves to the target. `retreat(robot, distance_m)` backs up from the current
  pose along that same axis.
- `navigate_and_reach(robot, base_goal, ee_target)` drives the base, then
  moves the arm.

## Examples

```bash
uv run python examples/04_pick_cube.py
uv run python examples/04b_mobile_pick.py
```

Both loop seeds 0 through 4. The xArm7 example grasps a 4 cm cube, lifts it,
and sets it down 20 cm away. The mobile example mounts `trs_so_arm100` on
`generic_diffdrive` with `RobotPlacement.arm_id`, drives 1.5 m, and closes
the gripper on the cube at the end effector.

## Limitations

- RRT-Connect ignores self-collision. It rejects contacts with scene bodies
  that are not in the allowed set.
- The SO-ARM jaw is not a parallel pinch at the home posture, so the mobile
  example closes on a cube spawned at the site instead of lifting it out of
  a bin. A weld to that jaw is unstable while the diff-drive pose is held
  kinematically.
- `navigate_and_reach` uses the base's default position tolerance.
- `move_pose` treats the pose as reached when the end effector is within
  5 mm and 0.05 rad after the trajectory, even if a soft actuator has not
  quite met the 2e-3 rad joint settle used by `move_joints`.
