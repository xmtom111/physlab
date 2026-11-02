# Changelog

All notable changes to physlab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-02

### Added

- Repository scaffold for the physlab package (`src` layout, configs, tests, examples, docs).
- Core layer: quaternion and pose math, shared types, error hierarchy, YAML configuration
  with `PHYSLAB_` environment overrides, structlog setup, JSONL event log, and a typed registry.
- Command line entry points `physlab --version` and `physlab config show`.
- Vendored PyOpenGL 3.1.0 wheel with the stray egg-info file removed so `uv run` can sync.
- MuJoCo Menagerie fetch (`physlab assets fetch`) and a robot catalog for the
  listed arms, mobile platforms, and procedural diff-drive and gantry
  (`physlab robots list`, `physlab robots check`).
- Hardware protocols for an arm, gripper, mobile base, and camera, plus a
  MuJoCo scene builder, stepping world, and simulation backends. Joint motion
  uses a minimum-jerk path. `move_pose` raises `NotImplementedError` until
  inverse kinematics. `examples/03_sim_arm_joints.py` and
  `examples/03b_sim_mobile_base.py` record MP4s.
- `SimConfig.viewer` and `SimSpec.joint_velocity_max_rad_s`. Both default so
  existing YAML still loads. An empty velocity tuple means 1.0 rad/s per joint.
- Motion layer: mink inverse kinematics, joint-space planning (straight line
  or RRT-Connect), trapezoidal time scaling, `SimArm.move_pose`, and the
  transit, approach, retreat, and navigate-and-reach skills.
  `examples/04_pick_cube.py` and `examples/04b_mobile_pick.py` run seeds 0
  through 4.
- Perception: ArUco detection and `SOLVEPNP_IPPE_SQUARE` pose estimation,
  stereo fusion, wrist refinement, and Laplacian / Tenengrad focus scores.
  An object with `aruco_id` gets a textured marker plate (`aruco_size_m`,
  default 0.03 m). `examples/05_locate_markers.py` saves annotated images.
