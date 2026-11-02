# physlab

physlab is a Python framework for running a lab from code: robot arms and mobile
bases, fabricated parts, assembly, and physics or chemistry experiments. The same
interfaces target MuJoCo first and real hardware later.

This release has the core layer, the robot catalog, and a MuJoCo simulation
backend. Frames, poses, configuration, logging, the event log, a typed
registry, and the `physlab` command are implemented. The hardware protocols
are implemented in simulation, including joint-space motion to a pose.
Perception, fabrication, and real hardware backends come later.

## Requirements

- Python 3.11 or newer (developed on 3.12)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-extras --dev
```

The `real` extra pulls hardware SDKs and, through LeRobot, a large PyTorch
install. Core tests do not need it. If that extra fails to install, use:

```bash
uv sync --dev
```

On current uv releases the `dev` group is installed by default, and `--dev` is
accepted as an alias for `--group dev`.

MuJoCo depends on PyOpenGL 3.1.0. The wheel built from the PyPI sdist also
installs a stray `egg-info` file that current uv cannot uninstall, so the next
`uv run` fails. `vendor/pyopengl-3.1.0-py3-none-any.whl` is that wheel with the
extra file removed. The OpenGL sources are unchanged.

## Command line

```bash
uv run physlab --version
uv run physlab config show
uv run physlab assets fetch
uv run physlab robots list
uv run physlab robots check --all
uv run python examples/01_core_poses.py
uv run python examples/02_list_and_check_robots.py
MUJOCO_GL=egl uv run python examples/03_sim_arm_joints.py
MUJOCO_GL=egl uv run python examples/03b_sim_mobile_base.py
uv run python examples/04_pick_cube.py
uv run python examples/04b_mobile_pick.py
uv run python examples/05_locate_markers.py
uv run python examples/06_print_and_pick.py
uv run python examples/07_assemble_microscope.py
```

If EGL cannot load, set `MUJOCO_GL=osmesa` and run the examples again. The
simulation code tries EGL first and falls back to OSMesa on its own when
`MUJOCO_GL` is unset. The videos are written under `run_dir` (default
`runs/`).

`physlab --version` prints `0.1.0`. `physlab config show` prints the YAML
configuration after `PHYSLAB_` environment overrides. Relative paths in
`configs/default.yaml` are resolved from the repository root.

Override a nested field with a double underscore:

```bash
PHYSLAB_SIM__TIMESTEP_S=0.001 uv run physlab config show
```

## Headless rendering

Later simulation tasks render with MuJoCo. Use EGL, and fall back to OSMesa
when EGL is not available:

```bash
export MUJOCO_GL=egl
# if import or a render call reports that EGL is missing:
export MUJOCO_GL=osmesa
```

`configs/default.yaml` stores the same choice in `sim.gl_backend` (`egl`,
`osmesa`, or `glfw`). Offscreen rendering uses that fallback. A window opens
only when `sim.viewer` is true and a display is available. The vendored
PyOpenGL 3.1.0 wheel has no `EGLDeviceEXT`, so EGL fails here and OSMesa is
the backend that actually renders.

## Layout

- `src/physlab/core` — frames, types, errors, config, logging, events, registry
- `src/physlab/robots` — Menagerie fetch, catalog, and composition
- `src/physlab/hal` — arm, gripper, base, and camera protocols
- `src/physlab/sim` — MuJoCo scene, world, backends, and MP4 recorder
- `src/physlab/cli.py` — command line
- `configs/default.yaml` — default configuration
- `examples/01_core_poses.py` — pose composition demo
- `examples/03_sim_arm_joints.py` — joint motion and an MP4
- `examples/04_pick_cube.py` — xArm7 pick and place
- `examples/05_locate_markers.py` — stereo ArUco localization and annotated images
- `docs/motion.md` — inverse kinematics and planning
- `docs/perception.md` — markers, fusion, and focus metrics
- `docs/fabrication.md` — parametric parts, print checks, and printer services
- `docs/assembly.md` — plans, pick/place, fastening, and focus screws
- `docs/optics.md` — paraxial helpers and the simulated microscope
- `examples/06_print_and_pick.py` — mock-print a lens holder and pick it up
- `examples/07_assemble_microscope.py` — assemble and focus the microscope
- `tests/` — pytest suite (`hardware` tests are opted out with `-m "not hardware"`)

The real-hardware package is still empty.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -m "not hardware" --cov=physlab
```

## Safety

Do not edit `configs/safety/real.yaml` or `configs/chem/reagents.yaml` from an
agent session. Those files are human-maintained. This release never commands
real hardware.

## License

MIT. See [LICENSE](LICENSE).
