# physlab

physlab is a Python framework for running a lab from code: robot arms and mobile
bases, fabricated parts, assembly, and physics or chemistry experiments. The same
interfaces target MuJoCo first and real hardware later.

This release is the core layer only. Frames, poses, configuration, logging, the
event log, a typed registry, and the `physlab` command are implemented. Robot
models, simulation, motion, perception, fabrication, and hardware backends are
packages reserved for later work.

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
uv run python examples/01_core_poses.py
```

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
`osmesa`, or `glfw`). Nothing in this release opens a window or a GL context.

## Layout

- `src/physlab/core` — frames, types, errors, config, logging, events, registry
- `src/physlab/cli.py` — command line
- `configs/default.yaml` — default configuration
- `examples/01_core_poses.py` — pose composition demo
- `tests/` — pytest suite (`hardware` tests are opted out with `-m "not hardware"`)

Other packages under `src/physlab` are empty on purpose.

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
