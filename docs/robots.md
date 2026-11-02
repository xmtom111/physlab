# Robots

The robot package fetches MuJoCo Menagerie and checks a YAML catalog against
those files. Arms, mobile platforms, and two procedural mechanisms share one
`RobotSpec`. Motion planning is not in this layer.

## Fetch

`configs/assets.lock` has `menagerie_commit`. `latest` means the `main` branch.
The fetch clones `https://github.com/google-deepmind/mujoco_menagerie` into
`assets/mujoco_menagerie`. When the lock says `latest`, the resolved hash is
written back. A later run that is already on that hash does not fetch.

```bash
uv run physlab assets fetch
# or
uv run python scripts/fetch_assets.py
```

`git` must be on `PATH`. A missing binary raises `AssetMissing`.

## Catalog

Each file in `configs/robots/<id>.yaml` is one `RobotSpec`. Joint, actuator,
site, and body names are the names in the Menagerie XML at the pinned commit.
If a folder is missing, the file stays in the catalog with `available: false`
and `unavailable_reason` set. None of the listed folders were missing at the
commit fetched for this release.

`RobotCatalog(configs_dir, assets_dir)` loads those files.

- `list()` returns every spec, sorted by id.
- `get(robot_id)` returns one spec or raises `ConfigError`.
- `check(robot_id)` loads the MJCF or builds the procedural spec, compiles it,
  and checks that every listed joint, actuator, site, body, and finger geom
  exists. `home_qpos_rad` must have one entry per arm joint. When `ee_site` is
  omitted and `ee_body` is set, the check records `site will be added at
  runtime` and still passes. `load_robot_spec` adds that site as `physlab_ee`.

```bash
uv run physlab robots list
uv run physlab robots check --all
uv run physlab robots check ufactory_xarm7
```

`robots list` columns are id, kind, available, dof, and gripper. Dof is
`len(arm_joints) + len(base_joints)` and does not count finger joints.

## Grippers and composition

`GripperSpec.attach_from` is a Menagerie folder, used for the UR5e and the
Kinova Gen3 with `robotiq_2f85`. `load_robot_spec` attaches that model at the
arm's `ee_site` with the prefix `robotiq_2f85/`. Actuator and finger-geom names
in the YAML are the names from `2f85.xml`, without the prefix.

`compose_mobile_manipulator(base, arm, name, assets_dir)` mounts the loaded arm
on `base.sim.mount_site` and returns a `RobotSpec` plus the `MjSpec`. It is
checked for `generic_diffdrive` with `trs_so_arm100` and with `ufactory_xarm7`.

## Procedural models

`generic_diffdrive` is a 0.40 x 0.30 x 0.12 m box (MuJoCo geom size is the
half-extent), two wheels of radius 0.05 m with velocity actuators, one passive
caster ball, and a `mount` site on the top center.

`gantry_xyz` is three prismatic joints with travel 0.5 m, 0.5 m, and 0.3 m,
position actuators, and a `tool` site.

## Example

```bash
uv run python examples/02_list_and_check_robots.py
```

The script lists the catalog. When Menagerie is present it checks every
available robot and exits with status 1 if any check fails.

## Limitations

- `lite6.xml` position actuators have no `name` attribute, so
  `arm_actuators` is empty. The six joints are still checked.
- Stretch's home keyframe does not set `qpos`. The catalog uses zeros.
- Open and closed gripper commands are the ctrlrange endpoints. The YAML
  `notes` field records which XML range that choice came from.
- Attaching an arm with an empty prefix keeps joint names stable and makes
  MuJoCo warn that saving the spec may be unreliable.
- Real drivers are names only. Nothing in this package talks to hardware.
- Tests that load Menagerie are marked `assets` and skipped, with the fetch
  command in the message, when `assets/mujoco_menagerie` is absent.
