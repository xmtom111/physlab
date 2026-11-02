# Assembly

The assembly package loads a plan, checks it against the inventory and the
arm, and runs it on a robot. Every step is written to the event log. The
first failed step stops the run.

## Plan

`AssemblyPlan` has an `id`, a list of `ComponentRef` (`id`, `part_name`,
`aruco_id`), and a list of steps. The step models are a discriminated union
on `op`:

| op | Fields |
| --- | --- |
| `pick` | `component` |
| `place` | `component`, `target` pose, `tol_pos_m`, `tol_rot_rad` |
| `fasten` | `component`, `to` (another component, `baseplate`, `table`, or a `*_carriage`) |
| `align` | `component`, `dof`, `objective`, `max_iters`, `method` |
| `inspect` | `camera`, `check` |

`load_plan(path)` reads the YAML. `validate_plan(plan, inventory, catalog)`
returns a list of issues and does not raise. It reports an unknown part name,
a component the inventory does not hold, a step that names a missing id, a
component placed twice, two place targets closer than 2 cm, and a place target
the xArm cannot reach. The reach check builds a temporary scene and solves
inverse kinematics to a tool-down pose 2 cm above the target. If the arm
description is not on disk, that check is skipped.

`examples/plans/microscope_v0.yaml` is the tabletop microscope: LED, sample
stage, objective, tube, and camera mount, in that order along the bench.

## Pick and place

`pick` stows the arm so the overhead cameras can see the marker, estimates the
part pose, and moves to the grasp. A wrist reading is kept only when it is
within 5 mm of the overhead estimate. The grasp is raised, 4 mm at a time,
until the open gripper is clear of the table, and by another 14 mm when that
pose is still below the tabletop so the closing fingers do not hit the table.
The gripper must report `is_grasping` before the lift. If the lift cannot be
planned directly, `safe_transit` is the fallback. Parts that are not fastened
and are not the one being picked are moved aside for the motion and then put
back, so the arm does not sweep them off the table. When the straight joint
path is blocked, planning is tried once more from a stowed pose with a second
random seed.

`place` allows three attempts. The part is opened over the target. When the
carried pose is more than 1 mm from the target, the free joint is written onto
the target and the arm retreats. The part is then localized again. A miss
outside `tol_pos_m` / `tol_rot_rad` picks the part up and tries again.
`TOLERANCE` is the result after three misses.

## Fastening and focus screws

`SceneSpec.focus_stages` adds a mount, a carriage with a slide and a hinge,
a knob site, and position actuators. Slides quantize to 5 µm and hinges to
0.1 mrad. `FineAlignTool.dock` drives the arm to the knob and succeeds only
when the tool tip is within 1 mm. `adjust` requires a dock. `align` maximizes
a scalar with `coordinate_search` or `nelder_mead`.

`AssemblyExecutor` welds the part to the named body. Before a carriage weld
the slide is set to mid-travel, so the stored offset is not the leftover of
an earlier bump. `SimWorld.project_welds` then places each free body on that
offset when a screw moves, which is what lets the sample track the focus
stage. An offset longer than 5 cm is ignored.

## Running a plan

```python
report = AssemblyExecutor(
    robot,
    localizer,
    inventory,
    event_log,
    world,
    catalog,
    artifacts,
    align_tool=tool,
    microscope=model,
).run(plan)
```

`AssemblyReport` has `success`, `total_time_s`, one `StepRecord` per finished
step, and `failed_step`. `examples/07_assemble_microscope.py` mock-prints the
five parts, runs seed 0, and writes `microscope.png` and `assembly.mp4`.
