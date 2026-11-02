# Fabrication

The fabrication package builds parametric parts, checks that an STL can be
printed, and submits the mesh to a print service. CAD units are millimeters.
`PartArtifact` lengths, frames, and collision primitives are meters. The part
origin is the center of the bottom face, Z up.

## Parts

`list_parts()` returns each library name and the JSON schema of its
parameters. `build_part(name, params)` validates those parameters and writes
an STL and a STEP file. `PART_LIBRARY` maps a name to `(params model, builder)`.

Every part has two finger notches on the ±Y faces and a shallow ArUco recess.
The `grasp` frame is the xArm tool pose that pinches the notch band with tool
+Z down. `mount` is the bottom center. Lens, tube, and LED parts also publish
`optical_axis`.

| Name | Defaults |
| --- | --- |
| `magnetic_base` | diameter 40 mm, height 12 mm, magnet 10 × 3 mm |
| `lens_holder` | lens diameter 16 mm, thickness 5 mm, axis height 24 mm |
| `tube` | inner diameter 18 mm, length 40 mm, wall 2 mm |
| `sample_stage` | 60 mm square, 10 mm travel well |
| `camera_mount` | `board="generic_32mm"` |
| `led_holder` | LED diameter 5 mm |
| `rail_baseplate` | length 140 mm, peg-slot pitch 20 mm |
| `pendulum_kit` | arm length 80 mm |
| `incline_ramp` | 15 degrees, height `100 mm * tan(angle)` |

Defaults fit a 220 × 220 × 250 mm bed.

## Print check

`PrinterProfile` is `bed_size_mm`, `max_height_mm`, `nozzle_mm`, and
`layer_mm`. `configs/printers/profile.yaml` is the 220 × 220 × 250 mm profile.
`check_printable(stl, profile)` uses trimesh. `ok` requires a watertight mesh,
extents inside the bed and height limit, and an estimated minimum feature at
least as large as the nozzle. `max_overhang_ratio` is 0 for a vertical wall
and 1 for a horizontal underside that is not on the bed; gripper notches score
near 1 and do not by themselves fail the check.

## Services

`PrintService.submit` returns a job id. `status` returns `QUEUED`, `SLICING`,
`PRINTING`, `DONE`, `FAILED`, or `CANCELLED`. `cancel` requests a stop.

`MockPrintService` estimates duration as `volume_mm3 * copies / (mm3_per_s *
speed_factor)` and reads a simulated clock. On `DONE` it records an
`Inventory` row. When a `SimWorld` is attached, it spawns the part at the next
free pose in `print_pickup_zone`, with the STL scaled to meters, the part's
collision primitives, and an ArUco id.

`McpPrintService` starts the server in `configs/printers/printer_mcp.yaml`
with the official MCP Python SDK (`stdio_client` and `ClientSession`). Tool
names, argument names, and result keys come only from that file. On connect
it calls `list_tools`. A mapped name that the server does not advertise raises
`ConfigError` and the message lists the server's actual tool names. `slice`
may be null. `physlab printer tools` prints the live tool names and input
schemas. `physlab printer submit <part> --params JSON` builds, checks, and
submits.

`fabricate(part_name, params, service, printer_profile)` builds, stops with
`issues` when the mesh check fails, otherwise submits and polls until the job
is terminal.

## Example

```bash
uv run python examples/06_print_and_pick.py
```

The script mock-prints the default lens holder into the pickup zone at
`(0.48, 0, 0.75)` m and the xArm7 lifts it.

## Limits

The default MCP mapping has null tool names until `physlab printer tools` is
run against a live printer server and the YAML is filled in. OpenCV is not
involved. The simulated mass is capped at 50 g so the arm can lift it. The
ArUco plate added by the scene builder sits on the collision top, not inside
the modeled recess.
