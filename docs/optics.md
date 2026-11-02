# Optics

Paraxial helpers live in `physlab.optics.paraxial`. The simulated microscope
reads live body poses and paints a camera frame. Lengths are meters.

## Paraxial

`propagate_matrix` and `thin_lens_matrix` are 2 by 2 ABCD matrices.
`propagate_rays` applies a free-space gap to heights and angles.
`image_distance` and `magnification` use the thin-lens equation. An object on
the focal plane raises `ConfigError`. `afocal_magnification` is
`-tube_lens_f / objective_f`. `blur_circle_diameter` is
`aperture * |defocus| / focal`.

A 0.2 m object distance and a 0.1 m focal length form an image at 0.2 m with
magnification -1.

## Microscope

`MicroscopeModel` is constructed with the objective and tube-lens focal
lengths, a `SensorSpec` (default 160 by 120, 5 µm pixels), and the body names
of the sample stage, objective, tube, camera mount, and LED holder. `render`
returns a `MicroscopeView`. `metrics()` is

`magnification`, `defocus_m`, `lateral_offset_m`, `illumination`.

The bench runs along world X. Defocus is the sample origin Z minus the
objective origin Z. Lateral error is the largest absolute Y mismatch of the
sample, tube, and camera against the objective. Illumination is
`exp(-((led_y/1 cm)^2 + (led_z/1 cm)^2))` relative to the objective, so a
mis-aimed LED darkens the frame.

The picture is `assets/generated/usaf_like_target.png`. The file is generated
on first use and cached. The crop is scaled by the afocal magnification,
shifted by the lateral error, Gaussian-blurred by the geometric blur circle,
and vignetted. Contrast also falls as `1 / (1 + (defocus / 40 µm)^2)`, so
`variance_of_laplacian` has a single peak at focus. An 8-bit Gaussian alone is
not monotonic past about 100 µm, and a focus search would otherwise walk into
that flat region. The lateral shift wraps the frame; the search still sees the
bars.

`examples/07_assemble_microscope.py` saves the focused frame after the sample
screw has maximized that score.
