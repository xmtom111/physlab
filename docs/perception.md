# Perception

The perception package finds ArUco markers, fuses them into a world pose, and
scores image sharpness. Units are SI. Quaternions are `(w, x, y, z)`.

## Markers

`marker_png(marker_id, size_px)` writes a `DICT_4X4_50` image under
`assets/generated` and reuses the file on later calls. Ids are 0 through 49.
The scene builder calls `marker_plate_png`, which adds a white quiet zone so
a dark object does not swallow the black border. The quiet zone is not part
of `aruco_size_m`.

`detect(frame)` runs `cv2.aruco.ArucoDetector`. Corner refinement is
`CORNER_REFINE_APRILTAG`, which holds the border corners on a filtered
texture. Each `Detection` stores the id and four corners in pixels, clockwise
from the marker's top-left.

`estimate_pose(detection, marker_size_m, intrinsics)` returns the marker pose
in the OpenCV camera frame (`frame="camera"`): X right, Y down, Z along the
optical axis. The marker frame has X to the right, Y up the printed face, and
Z out of the face toward the camera. The four object points, in corner order,
are the square

```text
(-s/2, +s/2, 0), (+s/2, +s/2, 0), (+s/2, -s/2, 0), (-s/2, -s/2, 0)
```

with `s = marker_size_m`. `cv2.solvePnPGeneric` uses `SOLVEPNP_IPPE_SQUARE`.
OpenCV 5 can return a tilted pose for a nearly fronto-parallel square and
omit the true pose from the pair. Both solutions are polished with
`cv2.solvePnPRefineLM`, and the pose in front of the camera with the smaller
reprojection error is kept. Distortion is zero.

## Localization

`Localizer(cameras, marker_size_m).locate(marker_id)` returns a
`LocalizedPose` or `None`.

MuJoCo's camera looks along -Z (X right, Y up). The OpenCV camera looks along
+Z (X right, Y down). The conversion is a 180 degree rotation about X, then
`Frame.pose_in_world`. Views whose position is more than 5 mm from the
component-wise median are dropped. The remaining positions are averaged. The
quaternion is the eigenvector of the largest eigenvalue of the sum of outer
products, after flipping signs into one hemisphere.

`refine_with_wrist(camera, marker_id, expected)` returns the wrist measurement
when that camera sees the marker, and `expected` otherwise.

## Focus

`variance_of_laplacian` and `tenengrad` are larger on sharper images. Both
fall as a Gaussian blur grows. `centroid_of_bright_spot` is the
intensity-weighted center of pixels at or above the 90th percentile and at
least 128. It returns `(u, v)` in pixels, or `None`.

## Example

```bash
uv run python examples/05_locate_markers.py
```

The script writes `overhead_left.png` and `overhead_right.png` under
`run_dir/05_locate_markers` with the detected outlines and ids drawn on.

## Limits

Texture filtering eats about a pixel of the black border, so the rendered
square is slightly smaller than `aruco_size_m` and the depth is slightly
long. The stereo test keeps that bias under 2 mm by using a 50 mm marker and
1280 by 960 cameras about 0.2 m above it. A marker that fills only ~100 pixels
can miss the 2 mm budget. `add_wrist_camera` defaults to body `link7`, which
is the xArm7 flange and is wrong for other arms.
