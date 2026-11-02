"""Locate two table markers and save the annotated stereo views."""

from __future__ import annotations

import cv2
import numpy as np

from physlab.core.config import load_config
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import Pose
from physlab.hal.factory import create_camera
from physlab.hal.interfaces import Frame
from physlab.perception.aruco import Detection, detect
from physlab.perception.localize import Localizer
from physlab.robots.catalog import RobotCatalog
from physlab.sim.scene import (
    ObjectSpec,
    PrimitiveKind,
    PrimitiveSpec,
    SceneSpec,
    TableSpec,
    add_overhead_stereo,
)
from physlab.sim.world import SimWorld

_MARKER_M = 0.05


def _marker(name: str, marker_id: int, x_m: float, y_m: float) -> ObjectSpec:
    """A dark box with a marker on its top face.

    Args:
        name: Body name.
        marker_id: ``DICT_4X4_50`` id.
        x_m: World X, meters.
        y_m: World Y, meters.

    Returns:
        Object spec sitting on the default table.
    """
    return ObjectSpec(
        name=name,
        collision=[PrimitiveSpec(type=PrimitiveKind.BOX, size_m=(0.08, 0.08, 0.02))],
        pose=Pose(position_m=(x_m, y_m, 0.76), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        mass_kg=0.1,
        rgba=(0.15, 0.15, 0.15, 1.0),
        aruco_id=marker_id,
        aruco_size_m=_MARKER_M,
    )


def _annotate(frame: Frame, detections: list[Detection]) -> np.ndarray:
    """Draw marker outlines and ids on a BGR image.

    Args:
        frame: RGB capture.
        detections: Markers found in ``frame``.

    Returns:
        BGR image suitable for ``cv2.imwrite``.
    """
    image = cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2BGR)
    for detection in detections:
        corners = np.asarray(detection.corners_px, dtype=np.int32).reshape(1, 4, 2)
        cv2.polylines(image, corners, isClosed=True, color=(0, 220, 0), thickness=2)
        origin = (int(detection.corners_px[0][0]), int(detection.corners_px[0][1]) - 6)
        cv2.putText(
            image,
            str(detection.id),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    """Build a stereo scene, localize both markers, and write the views."""
    configure_logging("INFO", False)
    log = get_logger("examples.05_locate_markers")
    loaded = load_config()
    scene = SceneSpec(
        table=TableSpec(),
        objects=[_marker("left_tag", 7, -0.05, 0.02), _marker("right_tag", 8, 0.06, -0.03)],
    )
    add_overhead_stereo(scene, height_m=0.98, baseline_m=0.08, width=1280, height=960)
    catalog = RobotCatalog(loaded.configs_dir, loaded.assets_dir)
    world = SimWorld(scene, catalog, loaded)
    cameras = [create_camera(world, name) for name in ("overhead_left", "overhead_right")]
    localizer = Localizer(cameras, _MARKER_M)
    directory = loaded.run_dir / "05_locate_markers"
    directory.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        frame = camera.capture()
        image = _annotate(frame, detect(frame))
        path = directory / f"{camera.name}.png"
        if not cv2.imwrite(str(path), image):
            raise SystemExit(f"could not write {path}")
        log.info("annotated_image", camera=camera.name, path=str(path), bytes=path.stat().st_size)
    for marker_id in (7, 8):
        found = localizer.locate(marker_id)
        if found is None:
            raise SystemExit(f"marker {marker_id} was not localized")
        log.info(
            "localized",
            marker_id=marker_id,
            n_views=found.n_views,
            position_m=found.pose_world.position_m,
            residual_m=found.residual_m,
        )


if __name__ == "__main__":
    main()
