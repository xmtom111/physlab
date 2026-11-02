"""Gripper notches, ArUco recesses, and part-frame helpers.

build123d solids use millimeters. Poses and collision sizes use meters. The
part origin is the center of the bottom face, Z up.
"""

from __future__ import annotations

import math
from pathlib import Path

import build123d as bd

from physlab.core.errors import AssetMissing
from physlab.core.frames import rpy_to_quat
from physlab.core.types import Pose
from physlab.fabrication.models import PartArtifact
from physlab.sim.scene import PrimitiveKind, PrimitiveSpec

# Tool +Z points down. The xArm TCP that pinched a 40 mm cube sat 25 mm below
# the pinch center, so the grasp frame uses that same offset.
_PINCH_TO_TCP_M = 0.025
_IDENTITY = (1.0, 0.0, 0.0, 0.0)
_DOWN = (0.0, 1.0, 0.0, 0.0)
_CENTER_MIN = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)


def centered_box(length_mm: float, width_mm: float, height_mm: float) -> bd.Part:
    """Return a box whose bottom center is the origin.

    Args:
        length_mm: Size along X, millimeters.
        width_mm: Size along Y, millimeters.
        height_mm: Size along Z, millimeters.

    Returns:
        Solid box.
    """
    return bd.Box(length_mm, width_mm, height_mm, align=_CENTER_MIN)


def apply_casing(
    solid: bd.Part,
    length_mm: float,
    width_mm: float,
    height_mm: float,
    *,
    marker: str,
) -> tuple[bd.Part, Pose, float]:
    """Cut finger notches and an ArUco recess into ``solid``.

    Notches open on the +Y and -Y faces, which is the xArm finger direction
    when the tool +Z axis points down.

    Args:
        solid: Body before the standard cuts.
        length_mm: Outer size along X.
        width_mm: Outer size along Y.
        height_mm: Outer size along Z.
        marker: ``"top"`` or ``"side"`` (the +X face).

    Returns:
        The cut solid, the recess pose, and the notch center height in
        millimeters.
    """
    notched, notch_z = _notches(solid, length_mm, width_mm, height_mm)
    finished, face = _marker(notched, length_mm, width_mm, height_mm, marker)
    return finished, face, notch_z


def standard_frames(notch_z_mm: float) -> dict[str, Pose]:
    """Return grasp and mount frames for a casing.

    Args:
        notch_z_mm: Height of the finger-notch center, millimeters.

    Returns:
        ``grasp`` at the xArm TCP and ``mount`` on the bottom center.
    """
    grasp_z = notch_z_mm / 1000.0 - _PINCH_TO_TCP_M
    return {
        "grasp": _pose((0.0, 0.0, grasp_z), _DOWN),
        "mount": _pose((0.0, 0.0, 0.0), _IDENTITY),
    }


def optical_axis(origin_mm: tuple[float, float, float]) -> Pose:
    """Return a frame whose +Z axis follows the part +X axis.

    Args:
        origin_mm: Frame origin, millimeters.

    Returns:
        Pose in the part frame.
    """
    return _pose(_meters(origin_mm), _quat(0.0, math.pi / 2.0, 0.0))


def box_primitive(length_mm: float, width_mm: float, height_mm: float) -> PrimitiveSpec:
    """Return a box collider matching an outer size.

    Args:
        length_mm: Full X size.
        width_mm: Full Y size.
        height_mm: Full Z size.

    Returns:
        Primitive centered on the part's vertical axis.
    """
    return PrimitiveSpec(
        type=PrimitiveKind.BOX,
        size_m=_meters((length_mm, width_mm, height_mm)),
        pose=_pose((0.0, 0.0, height_mm / 2000.0), _IDENTITY),
    )


def cylinder_primitive(radius_mm: float, height_mm: float) -> PrimitiveSpec:
    """Return a vertical cylinder collider.

    Args:
        radius_mm: Radius, millimeters.
        height_mm: Height, millimeters.

    Returns:
        Primitive whose bottom is on the part origin plane.
    """
    return PrimitiveSpec(
        type=PrimitiveKind.CYLINDER,
        size_m=(radius_mm / 1000.0, height_mm / 1000.0),
        pose=_pose((0.0, 0.0, height_mm / 2000.0), _IDENTITY),
    )


def export_part(solid: bd.Part, stl_path: Path, step_path: Path) -> None:
    """Write an STL and a STEP file for ``solid``.

    Args:
        solid: Millimeter solid.
        stl_path: STL destination.
        step_path: STEP destination.

    Raises:
        AssetMissing: An exporter returns false.
    """
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    stl_ok = bd.export_stl(solid, stl_path, tolerance=0.1, angular_tolerance=0.4)
    step_ok = bd.export_step(solid, step_path)
    if not stl_ok or not step_ok:
        raise AssetMissing(
            "CAD export failed",
            details={"stl": str(stl_path), "step": str(step_path)},
        )


def bbox_m(solid: bd.Part) -> tuple[float, float, float]:
    """Return the axis-aligned size of ``solid`` in meters.

    Args:
        solid: Millimeter solid.

    Returns:
        ``(x, y, z)`` meters.
    """
    box = solid.bounding_box()
    return (
        float(box.max.X - box.min.X) / 1000.0,
        float(box.max.Y - box.min.Y) / 1000.0,
        float(box.max.Z - box.min.Z) / 1000.0,
    )


def make_artifact(
    name: str,
    params: dict[str, float | int | str],
    solid: bd.Part,
    directory: Path,
    primitives: list[PrimitiveSpec],
    frames: dict[str, Pose],
    aruco_face: Pose,
) -> PartArtifact:
    """Export ``solid`` and pack a :class:`PartArtifact`.

    Args:
        name: Library name.
        params: Dumped parameter values.
        solid: Final millimeter solid.
        directory: Directory for the STL and STEP files.
        primitives: Collision primitives in meters.
        frames: Mating frames.
        aruco_face: Marker recess pose.

    Returns:
        Artifact pointing at the written files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stl_path = directory / f"{name}.stl"
    step_path = directory / f"{name}.step"
    export_part(solid, stl_path, step_path)
    return PartArtifact(
        name=name,
        params=params,
        stl_path=stl_path,
        step_path=step_path,
        bbox_m=bbox_m(solid),
        collision_primitives=primitives,
        frames=frames,
        aruco_face=aruco_face,
    )


def _notches(
    solid: bd.Part,
    length_mm: float,
    width_mm: float,
    height_mm: float,
) -> tuple[bd.Part, float]:
    notch_h = min(14.0, max(4.0, height_mm * 0.45))
    notch_z = height_mm * 0.5
    if notch_z + notch_h / 2.0 > height_mm - 0.2:
        notch_z = max(notch_h / 2.0, height_mm - notch_h / 2.0 - 0.2)
    depth = min(6.0, max(2.0, width_mm * 0.15))
    span = min(18.0, max(8.0, length_mm * 0.55))
    reach = depth + 2.0
    positive = bd.Pos(0, width_mm / 2.0, notch_z) * bd.Box(span, reach, notch_h)
    negative = bd.Pos(0, -width_mm / 2.0, notch_z) * bd.Box(span, reach, notch_h)
    return solid - positive - negative, notch_z


def _marker(
    solid: bd.Part,
    length_mm: float,
    width_mm: float,
    height_mm: float,
    marker: str,
) -> tuple[bd.Part, Pose]:
    del width_mm
    if marker == "top":
        cutter = bd.Pos(0, 0, height_mm) * bd.Box(
            12, 12, 2.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX)
        )
        return solid - cutter, _pose(_meters((0.0, 0.0, height_mm)), _IDENTITY)
    depth = 0.8
    cutter = bd.Pos(length_mm / 2.0 + 1.0 - depth, 0, height_mm * 0.55) * bd.Box(2, 12, 12)
    face = _pose(_meters((length_mm / 2.0, 0.0, height_mm * 0.55)), _quat(0.0, math.pi / 2.0, 0.0))
    return solid - cutter, face


def _quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    raw = rpy_to_quat(roll, pitch, yaw)
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _meters(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return (values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0)


def _pose(position_m: tuple[float, float, float], quat: tuple[float, float, float, float]) -> Pose:
    return Pose(position_m=position_m, quat_wxyz=quat, frame="body")
