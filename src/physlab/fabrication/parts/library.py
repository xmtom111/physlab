"""Parametric parts.

Every part has finger notches and an ArUco recess. Defaults fit a
220 x 220 x 250 mm bed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, TypeVar

import build123d as bd
from pydantic import BaseModel, ConfigDict, Field

from physlab.core.errors import ConfigError
from physlab.core.types import Pose
from physlab.fabrication.models import PartArtifact
from physlab.fabrication.parts.common import (
    apply_casing,
    box_primitive,
    centered_box,
    cylinder_primitive,
    make_artifact,
    optical_axis,
    standard_frames,
)
from physlab.sim.scene import PrimitiveSpec

_LibraryEntry = tuple[type[BaseModel], object]
_BUILDERS: dict[str, _LibraryEntry] = {}


class _Params(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class MagneticBaseParams(_Params):
    """Puck with a magnet pocket."""

    diameter_mm: float = Field(default=40, gt=0)
    height_mm: float = Field(default=12, gt=0)
    magnet_d_mm: float = Field(default=10, gt=0)
    magnet_h_mm: float = Field(default=3, gt=0)


class LensHolderParams(_Params):
    """Block with a horizontal lens bore."""

    lens_d_mm: float = Field(default=16, gt=0)
    lens_thickness_mm: float = Field(default=5, gt=0)
    axis_height_mm: float = Field(default=24, gt=0)


class TubeParams(_Params):
    """Standing tube."""

    inner_d_mm: float = Field(default=18, gt=0)
    length_mm: float = Field(default=40, gt=0)
    wall_mm: float = Field(default=2, gt=0)


class SampleStageParams(_Params):
    """Square stage with a travel well."""

    size_mm: float = Field(default=60, gt=0)
    z_travel_mm: float = Field(default=10, gt=0)


class CameraMountParams(_Params):
    """Plate with a pocket for a camera board."""

    board: str = "generic_32mm"


class LedHolderParams(_Params):
    """Block with a vertical LED hole."""

    led_d_mm: float = Field(default=5, gt=0)


class RailBaseplateParams(_Params):
    """Plate with peg slots on a fixed pitch."""

    length_mm: float = Field(default=140, gt=0)
    slot_pitch_mm: float = Field(default=20, gt=0)


class PendulumKitParams(_Params):
    """Base plus a pendulum arm, for later experiments."""

    arm_length_mm: float = Field(default=80, gt=0)


class InclineRampParams(_Params):
    """Wedge whose height is ``length * tan(angle)``."""

    angle_deg: float = Field(default=15, gt=0, lt=80)


def part_entries() -> dict[str, tuple[type[BaseModel], object]]:
    """Return the name to ``(params model, builder)`` mapping.

    Returns:
        Insertion-ordered registry.
    """
    return dict(_BUILDERS)


_P = TypeVar("_P", bound=_Params)


def _as(model: type[_P], params: BaseModel) -> _P:
    if isinstance(params, model):
        return params
    raise ConfigError("Unexpected part parameters", details={"model": model.__name__})


def _dump(params: _Params) -> dict[str, float | int | str]:
    dumped: dict[str, float | int | str] = {}
    for key, value in params.model_dump().items():
        if isinstance(value, bool) or not isinstance(value, float | int | str):
            raise ConfigError("Part parameter is not a scalar", details={"field": key})
        dumped[key] = value
    return dumped


def _register(name: str, model: type[BaseModel], builder: object) -> None:
    _BUILDERS[name] = (model, builder)


def _magnetic_base(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(MagneticBaseParams, params)
    radius = spec.diameter_mm / 2.0
    body = bd.Cylinder(radius, spec.height_mm, align=_align_min())
    pocket = bd.Pos(0, 0, spec.height_mm) * bd.Cylinder(
        spec.magnet_d_mm / 2.0,
        spec.magnet_h_mm + 1.0,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
    )
    solid, face, notch_z = apply_casing(
        body - pocket, spec.diameter_mm, spec.diameter_mm, spec.height_mm, marker="side"
    )
    return _pack(
        "magnetic_base",
        spec,
        solid,
        directory,
        [cylinder_primitive(radius, spec.height_mm)],
        notch_z,
        face,
    )


def _lens_holder(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(LensHolderParams, params)
    height = spec.axis_height_mm + spec.lens_d_mm / 2.0 + 8.0
    length = max(46.0, spec.lens_d_mm + 24.0)
    width = max(40.0, spec.lens_thickness_mm + 30.0)
    bore = bd.Pos(0, 0, spec.axis_height_mm) * bd.Cylinder(
        spec.lens_d_mm / 2.0, length + 4.0, rotation=(0, 90, 0)
    )
    solid, face, notch_z = apply_casing(
        centered_box(length, width, height) - bore, length, width, height, marker="top"
    )
    frames = standard_frames(notch_z)
    frames["optical_axis"] = optical_axis((0.0, 0.0, spec.axis_height_mm))
    return make_artifact(
        "lens_holder",
        _dump(spec),
        solid,
        directory,
        [box_primitive(length, width, height)],
        frames,
        face,
    )


def _tube(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(TubeParams, params)
    outer_r = spec.inner_d_mm / 2.0 + spec.wall_mm
    outer = bd.Cylinder(outer_r, spec.length_mm, align=_align_min())
    inner = bd.Pos(0, 0, -1) * bd.Cylinder(
        spec.inner_d_mm / 2.0, spec.length_mm + 4.0, align=_align_min()
    )
    diameter = outer_r * 2.0
    solid, face, notch_z = apply_casing(
        outer - inner, diameter, diameter, spec.length_mm, marker="side"
    )
    frames = standard_frames(notch_z)
    frames["optical_axis"] = optical_axis((0.0, 0.0, spec.length_mm / 2.0))
    return make_artifact(
        "tube",
        _dump(spec),
        solid,
        directory,
        # Closed xArm pads sit about 12 mm from the centerline.
        [cylinder_primitive(max(outer_r, 13.0), spec.length_mm)],
        frames,
        face,
    )


def _sample_stage(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(SampleStageParams, params)
    height = 16.0 + spec.z_travel_mm
    solid = centered_box(spec.size_mm, spec.size_mm, height)
    well = bd.Pos(0, 0, height) * bd.Box(
        spec.size_mm * 0.6,
        spec.size_mm * 0.6,
        spec.z_travel_mm + 1.0,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX),
    )
    finished, face, notch_z = apply_casing(
        solid - well, spec.size_mm, spec.size_mm, height, marker="side"
    )
    return _pack(
        "sample_stage",
        spec,
        finished,
        directory,
        [box_primitive(spec.size_mm, spec.size_mm, height)],
        notch_z,
        face,
    )


def _camera_mount(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(CameraMountParams, params)
    if spec.board != "generic_32mm":
        raise ConfigError("Unknown camera board", details={"board": spec.board})
    length, width, height = 72.0, 56.0, 16.0
    pocket = bd.Pos(0, 6, height) * bd.Box(
        32, 32, 4.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX)
    )
    solid, face, notch_z = apply_casing(
        centered_box(length, width, height) - pocket, length, width, height, marker="side"
    )
    return _pack(
        "camera_mount",
        spec,
        solid,
        directory,
        [box_primitive(length, width, height)],
        notch_z,
        face,
    )


def _led_holder(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(LedHolderParams, params)
    length, width, height = 46.0, 40.0, 36.0
    hole = bd.Pos(-8, 0, height) * bd.Cylinder(
        spec.led_d_mm / 2.0, 10.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX)
    )
    solid, face, notch_z = apply_casing(
        centered_box(length, width, height) - hole, length, width, height, marker="top"
    )
    frames = standard_frames(notch_z)
    frames["optical_axis"] = optical_axis((-8.0, 0.0, height - 5.0))
    return make_artifact(
        "led_holder",
        _dump(spec),
        solid,
        directory,
        [box_primitive(length, width, height)],
        frames,
        face,
    )


def _rail_baseplate(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(RailBaseplateParams, params)
    width, height = 70.0, 10.0
    plate = centered_box(spec.length_mm, width, height)
    solid = plate - _slots(spec.length_mm, spec.slot_pitch_mm, height)
    finished, face, notch_z = apply_casing(solid, spec.length_mm, width, height, marker="top")
    return _pack(
        "rail_baseplate",
        spec,
        finished,
        directory,
        [box_primitive(spec.length_mm, width, height)],
        notch_z,
        face,
    )


def _slots(length_mm: float, pitch_mm: float, height_mm: float) -> bd.Part:
    span = max(0.0, length_mm - 30.0)
    count = max(1, int(span // pitch_mm) + 1)
    origin = -((count - 1) * pitch_mm) / 2.0
    cutter = _slot(origin, height_mm)
    for index in range(1, count):
        cutter = cutter + _slot(origin + index * pitch_mm, height_mm)
    return cutter


def _slot(x_mm: float, height_mm: float) -> bd.Part:
    return bd.Pos(x_mm, 0, height_mm) * bd.Box(
        8, 8, 5, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX)
    )


def _pendulum_kit(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(PendulumKitParams, params)
    base_l, base_w, base_h = 50.0, 40.0, 30.0
    base = centered_box(base_l, base_w, base_h)
    arm = bd.Pos(base_l / 2.0 + spec.arm_length_mm / 2.0 - 2.0, 0, 22) * bd.Box(
        spec.arm_length_mm, 10, 8
    )
    solid, face, notch_z = apply_casing(base + arm, base_l, base_w, base_h, marker="top")
    primitives = [
        box_primitive(base_l, base_w, base_h),
        _shifted_box(spec.arm_length_mm, 10, 8, base_l / 2.0 + spec.arm_length_mm / 2.0 - 2.0, 22),
    ]
    return _pack("pendulum_kit", spec, solid, directory, primitives, notch_z, face)


def _incline_ramp(params: BaseModel, directory: Path) -> PartArtifact:
    spec = _as(InclineRampParams, params)
    length, width = 100.0, 48.0
    height = length * math.tan(math.radians(spec.angle_deg))
    wedge = bd.Wedge(
        length,
        width,
        height,
        0,
        0,
        0,
        0,
        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN),
    )
    solid, face, notch_z = apply_casing(wedge, length, width, height, marker="side")
    return _pack(
        "incline_ramp",
        spec,
        solid,
        directory,
        [box_primitive(length, width, height)],
        notch_z,
        face,
    )


def _pack(
    name: str,
    params: _Params,
    solid: bd.Part,
    directory: Path,
    primitives: list[PrimitiveSpec],
    notch_z: float,
    face: Pose,
) -> PartArtifact:
    return make_artifact(
        name, _dump(params), solid, directory, primitives, standard_frames(notch_z), face
    )


def _shifted_box(
    length_mm: float, width_mm: float, height_mm: float, x_mm: float, z_mm: float
) -> PrimitiveSpec:
    primitive = box_primitive(length_mm, width_mm, height_mm)
    pose = primitive.pose.model_copy(update={"position_m": (x_mm / 1000.0, 0.0, z_mm / 1000.0)})
    return primitive.model_copy(update={"pose": pose})


def _align_min() -> tuple[bd.Align, bd.Align, bd.Align]:
    return (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)


_register("magnetic_base", MagneticBaseParams, _magnetic_base)
_register("lens_holder", LensHolderParams, _lens_holder)
_register("tube", TubeParams, _tube)
_register("sample_stage", SampleStageParams, _sample_stage)
_register("camera_mount", CameraMountParams, _camera_mount)
_register("led_holder", LedHolderParams, _led_holder)
_register("rail_baseplate", RailBaseplateParams, _rail_baseplate)
_register("pendulum_kit", PendulumKitParams, _pendulum_kit)
_register("incline_ramp", InclineRampParams, _incline_ramp)
