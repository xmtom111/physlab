"""Scene description and the MuJoCo spec builder.

Collision geometry is primitive only. A mesh is attached as a visual geom with
``contype`` and ``conaffinity`` set to 0, because MuJoCo replaces mesh
colliders with their convex hull and that hull fills holes.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import mujoco
from pydantic import BaseModel, ConfigDict, Field, model_validator

from physlab.core.errors import AssetMissing, ConfigError
from physlab.core.types import Pose
from physlab.robots.catalog import RobotCatalog
from physlab.robots.compose import load_robot_spec
from physlab.robots.models import BaseType

_GEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


class PrimitiveKind(StrEnum):
    """Collision primitive. Meshes are not a collision type."""

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAPSULE = "capsule"


class PrimitiveSpec(BaseModel):
    """One collision primitive in an object's body frame.

    ``size_m`` is the full box edge lengths, a sphere radius, or
    ``(radius, height)`` for a cylinder or capsule. MuJoCo box and cylinder
    sizes are half extents; the conversion happens in the builder.

    Attributes:
        type: Primitive kind.
        size_m: Dimensions in meters, interpreted as above.
        pose: Pose of the primitive in the object body frame.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    type: PrimitiveKind
    size_m: tuple[float, ...]
    pose: Pose = Field(
        default_factory=lambda: Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    )


class TableSpec(BaseModel):
    """Rectangular table.

    Attributes:
        size_m: Top width and depth, meters.
        height_m: Top height above the floor, meters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    size_m: tuple[float, float] = (1.2, 0.8)
    height_m: float = 0.75


class RobotPlacement(BaseModel):
    """Where one catalog robot is instantiated.

    Attributes:
        robot_id: Catalog id.
        instance: Prefix applied to every element of this robot.
        base_pose: Pose of the robot anchor in the world frame.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    robot_id: str
    instance: str
    base_pose: Pose


class ObjectSpec(BaseModel):
    """A rigid object with primitive collision and an optional visual mesh.

    Attributes:
        name: Body name.
        visual_mesh: Path to a visual-only mesh, or ``None``.
        collision: Primitive colliders. Must be non-empty.
        pose: Object pose in the world frame.
        mass_kg: Total mass, kilograms.
        rgba: Color and alpha.
        free: True when the object gets a free joint.
        aruco_id: Optional marker id stored on a site.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    visual_mesh: Path | None = None
    collision: list[PrimitiveSpec]
    pose: Pose
    mass_kg: float = Field(gt=0)
    rgba: tuple[float, float, float, float] = (0.8, 0.4, 0.2, 1.0)
    free: bool = False
    aruco_id: int | None = None

    @model_validator(mode="after")
    def _has_collision(self) -> ObjectSpec:
        if not self.collision:
            raise ValueError("Object collision must contain at least one primitive")
        return self


class CameraSpec(BaseModel):
    """A pinhole camera, fixed in the world or attached to a body.

    Attributes:
        name: Camera name.
        pose: World pose. Ignored when ``attached_to_body`` is set.
        attached_to_body: Body the camera is fixed to, or ``None``.
        fovy_deg: Vertical field of view, degrees.
        width: Image width, pixels.
        height: Image height, pixels.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    pose: Pose | None = None
    attached_to_body: str | None = None
    fovy_deg: float = Field(default=60.0, gt=0, lt=180)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)

    @model_validator(mode="after")
    def _has_frame(self) -> CameraSpec:
        if self.pose is None and self.attached_to_body is None:
            raise ValueError("Camera requires pose or attached_to_body")
        return self


class SceneSpec(BaseModel):
    """Static scene compiled into one MuJoCo model.

    Attributes:
        table: Table, or ``None`` for an empty floor.
        floor_size_m: Floor width and depth, meters.
        robots: Placed robots.
        objects: Rigid objects.
        cameras: Cameras.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    table: TableSpec | None = None
    floor_size_m: tuple[float, float] = (4.0, 4.0)
    robots: list[RobotPlacement] = Field(default_factory=list)
    objects: list[ObjectSpec] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)


def build_mjspec(scene: SceneSpec, catalog: RobotCatalog) -> mujoco.MjSpec:
    """Compile a scene spec into a MuJoCo spec.

    Every element that comes from a robot is prefixed with ``"{instance}/"``.
    Object and room geometry keep the names from the scene spec.

    Args:
        scene: Scene to build.
        catalog: Catalog used to load robot descriptions.

    Returns:
        Uncompiled MuJoCo spec.

    Raises:
        ConfigError: Two robots share an instance name, or a camera body is
            missing.
        AssetMissing: A robot or visual mesh cannot be loaded.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "physlab_scene"
    _add_light(spec)
    _add_floor(spec, scene.floor_size_m)
    if scene.table is not None:
        _add_table(spec, scene.table)
    seen: set[str] = set()
    for placement in scene.robots:
        if placement.instance in seen:
            raise ConfigError(
                "Duplicate robot instance",
                details={"instance": placement.instance},
            )
        seen.add(placement.instance)
        _add_robot(spec, catalog, placement)
    for obj in scene.objects:
        _add_object(spec, obj)
    for camera in scene.cameras:
        _add_camera(spec, camera)
    _set_offscreen(spec, scene)
    return spec


def _add_light(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_light(name="sun", pos=[0.0, 0.0, 2.5], dir=[0.0, 0.2, -1.0])


def _add_floor(spec: mujoco.MjSpec, size_m: tuple[float, float]) -> None:
    body = spec.worldbody.add_body(name="floor", pos=[0.0, 0.0, -0.01])
    body.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[size_m[0] / 2.0, size_m[1] / 2.0, 0.01],
        rgba=[0.45, 0.45, 0.47, 1.0],
    )


def _add_table(spec: mujoco.MjSpec, table: TableSpec) -> None:
    body = spec.worldbody.add_body(name="table", pos=[0.0, 0.0, table.height_m / 2.0])
    body.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[table.size_m[0] / 2.0, table.size_m[1] / 2.0, table.height_m / 2.0],
        rgba=[0.55, 0.42, 0.30, 1.0],
    )


def _add_robot(spec: mujoco.MjSpec, catalog: RobotCatalog, placement: RobotPlacement) -> None:
    robot = catalog.get(placement.robot_id)
    child = load_robot_spec(robot, catalog.assets_dir)
    anchor = spec.worldbody.add_body(
        name=f"{placement.instance}_anchor",
        pos=list(placement.base_pose.position_m),
        quat=list(placement.base_pose.quat_wxyz),
    )
    if robot.sim.base_type is BaseType.DIFF_DRIVE:
        # The catalog chassis has no root joint. A free joint on the anchor lets
        # wheel contacts move the base without adding a joint to the robot spec.
        anchor.add_freejoint(name=f"{placement.instance}_base_free")
    site = anchor.add_site(name=f"{placement.instance}_anchor_site", size=[0.01, 0.01, 0.01])
    spec.attach(child, prefix=f"{placement.instance}/", site=site)


def _add_object(spec: mujoco.MjSpec, obj: ObjectSpec) -> None:
    body = spec.worldbody.add_body(
        name=obj.name,
        pos=list(obj.pose.position_m),
        quat=list(obj.pose.quat_wxyz),
    )
    if obj.free:
        body.add_freejoint(name=f"{obj.name}_free")
    for index, primitive in enumerate(obj.collision):
        _add_primitive(body, obj, primitive, index)
    if obj.visual_mesh is not None:
        _add_visual_mesh(spec, body, obj)
    if obj.aruco_id is not None:
        body.add_site(name=f"{obj.name}_aruco_{obj.aruco_id}", size=[0.005, 0.005, 0.005])


def _add_primitive(
    body: mujoco.MjsBody, obj: ObjectSpec, primitive: PrimitiveSpec, index: int
) -> None:
    kind = _GEOM[primitive.type.value]
    size = _mujoco_size(primitive)
    mass = obj.mass_kg if index == 0 else 0.0
    body.add_geom(
        name=f"{obj.name}_col_{index}",
        type=kind,
        size=size,
        pos=list(primitive.pose.position_m),
        quat=list(primitive.pose.quat_wxyz),
        rgba=list(obj.rgba),
        mass=mass,
        density=0.0,
        contype=1,
        conaffinity=1,
    )


def _add_visual_mesh(spec: mujoco.MjSpec, body: mujoco.MjsBody, obj: ObjectSpec) -> None:
    mesh_path = obj.visual_mesh
    if mesh_path is None or not mesh_path.is_file():
        raise AssetMissing(
            "Visual mesh was not found",
            details={"object": obj.name, "path": str(mesh_path)},
        )
    mesh_name = f"{obj.name}_visual_mesh"
    spec.add_mesh(name=mesh_name, file=str(mesh_path))
    body.add_geom(
        name=f"{obj.name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        rgba=list(obj.rgba),
        contype=0,
        conaffinity=0,
        density=0.0,
        mass=0.0,
    )


def _set_offscreen(spec: mujoco.MjSpec, scene: SceneSpec) -> None:
    width = max([640, *(camera.width for camera in scene.cameras)])
    height = max([480, *(camera.height for camera in scene.cameras)])
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height


def _add_camera(spec: mujoco.MjSpec, camera: CameraSpec) -> None:
    if camera.attached_to_body is not None:
        body = spec.body(camera.attached_to_body)
        if body is None or body.name != camera.attached_to_body:
            raise ConfigError(
                "Camera body was not found",
                details={"camera": camera.name, "body": camera.attached_to_body},
            )
        body.add_camera(name=camera.name, fovy=camera.fovy_deg)
        return
    pose = camera.pose
    if pose is None:
        raise ConfigError("Camera has no pose", details={"camera": camera.name})
    spec.worldbody.add_camera(
        name=camera.name,
        pos=list(pose.position_m),
        quat=list(pose.quat_wxyz),
        fovy=camera.fovy_deg,
    )


def _mujoco_size(primitive: PrimitiveSpec) -> list[float]:
    size = primitive.size_m
    if primitive.type is PrimitiveKind.BOX:
        if len(size) != 3:
            raise ConfigError("Box size_m must be (x, y, z) full lengths")
        return [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0]
    if primitive.type is PrimitiveKind.SPHERE:
        if len(size) != 1:
            raise ConfigError("Sphere size_m must be (radius,)")
        return [size[0], 0.0, 0.0]
    if primitive.type in {PrimitiveKind.CYLINDER, PrimitiveKind.CAPSULE}:
        if len(size) != 2:
            raise ConfigError("Cylinder and capsule size_m must be (radius, height)")
        return [size[0], size[1] / 2.0, 0.0]
    raise ConfigError("Unknown primitive", details={"type": primitive.type.value})
