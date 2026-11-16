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
from physlab.perception.aruco import marker_plate_png, plate_scale
from physlab.robots.catalog import RobotCatalog
from physlab.robots.compose import compose_mobile_manipulator, load_robot_spec
from physlab.robots.models import BaseType, RobotSpec

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
        robot_id: Catalog id. This is the base when ``arm_id`` is set.
        instance: Prefix applied to every element of this robot.
        base_pose: Pose of the robot anchor in the world frame.
        arm_id: Arm catalog id mounted on ``robot_id``'s mount site, or ``None``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    robot_id: str
    instance: str
    base_pose: Pose
    arm_id: str | None = None


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
        aruco_id: Optional marker id. A textured plate is added on the top face.
        aruco_size_m: Outer side of that marker, including its black border, meters.
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
    aruco_size_m: float = Field(default=0.03, gt=0)

    @model_validator(mode="after")
    def _has_collision(self) -> ObjectSpec:
        if not self.collision:
            raise ValueError("Object collision must contain at least one primitive")
        return self


class CameraSpec(BaseModel):
    """A pinhole camera, fixed in the world or attached to a body.

    Attributes:
        name: Camera name.
        pose: World pose when the camera is fixed. Local pose in the body when
            ``attached_to_body`` is set.
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


class FocusStageSpec(BaseModel):
    """A slide and a hinge that stand in for alignment screws.

    The carriage origin sits on ``pose`` when both joints are at zero. The
    slide resolution used by the fine-align tool is 5 µm. The hinge resolution
    is 0.1 mrad.

    Attributes:
        name: Prefix for the carriage, joints, actuators, and knob site.
        pose: Carriage pose at zero joint angles.
        travel_m: Slide range, meters, symmetric about zero.
        hinge_range_rad: Hinge range, radians, symmetric about zero.
        knob_offset_m: Knob site position in the carriage frame, meters.
        baseplate_pose: When set, a fixed non-colliding baseplate is added.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    pose: Pose
    travel_m: float = Field(default=0.003, gt=0)
    hinge_range_rad: float = Field(default=0.05, gt=0)
    knob_offset_m: tuple[float, float, float] = (0.0, 0.06, 0.05)
    baseplate_pose: Pose | None = None


class SceneSpec(BaseModel):
    """Static scene compiled into one MuJoCo model.

    Attributes:
        table: Table, or ``None`` for an empty floor.
        floor_size_m: Floor width and depth, meters.
        robots: Placed robots.
        objects: Rigid objects.
        cameras: Cameras.
        focus_stages: Screw-adjustable carriages. Empty for scenes that do not
            align a sample.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    chem_deck: bool = False
    table: TableSpec | None = None
    floor_size_m: tuple[float, float] = (4.0, 4.0)
    robots: list[RobotPlacement] = Field(default_factory=list)
    objects: list[ObjectSpec] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)
    focus_stages: list[FocusStageSpec] = Field(default_factory=list)


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
    for stage in scene.focus_stages:
        _add_focus_stage(spec, stage)
    if scene.chem_deck:
        from physlab.chem.gantry_bridge import add_chem_deck

        add_chem_deck(spec)
    _set_offscreen(spec, scene)
    return spec


def _add_focus_stage(spec: mujoco.MjSpec, stage: FocusStageSpec) -> None:
    if stage.baseplate_pose is not None:
        _add_baseplate(spec, stage.baseplate_pose)
    mount = spec.worldbody.add_body(
        name=f"{stage.name}_mount",
        pos=list(stage.pose.position_m),
        quat=list(stage.pose.quat_wxyz),
    )
    carriage = mount.add_body(name=f"{stage.name}_carriage")
    _slide_and_hinge(carriage, stage)
    carriage.add_geom(
        name=f"{stage.name}_mass",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.01, 0.005],
        mass=0.05,
        density=0.0,
        contype=0,
        conaffinity=0,
    )
    carriage.add_site(
        name=f"{stage.name}_knob",
        pos=list(stage.knob_offset_m),
        size=[0.004, 0.004, 0.004],
    )
    _screw_actuator(spec, f"{stage.name}_z", stage.travel_m)
    _screw_actuator(spec, f"{stage.name}_rx", stage.hinge_range_rad)


def _slide_and_hinge(carriage: mujoco.MjsBody, stage: FocusStageSpec) -> None:
    carriage.add_joint(
        name=f"{stage.name}_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 0.0, 1.0],
        limited=1,
        range=[-stage.travel_m, stage.travel_m],
    )
    carriage.add_joint(
        name=f"{stage.name}_rx",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1.0, 0.0, 0.0],
        limited=1,
        range=[-stage.hinge_range_rad, stage.hinge_range_rad],
    )


def _screw_actuator(spec: mujoco.MjSpec, joint: str, travel: float) -> None:
    # Same affine position servo the gantry uses: kp on the diagonal, damping.
    spec.add_actuator(
        name=joint,
        type=mujoco.mjtActuator.mjACTUATOR_POSITION,
        target=joint,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_AFFINE,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biasprm=[0.0, -200.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrllimited=1,
        ctrlrange=[-travel, travel],
    )


def _add_baseplate(spec: mujoco.MjSpec, pose: Pose) -> None:
    body = spec.worldbody.add_body(
        name="baseplate",
        pos=list(pose.position_m),
        quat=list(pose.quat_wxyz),
    )
    body.add_geom(
        name="baseplate_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.18, 0.05, 0.004],
        rgba=[0.35, 0.36, 0.38, 1.0],
        contype=0,
        conaffinity=0,
        density=0.0,
        mass=0.0,
    )


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


_COMPOSED: dict[tuple[str, str, str], RobotSpec] = {}


def placed_spec(catalog: RobotCatalog, placement: RobotPlacement) -> RobotSpec:
    """Return the spec for a placement, composing a mounted arm when requested.

    Args:
        catalog: Robot catalog.
        placement: Where the robot sits. ``arm_id`` mounts that arm on the base.

    Returns:
        The base spec, or the composed mobile-manipulator spec.
    """
    if placement.arm_id is None:
        return catalog.get(placement.robot_id)
    key = (placement.robot_id, placement.arm_id, str(catalog.assets_dir))
    cached = _COMPOSED.get(key)
    if cached is not None:
        return cached
    base = catalog.get(placement.robot_id)
    arm = catalog.get(placement.arm_id)
    composed, _model = compose_mobile_manipulator(
        base,
        arm,
        f"{base.id}__{arm.id}",
        catalog.assets_dir,
    )
    _COMPOSED[key] = composed
    return composed


def _add_robot(spec: mujoco.MjSpec, catalog: RobotCatalog, placement: RobotPlacement) -> None:
    if placement.arm_id is None:
        robot = catalog.get(placement.robot_id)
        child = load_robot_spec(robot, catalog.assets_dir)
    else:
        robot, child = compose_mobile_manipulator(
            catalog.get(placement.robot_id),
            catalog.get(placement.arm_id),
            f"{placement.robot_id}__{placement.arm_id}",
            catalog.assets_dir,
        )
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
        _add_aruco_plate(spec, body, obj)


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
        local = camera.pose
        body.add_camera(
            name=camera.name,
            fovy=camera.fovy_deg,
            pos=list(local.position_m) if local is not None else [0.0, 0.0, 0.0],
            quat=list(local.quat_wxyz) if local is not None else [1.0, 0.0, 0.0, 0.0],
        )
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


def add_overhead_stereo(
    scene: SceneSpec,
    height_m: float = 1.0,
    baseline_m: float = 0.4,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Add two downward cameras separated by ``baseline_m``.

    They sit at ``z = height_m`` and look straight down, so a marker on the
    table near the origin is seen by both.

    Args:
        scene: Scene that receives the cameras.
        height_m: Camera height, meters.
        baseline_m: Distance between the cameras along Y, meters.
        width: Image width, pixels.
        height: Image height, pixels.
    """
    offset = baseline_m / 2.0
    for name, y_m in (("overhead_left", -offset), ("overhead_right", offset)):
        scene.cameras.append(
            CameraSpec(
                name=name,
                pose=Pose(position_m=(0.0, y_m, height_m), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                fovy_deg=50.0,
                width=width,
                height=height,
            )
        )


def add_wrist_camera(scene: SceneSpec, instance: str, ee_body: str = "link7") -> None:
    """Add a camera on ``instance``'s end-effector body.

    The camera sits 5 cm along the body Z axis and looks along that axis.
    MuJoCo cameras look along -Z, so the local frame is rotated 180 degrees
    about X.

    Args:
        scene: Scene that receives the camera.
        instance: Robot placement prefix.
        ee_body: Body name inside the prefixed robot. xArm7's last link is
            ``link7``.
    """
    scene.cameras.append(
        CameraSpec(
            name=f"{instance}_wrist",
            attached_to_body=f"{instance}/{ee_body}",
            pose=Pose(
                position_m=(0.0, 0.0, 0.05),
                quat_wxyz=(0.0, 1.0, 0.0, 0.0),
                frame="body",
            ),
            fovy_deg=60.0,
            width=640,
            height=480,
        )
    )


def _add_aruco_plate(spec: mujoco.MjSpec, body: mujoco.MjsBody, obj: ObjectSpec) -> None:
    if obj.aruco_id is None:
        return
    side_px = 400
    plate = marker_plate_png(obj.aruco_id, side_px)
    texture = f"{obj.name}_aruco_tex"
    material = f"{obj.name}_aruco_mat"
    spec.add_texture(name=texture, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(plate))
    spec.add_material(
        name=material,
        textures=["", texture],
        emission=1.0,
        specular=0.0,
        shininess=0.0,
        rgba=[1.0, 1.0, 1.0, 1.0],
    )
    plate_m = obj.aruco_size_m * plate_scale(side_px)
    top = _collision_top_m(obj)
    body.add_geom(
        name=f"{obj.name}_aruco",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[plate_m / 2.0, plate_m / 2.0, 0.0004],
        pos=[0.0, 0.0, top + 0.0004],
        material=material,
        rgba=[1.0, 1.0, 1.0, 1.0],
        contype=0,
        conaffinity=0,
        density=0.0,
        mass=0.0,
    )


def _collision_top_m(obj: ObjectSpec) -> float:
    top = 0.0
    for primitive in obj.collision:
        center = primitive.pose.position_m[2]
        if primitive.type is PrimitiveKind.BOX:
            center += primitive.size_m[2] / 2.0
        elif primitive.type is PrimitiveKind.SPHERE:
            center += primitive.size_m[0]
        else:
            center += primitive.size_m[1] / 2.0
        top = max(top, center)
    return top


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
