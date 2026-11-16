"""Load a scene YAML file into a :class:`~physlab.sim.scene.SceneSpec`.

Scene files live in ``configs/scenes``. A name is the file stem. The loader
does not accept a filesystem path, so a tool cannot point it at another
config directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physlab.core.errors import ConfigError
from physlab.core.types import Pose
from physlab.sim.scene import (
    CameraSpec,
    RobotPlacement,
    SceneSpec,
    TableSpec,
    add_overhead_stereo,
    add_wrist_camera,
)


class _PoseFile(BaseModel):
    """Pose stored in a scene file. Lengths are meters and the quaternion is wxyz."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame: str = "world"

    def pose(self) -> Pose:
        """Return the physlab pose.

        Returns:
            Pose in ``frame``.
        """
        return Pose(position_m=self.position_m, quat_wxyz=self.quat_wxyz, frame=self.frame)


class _RobotFile(BaseModel):
    """One robot placement."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    robot_id: str
    instance: str
    base_pose: _PoseFile
    arm_id: str | None = None


class _CameraFile(BaseModel):
    """One fixed or attached camera."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str
    pose: _PoseFile | None = None
    attached_to_body: str | None = None
    fovy_deg: float = 60.0
    width: int = 640
    height: int = 480


class _TableFile(BaseModel):
    """Table top size and height, both in meters."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    size_m: tuple[float, float] = (1.2, 0.8)
    height_m: float = 0.75


class _OverheadFile(BaseModel):
    """Arguments for the built-in downward stereo pair."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    height_m: float = 1.0
    baseline_m: float = 0.4
    width: int = 640
    height: int = 480


class _WristFile(BaseModel):
    """A wrist camera on one robot instance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    instance: str
    ee_body: str = "link7"


class SceneFile(BaseModel):
    """On-disk scene. ``name`` is the stem an agent passes to ``load_scene``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str
    table: _TableFile | None = None
    floor_size_m: tuple[float, float] = (4.0, 4.0)
    robots: list[_RobotFile] = Field(default_factory=list)
    cameras: list[_CameraFile] = Field(default_factory=list)
    overhead_stereo: _OverheadFile | None = None
    wrist_cameras: list[_WristFile] = Field(default_factory=list)


def scene_dir(configs_dir: Path) -> Path:
    """Return the directory of scene files.

    Args:
        configs_dir: Physlab configs directory.

    Returns:
        ``configs_dir / "scenes"``.
    """
    return configs_dir / "scenes"


def list_scene_names(configs_dir: Path) -> list[str]:
    """List scene stems in alphabetical order.

    Args:
        configs_dir: Physlab configs directory.

    Returns:
        File stems. An empty directory yields an empty list.
    """
    directory = scene_dir(configs_dir)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def scene_path(configs_dir: Path, name: str) -> Path:
    """Resolve a scene stem to a file inside the scenes directory.

    Args:
        configs_dir: Physlab configs directory.
        name: File stem. Slashes and ``..`` are rejected.

    Returns:
        Path of the YAML file.

    Raises:
        ConfigError: The name is not a single stem or the file is missing.
    """
    if not name or name != Path(name).name or "/" in name or "\\" in name or ".." in name:
        raise ConfigError("Scene name must be a file stem", details={"name": name})
    path = scene_dir(configs_dir) / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(
            "Scene was not found",
            details={"name": name, "scenes": list_scene_names(configs_dir)},
        )
    return path


def load_scene_file(path: Path) -> tuple[str, SceneSpec]:
    """Parse a scene file.

    Args:
        path: YAML file written with the :class:`SceneFile` schema.

    Returns:
        Scene name and the spec the simulator can compile.

    Raises:
        ConfigError: The YAML is empty or does not match the schema.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("Scene file must be a mapping", details={"path": str(path)})
    try:
        parsed = SceneFile.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError("Scene file is invalid", details={"path": str(path)}) from exc
    table = (
        None
        if parsed.table is None
        else TableSpec(size_m=parsed.table.size_m, height_m=parsed.table.height_m)
    )
    scene = SceneSpec(
        table=table,
        floor_size_m=parsed.floor_size_m,
        robots=[
            RobotPlacement(
                robot_id=robot.robot_id,
                instance=robot.instance,
                base_pose=robot.base_pose.pose(),
                arm_id=robot.arm_id,
            )
            for robot in parsed.robots
        ],
        cameras=[_camera(camera) for camera in parsed.cameras],
    )
    if parsed.overhead_stereo is not None:
        stereo = parsed.overhead_stereo
        add_overhead_stereo(
            scene,
            height_m=stereo.height_m,
            baseline_m=stereo.baseline_m,
            width=stereo.width,
            height=stereo.height,
        )
    for wrist in parsed.wrist_cameras:
        add_wrist_camera(scene, wrist.instance, ee_body=wrist.ee_body)
    return parsed.name, scene


def _camera(camera: _CameraFile) -> CameraSpec:
    pose = None if camera.pose is None else camera.pose.pose()
    return CameraSpec(
        name=camera.name,
        pose=pose,
        attached_to_body=camera.attached_to_body,
        fovy_deg=camera.fovy_deg,
        width=camera.width,
        height=camera.height,
    )
