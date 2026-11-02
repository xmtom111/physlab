"""Build HAL devices bound to one simulation world."""

from __future__ import annotations

from physlab.core.errors import ConfigError
from physlab.hal.interfaces import Camera, Robot
from physlab.sim.backends import SimArm, SimCamera, SimGripper, SimMobileBase
from physlab.sim.scene import CameraSpec
from physlab.sim.world import SimWorld


def create_robot(world: SimWorld, instance: str) -> Robot:
    """Create the arm, gripper, and base for one placed robot.

    Args:
        world: Compiled scene.
        instance: ``RobotPlacement.instance`` prefix.

    Returns:
        Devices for that instance. Cameras are filled by name when they are
        attached to one of the instance bodies or named in the scene.

    Raises:
        ConfigError: ``instance`` is not in the scene.
    """
    placement = next((item for item in world.scene.robots if item.instance == instance), None)
    if placement is None:
        raise ConfigError("Robot instance is not in the scene", details={"instance": instance})
    spec = world.catalog.get(placement.robot_id)
    arm = SimArm(world, instance, spec) if spec.sim.arm_joints else None
    gripper = SimGripper(world, instance, spec) if spec.sim.gripper is not None else None
    base = SimMobileBase(world, instance, spec) if spec.sim.base_joints else None
    cameras = {
        camera.name: create_camera(world, camera.name)
        for camera in world.scene.cameras
        if _camera_for_instance(camera, instance)
    }
    return Robot(spec=spec, arm=arm, gripper=gripper, base=base, cameras=cameras)


def create_camera(world: SimWorld, name: str) -> Camera:
    """Create the camera named ``name``.

    Args:
        world: Compiled scene.
        name: ``CameraSpec.name``.

    Returns:
        A simulation camera with the spec's resolution and field of view.

    Raises:
        ConfigError: The scene has no camera with that name.
    """
    spec = next((camera for camera in world.scene.cameras if camera.name == name), None)
    if spec is None:
        raise ConfigError("Camera is not in the scene", details={"camera": name})
    return SimCamera(
        world,
        name,
        width=spec.width,
        height=spec.height,
        fovy_deg=spec.fovy_deg,
    )


def _camera_for_instance(camera: CameraSpec, instance: str) -> bool:
    if camera.attached_to_body is None:
        return True
    return camera.attached_to_body.startswith(f"{instance}/") or camera.attached_to_body.startswith(
        f"{instance}_"
    )
