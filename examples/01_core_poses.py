"""Demonstrate pose composition, inversion, and a body-frame offset."""

from __future__ import annotations

from physlab.core.frames import rpy_to_quat
from physlab.core.logging import configure_logging, get_logger
from physlab.core.types import CHILD_FRAME, Pose


def main() -> None:
    """Build a wrist pose, attach a 12 cm tool, and check the inverse.

    The tool pose is expressed in the wrist body frame (``frame="body"``).
    ``with_offset`` then shifts the tool origin by 1 cm along the tool X axis.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.01_core_poses")
    wrist = Pose(
        position_m=(0.45, -0.10, 0.25),
        quat_wxyz=tuple(float(value) for value in rpy_to_quat(0.0, 0.15, 0.40)),
        frame="world",
    )
    tool = Pose(
        position_m=(0.0, 0.0, 0.12),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        frame=CHILD_FRAME,
    )
    tool_in_world = wrist.compose(tool)
    shifted = tool_in_world.with_offset(0.01, 0.0, 0.0)
    identity = wrist.compose(wrist.inverse())
    origin = Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0), frame="world")
    translation_m, rotation_rad = identity.error_to(origin)
    log.info(
        "wrist_pose",
        frame=wrist.frame,
        position_m=wrist.position_m,
        quat_wxyz=wrist.quat_wxyz,
    )
    log.info(
        "tool_in_world",
        position_m=tool_in_world.position_m,
        quat_wxyz=tool_in_world.quat_wxyz,
    )
    log.info("shifted_along_body_x", position_m=shifted.position_m)
    log.info("compose_inverse_error", translation_m=translation_m, rotation_rad=rotation_rad)
    assert translation_m < 1e-9
    assert rotation_rad < 1e-9


if __name__ == "__main__":
    main()
