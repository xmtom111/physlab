"""Hardware interfaces shared by simulation and real robots.

Factory helpers are loaded lazily. Importing the protocols must not import
the MuJoCo backends, because those backends import :class:`Contact` from this
package.
"""

from __future__ import annotations

from typing import Any

from physlab.hal.interfaces import (
    Arm,
    Camera,
    CameraIntrinsics,
    Contact,
    Frame,
    Gripper,
    MobileBase,
    Robot,
)

__all__ = [
    "Arm",
    "Camera",
    "CameraIntrinsics",
    "Contact",
    "Frame",
    "Gripper",
    "MobileBase",
    "Robot",
    "create_camera",
    "create_robot",
]


def __getattr__(name: str) -> Any:  # factory objects are imported lazily
    """Load ``create_robot`` and ``create_camera`` on first use.

    Args:
        name: Attribute requested from the package.

    Returns:
        The factory function.

    Raises:
        AttributeError: ``name`` is not a factory export.
    """
    if name in {"create_camera", "create_robot"}:
        from physlab.hal import factory

        value = getattr(factory, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
