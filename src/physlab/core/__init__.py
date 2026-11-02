"""Core types, frames, configuration, logging, events, and registries."""

from physlab.core.config import GlBackend, PhysLabConfig, SimConfig, load_config
from physlab.core.errors import (
    ApprovalRequired,
    AssetMissing,
    CodeRejected,
    ConfigError,
    DependencyMissing,
    HardwareError,
    PerceptionFailed,
    PhysLabError,
    PlanningFailed,
    SafetyViolation,
)
from physlab.core.events import Actor, Event, EventLog, new_run_id
from physlab.core.frames import (
    matrix_to_quat,
    quat_angle_between,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    rpy_to_quat,
)
from physlab.core.logging import configure_logging, get_logger
from physlab.core.registry import Registry
from physlab.core.types import (
    CHILD_FRAME,
    FailureReason,
    JointState,
    Mode,
    MotionResult,
    Pose,
    Pose2D,
)

__all__ = [
    "CHILD_FRAME",
    "Actor",
    "ApprovalRequired",
    "AssetMissing",
    "CodeRejected",
    "ConfigError",
    "DependencyMissing",
    "Event",
    "EventLog",
    "FailureReason",
    "GlBackend",
    "HardwareError",
    "JointState",
    "Mode",
    "MotionResult",
    "PerceptionFailed",
    "PhysLabConfig",
    "PhysLabError",
    "PlanningFailed",
    "Pose",
    "Pose2D",
    "Registry",
    "SafetyViolation",
    "SimConfig",
    "configure_logging",
    "get_logger",
    "load_config",
    "matrix_to_quat",
    "new_run_id",
    "quat_angle_between",
    "quat_multiply",
    "quat_normalize",
    "quat_to_matrix",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "rpy_to_quat",
]
