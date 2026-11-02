"""Exception hierarchy for physlab.

Every domain failure raised by physlab is a :class:`PhysLabError`. Callers can
catch the base class or a specific subclass. ``details`` carries structured
context that is safe to log.
"""

from __future__ import annotations


class PhysLabError(Exception):
    """Base class for errors raised by physlab.

    Args:
        message: Human-readable description of the failure.
        details: Optional structured context. A shallow copy is stored.
    """

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        """Store ``message`` and a shallow copy of ``details``."""
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = {} if details is None else dict(details)


class ConfigError(PhysLabError):
    """A configuration file, environment override, or registry entry is invalid."""


class DependencyMissing(PhysLabError):  # noqa: N818
    """An optional extra is not installed for the requested backend."""


class AssetMissing(PhysLabError):  # noqa: N818
    """A required robot, mesh, or other asset file is not available."""


class SafetyViolation(PhysLabError):  # noqa: N818
    """An action was rejected by the safety policy."""


class PlanningFailed(PhysLabError):  # noqa: N818
    """Inverse kinematics or motion planning could not produce a trajectory."""


class PerceptionFailed(PhysLabError):  # noqa: N818
    """A perception step could not produce a usable estimate."""


class HardwareError(PhysLabError):
    """A hardware backend reported a failure."""


class ApprovalRequired(PhysLabError):  # noqa: N818
    """A real-world action is blocked until a human approves it."""


class CodeRejected(PhysLabError):  # noqa: N818
    """Controller source failed a static check or the sandbox policy."""
