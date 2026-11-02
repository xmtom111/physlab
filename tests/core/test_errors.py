"""Error types carry a message and a detached details dict."""

from __future__ import annotations

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


def test_error_details_are_copied() -> None:
    details: dict[str, object] = {"joint": "j1"}
    error = PhysLabError("failed", details)
    details["joint"] = "j2"
    assert error.message == "failed"
    assert error.details == {"joint": "j1"}
    assert str(error) == "failed"
    subclasses = (
        ConfigError,
        DependencyMissing,
        AssetMissing,
        SafetyViolation,
        PlanningFailed,
        PerceptionFailed,
        HardwareError,
        ApprovalRequired,
        CodeRejected,
    )
    for cls in subclasses:
        raised = cls("nope")
        assert isinstance(raised, PhysLabError)
        assert raised.details == {}
