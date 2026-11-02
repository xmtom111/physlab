"""Thin-lens image distance versus object distance.

The plant hides the focal length. A focus search moves the image distance
until a blur score peaks. The score uses the geometric blur circle, so the
best image distance is the paraxial conjugate.
"""

from __future__ import annotations

from physlab.experiments.protocol import (
    ExperimentProtocol,
    Measurement,
    ProcedureStep,
    SafetyClass,
    Variable,
)
from physlab.optics.paraxial import blur_circle_diameter, image_distance

TRUE_FOCAL_M = 0.05
_APERTURE_M = 0.01


class LensBench:
    """One lens of unknown focal length, from the controller's point of view."""

    def __init__(self) -> None:
        """Start at an object distance of 0.12 m."""
        self.object_m = 0.12

    def set_object_distance(self, ctx: object) -> None:
        """Read ``object_m`` from the trial.

        Args:
            ctx: Trial context.
        """
        self.object_m = _value(ctx, "object_m")

    def focus_image_distance(self, ctx: object) -> None:
        """Search the image distance that minimizes blur and record it.

        Args:
            ctx: Trial context that receives ``image_m``.
        """
        _records(ctx)["image_m"] = best_image_distance(self.object_m)


def lens_protocol(seed: int = 0) -> ExperimentProtocol:
    """Return the object-distance sweep.

    Args:
        seed: Shuffle seed.

    Returns:
        Protocol E3.
    """
    return ExperimentProtocol(
        id="E3_thin_lens",
        question="How does image distance depend on object distance?",
        hypothesis="1/f = 1/u + 1/v for a thin lens.",
        apparatus="microscope",
        independent=[Variable(name="object_m", unit="m", values=[0.08, 0.1, 0.12, 0.16, 0.2])],
        dependent=[
            Measurement(
                name="image_m",
                unit="m",
                method="focus search on geometric blur",
            )
        ],
        controlled={},
        procedure=[
            ProcedureStep(skill="set_object_distance"),
            ProcedureStep(skill="focus_image_distance"),
        ],
        repetitions=1,
        seed=seed,
        safety_class=SafetyClass.SIM_ONLY,
    )


def best_image_distance(object_m: float) -> float:
    """Return the image distance with the smallest blur.

    Args:
        object_m: Object distance, meters.

    Returns:
        Image distance, meters.
    """
    low, high = 0.02, 0.8
    for _ in range(40):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        if _blur(object_m, left) < _blur(object_m, right):
            high = right
        else:
            low = left
    return 0.5 * (low + high)


def _blur(object_m: float, image_m: float) -> float:
    true = image_distance(object_m, TRUE_FOCAL_M)
    return blur_circle_diameter(image_m - true, _APERTURE_M, TRUE_FOCAL_M)


def _value(ctx: object, name: str) -> float:
    return float(getattr(ctx, "values", {})[name])


def _records(ctx: object) -> dict[str, float]:
    records = getattr(ctx, "records", None)
    if not isinstance(records, dict):
        raise RuntimeError("trial context has no records")
    return records
