"""Thin-lens and ABCD helpers.

Object distance is positive in front of the lens. A positive image distance
is a real image on the opposite side. Angles are radians, positive when the
ray is climbing as it travels in +z.
"""

from __future__ import annotations

import numpy as np

from physlab.core.errors import ConfigError


def propagate_matrix(distance_m: float) -> np.ndarray:
    """Return the ABCD matrix of a free-space gap.

    Args:
        distance_m: Travel along the optical axis, meters.

    Returns:
        2 by 2 matrix acting on ``(height_m, angle_rad)``.
    """
    return np.array([[1.0, distance_m], [0.0, 1.0]], dtype=np.float64)


def thin_lens_matrix(focal_m: float) -> np.ndarray:
    """Return the ABCD matrix of a thin lens.

    Args:
        focal_m: Focal length, meters. Must be non-zero.

    Returns:
        2 by 2 matrix.

    Raises:
        ConfigError: ``focal_m`` is zero.
    """
    if focal_m == 0.0:
        raise ConfigError("Focal length must be non-zero")
    return np.array([[1.0, 0.0], [-1.0 / focal_m, 1.0]], dtype=np.float64)


def propagate_rays(
    heights_m: np.ndarray,
    angles_rad: np.ndarray,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance rays through free space.

    Args:
        heights_m: Ray heights, meters.
        angles_rad: Ray angles, radians.
        distance_m: Gap, meters.

    Returns:
        Updated heights and angles.
    """
    matrix = propagate_matrix(distance_m)
    stacked = np.vstack([np.asarray(heights_m, dtype=np.float64), np.asarray(angles_rad)])
    updated = matrix @ stacked
    return updated[0], updated[1]


def image_distance(object_distance_m: float, focal_m: float) -> float:
    """Return the image distance for a thin lens.

    Args:
        object_distance_m: Object distance in front of the lens, meters.
        focal_m: Focal length, meters.

    Returns:
        Image distance, meters. Positive is a real image.

    Raises:
        ConfigError: The object sits on the focal plane or a distance is not
            positive.
    """
    if object_distance_m <= 0.0 or focal_m == 0.0:
        raise ConfigError(
            "Object distance and focal length must be usable",
            details={"object_distance_m": object_distance_m, "focal_m": focal_m},
        )
    denom = (1.0 / focal_m) - (1.0 / object_distance_m)
    if abs(denom) < 1e-12:
        raise ConfigError("Object is at the focal plane", details={"focal_m": focal_m})
    return 1.0 / denom


def magnification(object_distance_m: float, focal_m: float) -> float:
    """Return the transverse magnification of a thin lens.

    Args:
        object_distance_m: Object distance, meters.
        focal_m: Focal length, meters.

    Returns:
        Magnification. Negative means the image is inverted.
    """
    return -image_distance(object_distance_m, focal_m) / object_distance_m


def afocal_magnification(objective_f_m: float, tube_lens_f_m: float) -> float:
    """Return the magnification of an infinity-corrected microscope.

    Args:
        objective_f_m: Objective focal length, meters.
        tube_lens_f_m: Tube-lens focal length, meters.

    Returns:
        Lateral magnification, negative when the image is inverted.

    Raises:
        ConfigError: The objective focal length is zero.
    """
    if objective_f_m == 0.0:
        raise ConfigError("Objective focal length must be non-zero")
    return -tube_lens_f_m / objective_f_m


def blur_circle_diameter(defocus_m: float, aperture_m: float, focal_m: float) -> float:
    """Return the geometric blur-circle diameter on a sensor.

    Args:
        defocus_m: How far the sensor (or the conjugate image) is from focus,
            meters.
        aperture_m: Pupil diameter, meters.
        focal_m: Focal length of the lens that forms the image, meters.

    Returns:
        Blur diameter, meters.

    Raises:
        ConfigError: ``focal_m`` is zero.
    """
    if focal_m == 0.0:
        raise ConfigError("Focal length must be non-zero")
    return abs(aperture_m * defocus_m / focal_m)
