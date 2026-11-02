"""Sharpness metrics and a bright-spot centroid.

``variance_of_laplacian`` and ``tenengrad`` both fall as an image is blurred.
``centroid_of_bright_spot`` is the intensity-weighted center of the bright pixels.
"""

from __future__ import annotations

import cv2
import numpy as np


def variance_of_laplacian(img: np.ndarray) -> float:
    """Return the variance of the Laplacian of ``img``.

    Args:
        img: Grayscale or RGB image.

    Returns:
        Variance. Larger values are sharper.
    """
    lap = cv2.Laplacian(_gray(img), cv2.CV_64F)
    return float(lap.var())


def tenengrad(img: np.ndarray) -> float:
    """Return the mean squared Sobel gradient magnitude.

    Args:
        img: Grayscale or RGB image.

    Returns:
        Tenengrad score. Larger values are sharper.
    """
    gray = _gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def centroid_of_bright_spot(img: np.ndarray) -> tuple[float, float] | None:
    """Return the intensity-weighted centroid of the bright pixels.

    Pixels above the 90th percentile and at least 128 are the spot. The
    result is ``(u, v)`` in pixels, x then y.

    Args:
        img: Grayscale or RGB image.

    Returns:
        Centroid, or ``None`` when no pixel is bright enough.
    """
    gray = _gray(img)
    cutoff = max(128.0, float(np.percentile(gray, 90)))
    mask = gray >= cutoff
    if not np.any(mask):
        return None
    rows, cols = np.nonzero(mask)
    weights = gray[rows, cols].astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return float(np.dot(cols, weights) / total), float(np.dot(rows, weights) / total)


def _gray(img: np.ndarray) -> np.ndarray:
    image = np.asarray(img)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
