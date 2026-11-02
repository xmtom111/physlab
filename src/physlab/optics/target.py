"""A generated resolution target in the style of a USAF chart."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from physlab.core.config import load_config
from physlab.core.errors import PerceptionFailed


def usaf_like_target(size_px: int = 512) -> Path:
    """Write a bar-chart target and return its path.

    The file is cached under ``assets/generated``. Later calls reuse it.

    Args:
        size_px: Image side length, pixels.

    Returns:
        Path of the PNG.

    Raises:
        PerceptionFailed: The PNG could not be written.
    """
    path = load_config().assets_dir / "generated" / "usaf_like_target.png"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _draw(size_px)
    if not cv2.imwrite(str(path), image):
        raise PerceptionFailed("Could not write the resolution target", details={"path": str(path)})
    return path


def _draw(size_px: int) -> np.ndarray:
    image = np.full((size_px, size_px), 230, dtype=np.uint8)
    pitch = size_px // 8
    for group in range(6):
        bar = max(2, pitch // (2 + group))
        origin = 24 + group * pitch
        for index in range(3):
            y0 = origin + index * bar * 2
            image[y0 : y0 + bar, origin : origin + bar * 5] = 20
            x0 = origin + index * bar * 2
            image[origin + bar * 6 : origin + bar * 11, x0 : x0 + bar] = 20
    return image
