"""Paraxial optics and the simulated microscope."""

from physlab.optics.microscope import MicroscopeModel, MicroscopeView
from physlab.optics.paraxial import (
    afocal_magnification,
    blur_circle_diameter,
    image_distance,
    magnification,
)

__all__ = [
    "MicroscopeModel",
    "MicroscopeView",
    "afocal_magnification",
    "blur_circle_diameter",
    "image_distance",
    "magnification",
]
