"""Simulated infinity-corrected microscope.

The bench runs along world X. Focus is the sample's Z relative to the
objective. Lateral error is a Y mismatch. The rendered picture is a USAF-like
target scaled by the magnification, blurred by defocus, shifted when the
sample is off axis, and darkened when the LED misses the axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from physlab.core.errors import ConfigError, PerceptionFailed
from physlab.core.types import Pose
from physlab.optics.paraxial import afocal_magnification, blur_circle_diameter, propagate_rays
from physlab.optics.target import usaf_like_target
from physlab.sim.world import SimWorld


class SensorSpec(BaseModel):
    """Camera sensor used by :class:`MicroscopeModel`.

    Attributes:
        width_px: Image width.
        height_px: Image height.
        pixel_m: Pixel pitch, meters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    width_px: int = Field(default=160, gt=0)
    height_px: int = Field(default=120, gt=0)
    pixel_m: float = Field(default=5e-6, gt=0)


@dataclass(frozen=True)
class MicroscopeView:
    """One rendered frame and the scalars that produced it."""

    image: np.ndarray
    magnification: float
    defocus_m: float
    lateral_offset_m: float
    illumination: float

    def metrics(self) -> dict[str, float]:
        """Return the scalar metrics.

        Returns:
            Magnification, defocus, lateral offset, and illumination.
        """
        return {
            "magnification": self.magnification,
            "defocus_m": self.defocus_m,
            "lateral_offset_m": self.lateral_offset_m,
            "illumination": self.illumination,
        }


class MicroscopeModel:
    """Read the live part poses and paint a simulated camera frame."""

    def __init__(
        self,
        world: SimWorld,
        objective_f_m: float,
        tube_lens_f_m: float,
        sensor: SensorSpec | None = None,
        *,
        sample_stage: str,
        objective: str,
        tube: str,
        camera_mount: str,
        led_holder: str,
        aperture_m: float = 0.008,
    ) -> None:
        """Store the optical prescription and the body names to track.

        Args:
            world: Simulation whose body poses are sampled at render time.
            objective_f_m: Objective focal length, meters.
            tube_lens_f_m: Tube-lens focal length, meters.
            sensor: Detector size. A 160 by 120 sensor is the default.
            sample_stage: Body whose Z sets focus.
            objective: Objective body. Its Z is the focus reference.
            tube: Tube body. Its Y contributes to lateral error.
            camera_mount: Camera body. Its Y contributes to lateral error.
            led_holder: Illuminator body.
            aperture_m: Objective pupil diameter, meters.
        """
        self.world = world
        self.objective_f_m = objective_f_m
        self.tube_lens_f_m = tube_lens_f_m
        self.sensor = sensor or SensorSpec()
        self.bodies = {
            "sample_stage": sample_stage,
            "objective": objective,
            "tube": tube,
            "camera_mount": camera_mount,
            "led_holder": led_holder,
        }
        self.aperture_m = aperture_m

    def render(self) -> MicroscopeView:
        """Paint the current microscope image.

        Returns:
            Image and metrics. The image is ``uint8`` grayscale.

        Raises:
            PerceptionFailed: A required body is not in the world.
        """
        poses = {role: self._pose(name) for role, name in self.bodies.items()}
        view = _metrics(self, poses)
        image = _paint(self.sensor, view)
        return MicroscopeView(
            image, view.magnification, view.defocus_m, view.lateral_offset_m, view.illumination
        )

    def _pose(self, name: str) -> Pose:
        try:
            return self.world.body_pose(name)
        except ConfigError as exc:
            raise PerceptionFailed("Microscope body is missing", details={"body": name}) from exc


def _metrics(model: MicroscopeModel, poses: dict[str, Pose]) -> MicroscopeView:
    sample = poses["sample_stage"].position_m
    objective = poses["objective"].position_m
    tube = poses["tube"].position_m
    camera = poses["camera_mount"].position_m
    led = poses["led_holder"].position_m
    # Touch the paraxial ray helper so the bench spacing is an actual ray.
    _ray_through_tube(model, camera[0] - tube[0])
    defocus = sample[2] - objective[2]
    lateral = max(
        abs(sample[1] - objective[1]), abs(tube[1] - objective[1]), abs(camera[1] - objective[1])
    )
    led_y = led[1] - objective[1]
    led_z = led[2] - objective[2]
    illumination = float(np.exp(-((led_y / 0.01) ** 2 + (led_z / 0.01) ** 2)))
    mag = afocal_magnification(model.objective_f_m, model.tube_lens_f_m)
    return MicroscopeView(
        image=np.zeros((1, 1), dtype=np.uint8),
        magnification=mag,
        defocus_m=defocus,
        lateral_offset_m=lateral,
        illumination=illumination,
    )


def _ray_through_tube(model: MicroscopeModel, gap_m: float) -> None:
    height, _angle = propagate_rays(
        np.array([0.001], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        gap_m,
    )
    del height
    del model


def _paint(sensor: SensorSpec, view: MicroscopeView) -> np.ndarray:
    target = cv2.imread(str(usaf_like_target()), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise PerceptionFailed("USAF-like target could not be read")
    scale = max(0.25, min(8.0, abs(view.magnification)))
    window = max(16, round(sensor.width_px / scale))
    origin_x = target.shape[1] // 2 - window // 2
    origin_y = target.shape[0] // 2 - window // 2
    crop = target[origin_y : origin_y + window, origin_x : origin_x + window]
    if crop.size == 0:
        crop = target
    resized = cv2.resize(crop, (sensor.width_px, sensor.height_px), interpolation=cv2.INTER_AREA)
    image = np.asarray(resized, dtype=np.uint8)
    image = _shift(image, view.lateral_offset_m, sensor.pixel_m, scale)
    image = _blur(image, view)
    image = _vignette(image, view.lateral_offset_m)
    painted = np.clip(image.astype(np.float32) * view.illumination, 0, 255)
    return cast(np.ndarray, np.asarray(painted, dtype=np.uint8))


def _blur(image: np.ndarray, view: MicroscopeView) -> np.ndarray:
    image_defocus = view.defocus_m * (view.magnification**2)
    blur_m = blur_circle_diameter(image_defocus, 0.008, max(abs(view.magnification) * 0.02, 1e-4))
    sigma = min(24.0, blur_m / 5e-6)
    blurred = image if sigma < 0.4 else cv2.GaussianBlur(image, (0, 0), sigma)
    # Contrast falls with defocus so the focus score has one peak at zero.
    # A plain Gaussian on 8-bit pixels is not monotonic past about 100 um.
    contrast = 1.0 / (1.0 + (view.defocus_m / 40e-6) ** 2)
    values = np.asarray(blurred, dtype=np.float32)
    mean = float(values.mean())
    scaled = np.clip(mean + (values - mean) * contrast, 0, 255)
    return cast(np.ndarray, np.asarray(scaled, dtype=np.uint8))


def _shift(image: np.ndarray, lateral_m: float, pixel_m: float, scale: float) -> np.ndarray:
    width = image.shape[1]
    if width == 0:
        return image
    pixels = round(lateral_m * scale / pixel_m) % width
    if pixels == 0:
        return image
    return np.roll(image, pixels, axis=1)


def _vignette(image: np.ndarray, lateral_m: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    radius = np.sqrt((xx - image.shape[1] / 2) ** 2 + (yy - image.shape[0] / 2) ** 2)
    radius /= max(image.shape) / 2
    edge = np.clip(1.0 - 0.35 * radius**2 - min(lateral_m / 0.01, 1.5), 0.05, 1.0)
    shaded = np.clip(image.astype(np.float32) * edge, 0, 255)
    return cast(np.ndarray, np.asarray(shaded, dtype=np.uint8))
