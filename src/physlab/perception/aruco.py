"""ArUco markers: render, detect, and estimate a pose.

The dictionary is ``DICT_4X4_50``. Detection uses OpenCV's ``ArucoDetector``.
The pose is the marker frame in the OpenCV camera frame (X right, Y down, Z
along the optical axis). ``solvePnP`` uses ``SOLVEPNP_IPPE_SQUARE``, then
``solvePnPRefineLM`` polishes both returned solutions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from physlab.core.config import load_config
from physlab.core.errors import ConfigError, PerceptionFailed
from physlab.core.types import Pose
from physlab.hal.interfaces import CameraIntrinsics, Frame

_DEFAULT_DICTIONARY = "DICT_4X4_50"
_Corner = tuple[float, float]


class Detection(BaseModel):
    """One marker found in an image.

    Attributes:
        id: Marker id in the dictionary.
        corners_px: Four corners in pixels, clockwise from the top-left:
            top-left, top-right, bottom-right, bottom-left.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(ge=0)
    corners_px: tuple[_Corner, _Corner, _Corner, _Corner]


def marker_png(marker_id: int, size_px: int, dictionary: str = _DEFAULT_DICTIONARY) -> Path:
    """Write a canonical marker image and return its path.

    The file is cached under ``assets/generated``. A later call with the same
    id, size, and dictionary reuses the file.

    Args:
        marker_id: Dictionary id. ``DICT_4X4_50`` allows 0 through 49.
        size_px: Side length of the image, pixels.
        dictionary: OpenCV predefined dictionary name.

    Returns:
        Path of the PNG.

    Raises:
        ConfigError: The id, size, or dictionary name is invalid.
    """
    if size_px < 16:
        raise ConfigError("Marker image is too small", details={"size_px": size_px})
    _dictionary(dictionary, marker_id)
    directory = load_config().assets_dir / "generated"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dictionary}_{marker_id}_{size_px}.png"
    if path.is_file():
        return path
    image = _marker_image(dictionary, marker_id, size_px)
    if not cv2.imwrite(str(path), image):
        raise PerceptionFailed("Could not write marker image", details={"path": str(path)})
    return path


def plate_scale(size_px: int = 400, quiet_fraction: float = 0.2) -> float:
    """Return the padded plate side divided by the canonical marker side.

    Args:
        size_px: Canonical marker side, pixels.
        quiet_fraction: White border on each side, as a fraction of ``size_px``.

    Returns:
        Scale from marker side to plate side.
    """
    border = _quiet_border(size_px, quiet_fraction)
    return (size_px + 2 * border) / size_px


def marker_plate_png(
    marker_id: int,
    size_px: int,
    dictionary: str = _DEFAULT_DICTIONARY,
    *,
    quiet_fraction: float = 0.2,
) -> Path:
    """Write the marker with a white quiet zone around the black border.

    The canonical marker stays ``size_px`` across. The quiet zone is extra
    white pixels so a camera can detect the marker when the object under it
    is not white.

    Args:
        marker_id: Dictionary id.
        size_px: Canonical marker side, pixels.
        dictionary: OpenCV dictionary name.
        quiet_fraction: White border on each side, as a fraction of ``size_px``.

    Returns:
        Path of the padded PNG.
    """
    canonical = marker_png(marker_id, size_px, dictionary)
    border = _quiet_border(size_px, quiet_fraction)
    path = canonical.with_name(f"{canonical.stem}_quiet{border}.png")
    if path.is_file():
        return path
    image = cv2.imread(str(canonical), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise PerceptionFailed("Could not read marker image", details={"path": str(canonical)})
    padded = cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )
    if not cv2.imwrite(str(path), padded):
        raise PerceptionFailed("Could not write marker plate", details={"path": str(path)})
    return path


def detect(frame: Frame, dictionary: str = _DEFAULT_DICTIONARY) -> list[Detection]:
    """Find markers in ``frame``.

    Args:
        frame: RGB capture.
        dictionary: OpenCV dictionary name.

    Returns:
        Detections in detector order. Empty when nothing is found.
    """
    gray = cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(_dictionary(dictionary, 0), _detector_parameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return []
    found: list[Detection] = []
    for marker_id, marker_corners in zip(ids.ravel().tolist(), corners, strict=True):
        points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        found.append(
            Detection(
                id=int(marker_id),
                corners_px=(
                    (float(points[0, 0]), float(points[0, 1])),
                    (float(points[1, 0]), float(points[1, 1])),
                    (float(points[2, 0]), float(points[2, 1])),
                    (float(points[3, 0]), float(points[3, 1])),
                ),
            )
        )
    return found


def estimate_pose(detection: Detection, marker_size_m: float, intrinsics: CameraIntrinsics) -> Pose:
    """Estimate the marker pose in the OpenCV camera frame.

    The marker frame has its origin at the marker center, X to the right, Y
    up the marker, and Z out of the printed face toward the camera when the
    marker faces the lens. Image corners are matched to that square in the
    order top-left, top-right, bottom-right, bottom-left.

    Args:
        detection: Corners from :func:`detect`.
        marker_size_m: Outer side of the marker, including its black border,
            meters.
        intrinsics: Pinhole parameters. Distortion is taken to be zero.

    Returns:
        Pose of the marker in the camera frame. ``frame`` is ``"camera"``.

    Raises:
        PerceptionFailed: ``solvePnP`` does not return a pose in front of the
            camera after refinement.
        ConfigError: ``marker_size_m`` is not positive.
    """
    if marker_size_m <= 0.0:
        raise ConfigError("Marker size must be positive", details={"marker_size_m": marker_size_m})
    rotation, translation = _square_pose(detection, marker_size_m, intrinsics)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return Pose.from_matrix(matrix, frame="camera")


def _square_pose(
    detection: Detection,
    marker_size_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    half = marker_size_m / 2.0
    # +Y is up on the marker. Image +Y is down, so the top edge is +Y.
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    image_points = np.asarray(detection.corners_px, dtype=np.float64)
    camera = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    ok, rotations, translations, _errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise PerceptionFailed("solvePnP failed", details={"id": detection.id})
    choice = _refine_front(object_points, image_points, camera, rotations, translations)
    if choice is None:
        raise PerceptionFailed("Marker pose is behind the camera", details={"id": detection.id})
    return choice


def _refine_front(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera: np.ndarray,
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Polish each IPPE solution and keep the one in front with the least error.

    OpenCV 5's ``SOLVEPNP_IPPE_SQUARE`` returns two poses. On a nearly
    fronto-parallel square the lower reported error can be the tilted pose, and
    the true pose can be missing until Levenberg-Marquardt starts from both.
    """
    distortion = np.zeros(5, dtype=np.float64)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_error = float("inf")
    for rotation, translation in zip(rotations, translations, strict=True):
        refined_r, refined_t = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera,
            distortion,
            rotation,
            translation,
        )
        z_m = float(np.asarray(refined_t, dtype=np.float64).reshape(3)[2])
        if z_m <= 0.0:
            continue
        projected, _jacobian = cv2.projectPoints(
            object_points, refined_r, refined_t, camera, distortion
        )
        error = float(np.sqrt(np.mean((projected.reshape(4, 2) - image_points) ** 2)))
        if error < best_error:
            best_error = error
            best = (
                np.asarray(refined_r, dtype=np.float64),
                np.asarray(refined_t, dtype=np.float64),
            )
    return best


def _detector_parameters() -> cv2.aruco.DetectorParameters:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return parameters


def _dictionary(name: str, marker_id: int) -> cv2.aruco.Dictionary:
    code = getattr(cv2.aruco, name, None)
    if not isinstance(code, int):
        raise ConfigError("Unknown ArUco dictionary", details={"dictionary": name})
    dictionary = cv2.aruco.getPredefinedDictionary(code)
    if marker_id < 0 or marker_id >= int(dictionary.bytesList.shape[0]):
        raise ConfigError(
            "Marker id is outside the dictionary",
            details={
                "id": marker_id,
                "dictionary": name,
                "count": int(dictionary.bytesList.shape[0]),
            },
        )
    return dictionary


def _quiet_border(size_px: int, quiet_fraction: float) -> int:
    return max(1, round(size_px * quiet_fraction))


def _marker_image(dictionary: str, marker_id: int, size_px: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(_dictionary(dictionary, marker_id), marker_id, size_px)
