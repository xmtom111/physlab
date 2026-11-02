"""Fuse ArUco views into one world pose.

Each camera produces a pose in the OpenCV camera frame. That frame is
converted into the MuJoCo camera frame (X right, Y up, Z backward) and then
into the world with ``Frame.pose_in_world``. Views more than 5 mm from the
median position are dropped. Positions are averaged. Quaternions are averaged
with the eigenvector method.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from physlab.core.frames import quat_normalize
from physlab.core.types import Pose
from physlab.hal.interfaces import Camera
from physlab.perception.aruco import detect, estimate_pose

_OUTLIER_M = 0.005
# OpenCV camera (X right, Y down, Z forward) into MuJoCo camera
# (X right, Y up, Z backward): 180 degrees about X.
_OPENCV_IN_MUJOCO = Pose(position_m=(0.0, 0.0, 0.0), quat_wxyz=(0.0, 1.0, 0.0, 0.0), frame="body")


class LocalizedPose(BaseModel):
    """A marker pose fused from one or more cameras.

    Attributes:
        pose_world: Marker pose in the world frame.
        n_views: Inlier views that were averaged.
        residual_m: Mean distance of those views from the fused position, meters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    pose_world: Pose
    n_views: int = Field(ge=1)
    residual_m: float = Field(ge=0)


class Localizer:
    """Locate one marker size with a fixed list of cameras."""

    def __init__(self, cameras: list[Camera], marker_size_m: float) -> None:
        """Store the cameras and the physical marker size.

        Args:
            cameras: Cameras that can see the work area.
            marker_size_m: Outer marker side, meters. Same value as
                ``ObjectSpec.aruco_size_m``.
        """
        self.cameras = cameras
        self.marker_size_m = marker_size_m

    def locate(self, marker_id: int) -> LocalizedPose | None:
        """Fuse every camera that sees ``marker_id``.

        Args:
            marker_id: Dictionary id.

        Returns:
            The fused pose, or ``None`` when no inlier view exists.
        """
        poses = [
            pose for camera in self.cameras if (pose := self._view(camera, marker_id)) is not None
        ]
        return _fuse(poses)

    def refine_with_wrist(self, camera: Camera, marker_id: int, expected: Pose) -> Pose:
        """Replace ``expected`` with a wrist-camera measurement when one exists.

        The wrist view is the fine stage: a close camera overrides the coarser
        estimate. ``expected`` is returned unchanged when the marker is not in
        the wrist image.

        Args:
            camera: Wrist or other close camera.
            marker_id: Dictionary id.
            expected: Coarse pose, usually from :meth:`locate`.

        Returns:
            Refined world pose, or ``expected``.
        """
        measured = self._view(camera, marker_id)
        return expected if measured is None else measured

    def _view(self, camera: Camera, marker_id: int) -> Pose | None:
        frame = camera.capture()
        if frame.pose_in_world is None:
            return None
        detection = next((item for item in detect(frame) if item.id == marker_id), None)
        if detection is None:
            return None
        in_opencv = estimate_pose(detection, self.marker_size_m, frame.intrinsics)
        in_mujoco = _OPENCV_IN_MUJOCO.compose(in_opencv.model_copy(update={"frame": "body"}))
        return frame.pose_in_world.compose(in_mujoco)


def _fuse(poses: list[Pose]) -> LocalizedPose | None:
    if not poses:
        return None
    positions = np.asarray([pose.position_m for pose in poses], dtype=np.float64)
    median = np.median(positions, axis=0)
    keep = [
        index
        for index, row in enumerate(positions)
        if float(np.linalg.norm(row - median)) <= _OUTLIER_M
    ]
    if not keep:
        return None
    inliers = [poses[index] for index in keep]
    mean = np.mean(np.asarray([pose.position_m for pose in inliers], dtype=np.float64), axis=0)
    quat = _average_quaternions([np.asarray(pose.quat_wxyz, dtype=np.float64) for pose in inliers])
    distances = [np.linalg.norm(np.asarray(pose.position_m) - mean) for pose in inliers]
    residual = float(np.mean(distances))
    return LocalizedPose(
        pose_world=Pose(
            position_m=(float(mean[0]), float(mean[1]), float(mean[2])),
            quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        ),
        n_views=len(inliers),
        residual_m=residual,
    )


def _average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    reference = quaternions[0]
    matrix = np.zeros((4, 4), dtype=np.float64)
    for quaternion in quaternions:
        aligned = quaternion if float(np.dot(quaternion, reference)) >= 0.0 else -quaternion
        matrix += np.outer(aligned, aligned)
    _values, vectors = np.linalg.eigh(matrix)
    return quat_normalize(vectors[:, int(np.argmax(_values))])
