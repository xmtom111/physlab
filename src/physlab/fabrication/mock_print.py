"""A print service that finishes on a simulated clock and can spawn the part."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from physlab.core.errors import ConfigError
from physlab.core.types import Pose
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.models import InventoryItem, PrintJob, PrintState, PrintStatus
from physlab.sim.scene import ObjectSpec
from physlab.sim.world import SimWorld

_PLA_DENSITY_G_CM3 = 1.0


class MockPrintService:
    """Estimate print time from volume and optionally drop the part in sim.

    Time comes from ``clock`` (monotonic seconds by default). The duration is
    ``volume_mm3 * copies / (mm3_per_s * speed_factor)``. State moves from
    queued to slicing to printing to done as that fraction advances. On
    ``DONE``, the part is spawned at the next free pickup pose when a world
    was provided.
    """

    def __init__(
        self,
        world: SimWorld | None = None,
        speed_factor: float = 1.0,
        *,
        mm3_per_s: float = 15.0,
        clock: Callable[[], float] | None = None,
        pickup_poses: list[Pose] | None = None,
    ) -> None:
        """Store the world, rate, and pickup slots.

        Args:
            world: Simulation that receives finished parts, or ``None``.
            speed_factor: Multiplier on ``mm3_per_s``.
            mm3_per_s: Nominal extrusion rate, cubic millimeters per second.
            clock: Simulated seconds. Defaults to :func:`time.monotonic`.
            pickup_poses: Slot poses for the part origin. The first default
                slot is the xArm pick pose used by the cube example.

        Raises:
            ConfigError: The rate or speed is not positive, or no pickup pose
                was given.
        """
        if speed_factor <= 0 or mm3_per_s <= 0:
            raise ConfigError(
                "Print rate must be positive",
                details={"speed_factor": speed_factor, "mm3_per_s": mm3_per_s},
            )
        self.world = world
        self.speed_factor = speed_factor
        self.mm3_per_s = mm3_per_s
        self.inventory = Inventory()
        self._clock = clock or time.monotonic
        self._poses = list(pickup_poses) if pickup_poses is not None else _default_pickups()
        if not self._poses:
            raise ConfigError("print_pickup_zone has no poses")
        self._jobs: dict[str, _MockJob] = {}
        self._next_id = 1
        self._next_aruco = 31
        self._used_slots = 0

    def submit(self, job: PrintJob) -> str:
        """Queue ``job`` and return its id.

        Args:
            job: Part, material, quality, and copies.

        Returns:
            Job id.
        """
        volume_mm3 = _volume_mm3(job.part.stl_path) * job.copies
        duration = volume_mm3 / (self.mm3_per_s * self.speed_factor)
        job_id = f"mock-{self._next_id}"
        self._next_id += 1
        self._jobs[job_id] = _MockJob(
            job=job,
            started_s=self._clock(),
            duration_s=duration,
            instance_id=f"{job.part.name}_{job_id.replace('-', '_')}",
        )
        return job_id

    def status(self, job_id: str) -> PrintStatus:
        """Advance ``job_id`` on the simulated clock.

        Args:
            job_id: Id from :meth:`submit`.

        Returns:
            The state implied by the elapsed simulated time.

        Raises:
            ConfigError: ``job_id`` is unknown.
        """
        record = self._require(job_id)
        if record.cancelled:
            return PrintStatus(
                state=PrintState.CANCELLED, progress=record.progress, message="cancelled"
            )
        fraction = _fraction(self._clock() - record.started_s, record.duration_s)
        record.progress = fraction
        state = _state_for(fraction)
        if state is PrintState.DONE:
            self._finish(record)
        eta = None if record.duration_s <= 0 else max(0.0, record.duration_s * (1.0 - fraction))
        return PrintStatus(state=state, progress=fraction, message=state.value.lower(), eta_s=eta)

    def cancel(self, job_id: str) -> None:
        """Mark ``job_id`` cancelled when it is not already done.

        Args:
            job_id: Id from :meth:`submit`.

        Raises:
            ConfigError: ``job_id`` is unknown.
        """
        record = self._require(job_id)
        if record.spawned:
            return
        record.cancelled = True

    def inventory_entry(self, job_id: str) -> InventoryItem | None:
        """Return the inventory row for a finished job.

        Args:
            job_id: Id from :meth:`submit`.

        Returns:
            The row after ``DONE``, otherwise ``None``.
        """
        record = self._jobs.get(job_id)
        if record is None or record.item is None:
            return None
        return record.item

    def _finish(self, record: _MockJob) -> None:
        if record.item is not None:
            return
        pose: Pose | None = None
        location = "virtual"
        aruco_id: int | None = None
        if self.world is not None:
            pose, aruco_id = self._spawn(record)
            location = "print_pickup_zone"
        record.item = self.inventory.add(
            InventoryItem(
                part_instance_id=record.instance_id,
                part_name=record.job.part.name,
                aruco_id=aruco_id,
                location=location,
                pose=pose,
            )
        )
        record.spawned = True

    def _spawn(self, record: _MockJob) -> tuple[Pose, int]:
        if self.world is None:
            raise ConfigError("No simulation world is attached")
        if self._used_slots >= len(self._poses):
            raise ConfigError("print_pickup_zone is full")
        pose = self._poses[self._used_slots]
        self._used_slots += 1
        aruco_id = self._next_aruco
        self._next_aruco += 1
        if aruco_id > 49:
            raise ConfigError("No ArUco id remains in DICT_4X4_50")
        visual = _meters_stl(record.job.part.stl_path)
        volume_cm3 = _volume_mm3(record.job.part.stl_path) / 1000.0
        mass = min(0.05, max(0.02, volume_cm3 * _PLA_DENSITY_G_CM3 / 1000.0))
        self.world.add_object_at_runtime(
            ObjectSpec(
                name=record.instance_id,
                visual_mesh=visual,
                collision=list(record.job.part.collision_primitives),
                pose=pose,
                mass_kg=mass,
                rgba=(0.82, 0.84, 0.86, 1.0),
                free=True,
                aruco_id=aruco_id,
                aruco_size_m=0.03,
            )
        )
        return pose, aruco_id

    def _require(self, job_id: str) -> _MockJob:
        record = self._jobs.get(job_id)
        if record is None:
            raise ConfigError("Unknown print job", details={"job_id": job_id})
        return record


@dataclass
class _MockJob:
    job: PrintJob
    started_s: float
    duration_s: float
    instance_id: str
    progress: float = 0.0
    cancelled: bool = False
    spawned: bool = False
    item: InventoryItem | None = None


def _default_pickups() -> list[Pose]:
    identity = (1.0, 0.0, 0.0, 0.0)
    return [
        Pose(position_m=(0.48, 0.0, 0.75), quat_wxyz=identity),
        Pose(position_m=(0.48, 0.12, 0.75), quat_wxyz=identity),
        Pose(position_m=(0.36, 0.0, 0.75), quat_wxyz=identity),
    ]


def _fraction(elapsed_s: float, duration_s: float) -> float:
    if duration_s <= 0:
        return 1.0
    return min(1.0, max(0.0, elapsed_s / duration_s))


def _state_for(fraction: float) -> PrintState:
    if fraction < 0.05:
        return PrintState.QUEUED
    if fraction < 0.2:
        return PrintState.SLICING
    if fraction < 1.0:
        return PrintState.PRINTING
    return PrintState.DONE


def _volume_mm3(stl_path: Path) -> float:
    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return 0.0
    return abs(float(mesh.volume))


def _meters_stl(stl_path: Path) -> Path:
    destination = stl_path.with_name(f"{stl_path.stem}_m.stl")
    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ConfigError("STL did not contain a mesh", details={"path": str(stl_path)})
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
    faces = np.asarray(mesh.faces)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(destination)
    return destination
