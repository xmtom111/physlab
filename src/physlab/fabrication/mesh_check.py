"""Decide whether an STL can be printed on a given machine."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import trimesh
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physlab.core.config import load_config
from physlab.core.errors import AssetMissing, ConfigError

_BED_GAP_MM = 0.4


class PrinterProfile(BaseModel):
    """Machine limits used by the printability check.

    Attributes:
        bed_size_mm: Usable bed width and depth, millimeters.
        max_height_mm: Maximum part height, millimeters.
        nozzle_mm: Nozzle diameter, millimeters.
        layer_mm: Layer height, millimeters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    bed_size_mm: tuple[float, float]
    max_height_mm: float = Field(gt=0)
    nozzle_mm: float = Field(gt=0)
    layer_mm: float = Field(gt=0)


class PrintabilityReport(BaseModel):
    """Result of :func:`check_printable`.

    Attributes:
        ok: True when the mesh is watertight, fits the bed, and its estimated
            minimum feature is at least the nozzle diameter.
        watertight: Trimesh reports a closed volume.
        volume_cm3: Absolute mesh volume, cubic centimeters.
        fits_bed: Extents fit ``bed_size_mm`` and ``max_height_mm``.
        max_overhang_ratio: Worst non-bed downward face, 0 for a vertical wall
            and 1 for a horizontal underside. Gripper notches score near 1.
        min_feature_estimate_mm: Smallest millimeter-scale gap between mesh
            coordinates, or the smallest extent when no gap is resolved.
        issues: Reasons ``ok`` is false.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    watertight: bool
    volume_cm3: float
    fits_bed: bool
    max_overhang_ratio: float
    min_feature_estimate_mm: float
    issues: list[str]


def load_printer_profile(path: Path | None = None) -> PrinterProfile:
    """Load a printer profile from YAML.

    Args:
        path: Profile file. Defaults to ``configs/printers/profile.yaml``.

    Returns:
        Validated profile.

    Raises:
        ConfigError: The file is missing or does not match the schema.
    """
    profile_path = path or (load_config().configs_dir / "printers" / "profile.yaml")
    if not profile_path.is_file():
        raise ConfigError("Printer profile was not found", details={"path": str(profile_path)})
    try:
        loaded = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(
            "Could not read the printer profile",
            details={"path": str(profile_path), "reason": str(exc)},
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(
            "Printer profile YAML is invalid",
            details={"path": str(profile_path), "reason": str(exc)},
        ) from exc
    try:
        return PrinterProfile.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError(
            "Printer profile is invalid",
            details={"path": str(profile_path), "errors": str(exc)},
        ) from exc


def check_printable(stl_path: Path, printer: PrinterProfile) -> PrintabilityReport:
    """Check ``stl_path`` against ``printer``.

    The STL is in millimeters, which is the unit build123d exports.

    Args:
        stl_path: Mesh to inspect.
        printer: Bed, height, nozzle, and layer limits.

    Returns:
        A report. ``ok`` is false when any issue is recorded.

    Raises:
        AssetMissing: The STL cannot be read.
    """
    mesh = _load_mesh(stl_path)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    fits = _fits(extents, printer)
    watertight = bool(mesh.is_watertight)
    feature = _min_feature_mm(mesh)
    issues = _issues(watertight, fits, extents, feature, printer)
    volume = abs(float(mesh.volume)) / 1000.0
    return PrintabilityReport(
        ok=not issues,
        watertight=watertight,
        volume_cm3=volume,
        fits_bed=fits,
        max_overhang_ratio=_overhang_ratio(mesh),
        min_feature_estimate_mm=feature,
        issues=issues,
    )


def _load_mesh(stl_path: Path) -> trimesh.Trimesh:
    if not stl_path.is_file():
        raise AssetMissing("STL was not found", details={"path": str(stl_path)})
    try:
        loaded = trimesh.load(stl_path, force="mesh")
    except Exception as exc:
        raise AssetMissing(
            "STL could not be read",
            details={"path": str(stl_path), "reason": str(exc)},
        ) from exc
    if not isinstance(loaded, trimesh.Trimesh):
        raise AssetMissing("STL did not contain a mesh", details={"path": str(stl_path)})
    return loaded


def _fits(extents: np.ndarray, printer: PrinterProfile) -> bool:
    bed_x, bed_y = printer.bed_size_mm
    return bool(
        extents[0] <= bed_x + 1e-6
        and extents[1] <= bed_y + 1e-6
        and extents[2] <= printer.max_height_mm + 1e-6
    )


def _issues(
    watertight: bool,
    fits: bool,
    extents: np.ndarray,
    feature_mm: float,
    printer: PrinterProfile,
) -> list[str]:
    issues: list[str] = []
    if not watertight:
        issues.append("mesh is not watertight")
    if not fits:
        issues.append(
            "part does not fit the bed "
            f"({extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm, "
            f"bed {printer.bed_size_mm[0]:.0f} x {printer.bed_size_mm[1]:.0f} x "
            f"{printer.max_height_mm:.0f} mm)"
        )
    if feature_mm + 1e-6 < printer.nozzle_mm:
        issues.append(
            f"estimated minimum feature {feature_mm:.2f} mm is below the "
            f"{printer.nozzle_mm:.2f} mm nozzle"
        )
    return issues


def _min_feature_mm(mesh: trimesh.Trimesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    best = float("inf")
    for axis in range(3):
        coords = np.unique(np.round(vertices[:, axis], 0))
        if coords.size < 2:
            continue
        gaps = np.diff(coords)
        wide = [float(gap) for gap in gaps.tolist() if float(gap) >= 1.0]
        if wide:
            best = min(best, min(wide))
    if best == float("inf"):
        values = [float(value) for value in np.asarray(mesh.extents, dtype=np.float64).tolist()]
        return min(values)
    return best


def _overhang_ratio(mesh: trimesh.Trimesh) -> float:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    if normals.size == 0:
        return 0.0
    above_bed = centers[:, 2] > float(mesh.bounds[0, 2]) + _BED_GAP_MM
    downward = normals[:, 2] < -1e-3
    mask = above_bed & downward
    if not np.any(mask):
        return 0.0
    chosen = normals[mask]
    horizontal = np.sqrt(chosen[:, 0] ** 2 + chosen[:, 1] ** 2)
    angles = np.arctan2(-chosen[:, 2], np.maximum(horizontal, 1e-9))
    return float(np.max(angles) / (np.pi / 2.0))
