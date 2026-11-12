"""Render an STL as a PNG with trimesh and matplotlib.

The mesh file is in millimeters, which is how the fabrication library writes
STL. The image is a shaded triangle view, not a photograph.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import trimesh

from physlab.core.errors import ConfigError


def render_stl_png(stl_path: Path) -> bytes:
    """Draw ``stl_path`` and return a PNG.

    Args:
        stl_path: Triangle mesh in millimeters.

    Returns:
        PNG bytes.

    Raises:
        ConfigError: The file is missing or is not a triangle mesh.
    """
    if not stl_path.is_file():
        raise ConfigError("STL was not found", details={"path": str(stl_path)})
    loaded = trimesh.load(stl_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ConfigError("STL did not contain a triangle mesh", details={"path": str(stl_path)})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces)
    if len(faces) > 2500:
        step = int(np.ceil(len(faces) / 2500))
        faces = faces[::step]
    triangles = vertices[faces]
    figure = plt.figure(figsize=(4.0, 4.0), dpi=80)
    axis = figure.add_subplot(111, projection="3d")
    collection = Poly3DCollection(triangles, alpha=0.9)
    collection.set_facecolor((0.55, 0.62, 0.70, 0.95))
    collection.set_edgecolor((0.15, 0.18, 0.22, 0.25))
    axis.add_collection3d(collection)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    axis.set_xlim(float(mins[0]), float(maxs[0]))
    axis.set_ylim(float(mins[1]), float(maxs[1]))
    axis.set_zlim(float(mins[2]), float(maxs[2]))
    axis.set_xlabel("x mm")
    axis.set_ylabel("y mm")
    axis.set_zlabel("z mm")
    axis.set_title(stl_path.stem)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()
