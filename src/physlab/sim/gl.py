"""Choose a headless MuJoCo GL backend.

EGL is preferred. This environment's PyOpenGL 3.1.0 wheel does not expose
``EGLDeviceEXT``, so the import falls back to OSMesa. ``MUJOCO_GL`` is read
when the renderer module is first imported.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from physlab.core.errors import PerceptionFailed

_RENDERER: Any | None = None
_BACKEND: str | None = None


def renderer_class() -> Any:  # MuJoCo's Renderer class is untyped.
    """Import MuJoCo's ``Renderer``, trying EGL and then OSMesa.

    Returns:
        The ``Renderer`` class from ``mujoco.rendering.classic.renderer``.

    Raises:
        PerceptionFailed: Neither EGL nor OSMesa can be loaded.
    """
    global _RENDERER, _BACKEND
    if _RENDERER is not None:
        return _RENDERER
    requested = os.environ.get("MUJOCO_GL")
    order = [requested] if requested else ["egl", "osmesa"]
    if "osmesa" not in order:
        order.append("osmesa")
    errors: list[str] = []
    for backend in order:
        os.environ["MUJOCO_GL"] = backend
        # MuJoCo's EGL import sets PYOPENGL_PLATFORM and a later OSMesa import
        # refuses to run while that variable still says egl.
        os.environ["PYOPENGL_PLATFORM"] = backend
        _drop_renderer_modules()
        try:
            module = importlib.import_module("mujoco.rendering.classic.renderer")
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            # EGL on PyOpenGL 3.1.0 raises AttributeError (no EGLDeviceEXT),
            # not ImportError. Try the next backend.
            errors.append(f"{backend}: {exc}")
            _drop_renderer_modules()
            continue
        renderer = getattr(module, "Renderer", None)
        if renderer is None:
            errors.append(f"{backend}: Renderer was not exported")
            continue
        _RENDERER = renderer
        _BACKEND = backend
        return renderer
    raise PerceptionFailed(
        "Could not load a MuJoCo GL renderer. Set MUJOCO_GL=osmesa if EGL is unavailable.",
        details={"errors": errors},
    )


def gl_backend_name() -> str:
    """Return the backend selected by :func:`renderer_class`.

    Returns:
        ``egl`` or ``osmesa``.

    Raises:
        PerceptionFailed: No renderer could be loaded.
    """
    renderer_class()
    return _BACKEND or "osmesa"


def _drop_renderer_modules() -> None:
    """Drop renderer and PyOpenGL modules so the next backend can import.

    PyOpenGL locks its platform on first import. An EGL failure leaves that
    platform in place, and OSMesa will not load until those modules are gone.
    The core ``mujoco`` module is left imported.
    """
    for name in list(sys.modules):
        drop_mujoco = name in {"mujoco.renderer", "mujoco.egl", "mujoco.osmesa"} or name.startswith(
            ("mujoco.rendering", "mujoco.egl.", "mujoco.osmesa.")
        )
        drop_opengl = name == "OpenGL" or name.startswith("OpenGL.")
        if drop_mujoco or drop_opengl:
            del sys.modules[name]
