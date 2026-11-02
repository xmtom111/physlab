"""Load the physlab configuration from YAML and the environment.

``load_config`` reads ``configs/default.yaml`` (or an explicit path) and applies
``PHYSLAB_`` environment variables. Nested fields use a double underscore:
``PHYSLAB_SIM__TIMESTEP_S=0.001`` overrides ``sim.timestep_s``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physlab.core.errors import ConfigError
from physlab.core.types import Mode

_ENV_PREFIX = "PHYSLAB_"


class GlBackend(StrEnum):
    """MuJoCo GL backend requested for headless or windowed rendering."""

    EGL = "egl"
    OSMESA = "osmesa"
    GLFW = "glfw"


class SimConfig(BaseModel):
    """Simulation settings that later tasks pass into the MuJoCo world.

    Attributes:
        timestep_s: Integrator step, seconds.
        render_width: Offscreen framebuffer width, pixels.
        render_height: Offscreen framebuffer height, pixels.
        gl_backend: Renderer. ``egl`` is the headless default; use ``osmesa``
            when EGL is unavailable.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    timestep_s: float = Field(default=0.002, gt=0)
    render_width: int = Field(default=640, gt=0)
    render_height: int = Field(default=480, gt=0)
    gl_backend: GlBackend = GlBackend.EGL


class PhysLabConfig(BaseModel):
    """Effective process configuration.

    Path fields are absolute after :func:`load_config`. Relative paths in the
    YAML file resolve against the project root, which is the parent of the
    ``configs`` directory when the file lives there.

    Attributes:
        mode: Simulation or real hardware.
        run_dir: Directory that holds per-run event logs.
        assets_dir: Directory of downloaded robot and mesh assets.
        configs_dir: Directory of YAML configuration.
        sim: Simulator settings.
        seed: Seed for deterministic random processes.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    mode: Mode
    run_dir: Path
    assets_dir: Path
    configs_dir: Path
    sim: SimConfig = Field(default_factory=SimConfig)
    seed: int = Field(default=0, ge=0)


def load_config(path: Path | None = None) -> PhysLabConfig:
    """Load configuration from YAML, then apply ``PHYSLAB_`` overrides.

    When ``path`` is omitted, the search starts at the working directory and
    walks parents for ``configs/default.yaml``. If that file is not found, the
    source checkout that contains this module is tried.

    Args:
        path: YAML file to load. ``None`` selects the default file.

    Returns:
        Validated configuration with absolute path fields.

    Raises:
        ConfigError: The file is missing, the YAML is invalid, or a field
            fails validation.
    """
    config_path = _resolve_config_path(path)
    merged = _merge(_read_yaml(config_path), _env_overrides(os.environ))
    try:
        loaded = PhysLabConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(
            "Invalid configuration",
            details={"path": str(config_path), "errors": _format_validation(exc)},
        ) from exc
    root = _project_root(config_path)
    return loaded.model_copy(
        update={
            "run_dir": _resolve_dir(loaded.run_dir, root),
            "assets_dir": _resolve_dir(loaded.assets_dir, root),
            "configs_dir": _resolve_dir(loaded.configs_dir, root),
        }
    )


def _resolve_config_path(path: Path | None) -> Path:
    if path is None:
        return _discover_config_path()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ConfigError("Configuration file not found", details={"path": str(resolved)})
    return resolved


def _discover_config_path() -> Path:
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / "configs" / "default.yaml"
        if candidate.is_file():
            return candidate
    checkout = _checkout_config_path()
    if checkout is not None:
        return checkout
    raise ConfigError("Could not find configs/default.yaml", details={"cwd": str(cwd)})


def _checkout_config_path() -> Path | None:
    module_path = Path(__file__).resolve()
    if len(module_path.parents) < 4 or module_path.parents[2].name != "src":
        return None
    candidate = module_path.parents[3] / "configs" / "default.yaml"
    if candidate.is_file():
        return candidate
    return None


def _project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _resolve_dir(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _read_yaml(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            "Could not read configuration",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(
            "Configuration YAML is invalid",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a mapping", details={"path": str(path)})
    return _normalize_mapping(loaded)


def _normalize_mapping(value: object) -> dict[str, object]:
    raw = _string_dict(value)
    return {key: _normalize_node(item) for key, item in raw.items()}


def _normalize_node(value: object) -> object:
    if isinstance(value, dict):
        return _normalize_mapping(value)
    if isinstance(value, list):
        return [_normalize_node(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ConfigError(
        "Unsupported configuration value",
        details={"type": type(value).__name__},
    )


def _string_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError("Configuration value must be a mapping")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(
                "Configuration keys must be strings",
                details={"key_type": type(key).__name__},
            )
        mapped[key] = item
    return mapped


def _merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(_string_dict(current), _string_dict(value))
        else:
            merged[key] = value
    return merged


def _env_overrides(environ: Mapping[str, str]) -> dict[str, object]:
    assignments: list[tuple[tuple[str, ...], str]] = []
    for key, raw in environ.items():
        parts = _env_parts(key)
        if parts:
            assignments.append((parts, raw))
    root: dict[str, object] = {}
    for parts, raw in sorted(assignments, key=lambda item: len(item[0])):
        _assign(root, parts, raw)
    return root


def _env_parts(key: str) -> tuple[str, ...] | None:
    if not key.startswith(_ENV_PREFIX):
        return None
    body = key.removeprefix(_ENV_PREFIX)
    parts = tuple(part.lower() for part in body.split("__") if part)
    if not parts:
        return None
    return parts


def _assign(node: dict[str, object], parts: Sequence[str], raw: str) -> None:
    head = parts[0]
    tail = parts[1:]
    if not tail:
        node[head] = raw
        return
    existing = node.get(head)
    if existing is None:
        child: dict[str, object] = {}
        node[head] = child
        _assign(child, tail, raw)
        return
    if isinstance(existing, dict):
        child_map = _string_dict(existing)
        node[head] = child_map
        _assign(child_map, tail, raw)
        return
    raise ConfigError(
        "Conflicting environment overrides",
        details={"key": _ENV_PREFIX + "__".join(part.upper() for part in parts)},
    )


def _format_validation(exc: ValidationError) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        formatted.append(
            {
                "loc": location,
                "msg": str(error["msg"]),
                "type": str(error["type"]),
            }
        )
    return formatted
