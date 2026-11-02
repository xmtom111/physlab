"""Build named parts and list their parameter schemas."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from physlab.core.config import load_config
from physlab.core.errors import ConfigError
from physlab.fabrication.models import ListedPart, PartArtifact
from physlab.fabrication.parts.library import part_entries


class _Builder(Protocol):
    def __call__(self, params: BaseModel, directory: Path) -> PartArtifact:
        """Build one part from validated parameters."""


def list_parts() -> list[ListedPart]:
    """Return every library part and the JSON schema of its parameters.

    Returns:
        Parts in registry order.
    """
    listed: list[ListedPart] = []
    for name, (model, _builder) in PART_LIBRARY.items():
        listed.append(ListedPart(name=name, json_schema=_schema(model)))
    return listed


def build_part(
    name: str,
    params: Mapping[str, object] | None = None,
    directory: Path | None = None,
) -> PartArtifact:
    """Validate ``params`` and build one part.

    Args:
        name: Library name.
        params: Parameter overrides. Omitted keys use the model defaults.
        directory: Where to write the STL and STEP files. Defaults to
            ``assets/generated/parts``.

    Returns:
        The exported artifact.

    Raises:
        ConfigError: ``name`` is unknown or ``params`` does not match the schema.
    """
    if name not in PART_LIBRARY:
        raise ConfigError(
            "Unknown part",
            details={"name": name, "parts": sorted(PART_LIBRARY)},
        )
    model, builder = PART_LIBRARY[name]
    try:
        parsed = model.model_validate(dict(params or {}))
    except ValidationError as exc:
        raise ConfigError(
            "Invalid part parameters",
            details={"name": name, "errors": str(exc)},
        ) from exc
    target = directory or (load_config().assets_dir / "generated" / "parts")
    return builder(parsed, target)


def _as_builder(builder: object) -> _Builder:
    if not callable(builder):
        raise ConfigError("Part builder is not callable")
    return cast(_Builder, builder)


# name -> (parameter model, builder). Filled once at import by the part modules.
PART_LIBRARY: dict[str, tuple[type[BaseModel], _Builder]] = {
    name: (model, _as_builder(builder)) for name, (model, builder) in part_entries().items()
}


def _schema(model: type[BaseModel]) -> dict[str, object]:
    # JSON Schema is an open document. The values are nested objects, arrays,
    # and scalars, so the public field stays ``dict[str, object]``.
    raw = model.model_json_schema()
    return cast(dict[str, object], raw)
