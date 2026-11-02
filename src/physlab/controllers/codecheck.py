"""Static allowlist for controller source.

This check is a guardrail for accidents and for obvious escapes (imports,
``exec``, dunder attribute walks). It is not a security boundary. A determined
string can still hide a call the AST walk does not name. The process sandbox
in :mod:`physlab.controllers.sandbox` is the boundary: a separate process with
a CPU limit, a 1 GiB address-space limit, and a wall-clock timeout.
"""

from __future__ import annotations

import ast
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

_ALLOWED_MODULES = frozenset(
    {"math", "numpy", "dataclasses", "typing", "collections", "physlab.controllers.api"}
)
_FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "os",
        "sys",
        "subprocess",
        "socket",
        "threading",
        "time",
    }
)


class Violation(BaseModel):
    """One rejected construct.

    Attributes:
        line: 1-based source line.
        column: 0-based column, when the parser reported one.
        name: Forbidden name or module.
        message: What was rejected.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    line: int = Field(ge=1)
    column: int = Field(ge=0)
    name: str
    message: str


class CodeCheckResult(BaseModel):
    """Outcome of :func:`check_source`.

    Attributes:
        ok: True when ``violations`` is empty.
        violations: Rejected constructs, in source order.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    violations: list[Violation] = Field(default_factory=list)


def check_source(source: str) -> CodeCheckResult:
    """Parse ``source`` and reject anything outside the controller allowlist.

    Args:
        source: Controller program text.

    Returns:
        Violations with line numbers. A syntax error is one violation.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line = exc.lineno or 1
        return CodeCheckResult(
            ok=False,
            violations=[
                Violation(line=line, column=exc.offset or 0, name="syntax", message=exc.msg)
            ],
        )
    violations = [_visit(node) for node in ast.walk(tree)]
    found = [item for item in violations if item is not None]
    found.sort(key=lambda item: (item.line, item.column))
    return CodeCheckResult(ok=not found, violations=found)


def _visit(node: ast.AST) -> Violation | None:
    if isinstance(node, ast.Import):
        return _import(node)
    if isinstance(node, ast.ImportFrom):
        return _import_from(node)
    if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
        return _at(node, node.id, f"name {node.id} is forbidden")
    if isinstance(node, ast.Attribute) and _forbidden_attr(node.attr):
        return _at(node, node.attr, f"attribute {node.attr} is forbidden")
    if isinstance(node, ast.Call):
        return _call(node)
    return None


def _import(node: ast.Import) -> Violation | None:
    for alias in node.names:
        if not _module_allowed(alias.name):
            return _at(node, alias.name, f"import {alias.name} is not allowed")
    return None


def _import_from(node: ast.ImportFrom) -> Violation | None:
    module = node.module or ""
    if node.level or not _module_allowed(module):
        name = "." * node.level + module
        text = name or "a relative module"
        return _at(node, name or "relative", f"import from {text} is not allowed")
    return None


def _call(node: ast.Call) -> Violation | None:
    if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return None
    target = node.args[1].value
    if isinstance(target, str) and _forbidden_attr(target):
        return _at(node, target, f"getattr of {target} is forbidden")
    return None


def _module_allowed(name: str) -> bool:
    return any(name == allowed or name.startswith(allowed + ".") for allowed in _ALLOWED_MODULES)


def _forbidden_attr(name: str) -> bool:
    return name in _FORBIDDEN_NAMES or name.startswith("__")


def _at(node: ast.AST, name: str, message: str) -> Violation:
    line = getattr(node, "lineno", 1) or 1
    column = getattr(node, "col_offset", 0) or 0
    return Violation(line=line, column=column, name=name, message=message)
