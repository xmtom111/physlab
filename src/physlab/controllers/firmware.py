"""Arduino CLI wrappers.

MicroPython generation is out of scope for this task. These helpers shell out
to ``arduino-cli`` when it is on ``PATH`` and raise
:class:`physlab.core.errors.DependencyMissing` when it is not. Tests pass a
fake executable on ``PATH`` and never invoke a real toolchain.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from physlab.core.errors import DependencyMissing, HardwareError


class ArduinoResult(BaseModel):
    """Captured ``arduino-cli`` output.

    Attributes:
        command: Arguments after the executable.
        returncode: Process status. Zero is success.
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def compile_arduino(sketch_dir: Path, fqbn: str) -> ArduinoResult:
    """Compile ``sketch_dir`` for ``fqbn``.

    Args:
        sketch_dir: Directory that contains the ``.ino`` sketch.
        fqbn: Fully qualified board name, for example ``arduino:avr:uno``.

    Returns:
        Captured CLI result.

    Raises:
        DependencyMissing: ``arduino-cli`` is not on ``PATH``.
        HardwareError: The CLI exited non-zero.
    """
    return _invoke(["compile", "--fqbn", fqbn, str(sketch_dir)])


def upload_arduino(sketch_dir: Path, fqbn: str, port: str) -> ArduinoResult:
    """Upload ``sketch_dir`` to ``port``.

    Args:
        sketch_dir: Directory that contains the compiled sketch.
        fqbn: Fully qualified board name.
        port: Serial port the board is attached to.

    Returns:
        Captured CLI result.

    Raises:
        DependencyMissing: ``arduino-cli`` is not on ``PATH``.
        HardwareError: The CLI exited non-zero.
    """
    return _invoke(["upload", "--fqbn", fqbn, "--port", port, str(sketch_dir)])


def _invoke(args: list[str]) -> ArduinoResult:
    binary = shutil.which("arduino-cli")
    if binary is None:
        raise DependencyMissing(
            "arduino-cli is not installed",
            details={"argv": args},
        )
    completed = subprocess.run(
        [binary, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    result = ArduinoResult(
        command=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if result.returncode != 0:
        raise HardwareError(
            "arduino-cli failed",
            details={"argv": args, "stderr": result.stderr, "returncode": result.returncode},
        )
    return result
