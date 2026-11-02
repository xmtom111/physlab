"""Controller API, static checks, and the process sandbox.

Controller source may import only :mod:`physlab.controllers.api`. The AST
check is a guardrail. The spawned process, with a CPU limit, a 1 GiB address
space, and a wall timeout, is the security boundary.
"""

from physlab.controllers.api import Controller, ControllerContext, DeviceManifest
from physlab.controllers.codecheck import CodeCheckResult, check_source
from physlab.controllers.runner import Phase, RunReport, run_controller

__all__ = [
    "CodeCheckResult",
    "Controller",
    "ControllerContext",
    "DeviceManifest",
    "Phase",
    "RunReport",
    "check_source",
    "run_controller",
]
