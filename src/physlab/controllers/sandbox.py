"""Process boundary for controller code.

:mod:`physlab.controllers.codecheck` is a guardrail. This module is the
boundary. The controller runs in a spawned process with ``RLIMIT_CPU`` and a
1 GiB ``RLIMIT_AS``. The parent enforces a wall-clock timeout and owns the
simulator. The child never receives a MuJoCo model.
"""

from __future__ import annotations

import resource
import sys
import traceback
from collections.abc import Mapping
from multiprocessing.connection import Connection
from typing import cast

from physlab.controllers.api import Controller, ControllerContext

_ADDRESS_BYTES = 1 << 30


def apply_limits(cpu_s: int) -> None:
    """Apply the CPU and address-space limits. Called only in the child.

    Args:
        cpu_s: Soft and hard ``RLIMIT_CPU``, seconds. At least one.
    """
    seconds = max(1, cpu_s)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    resource.setrlimit(resource.RLIMIT_AS, (_ADDRESS_BYTES, _ADDRESS_BYTES))


def child_main(
    source: str,
    conn: Connection,
    params: dict[str, float],
    dt_s: float,
    stdout_path: str,
    cpu_s: int,
) -> None:
    """Load ``source``, then answer observation messages until ``stop``.

    Args:
        source: Checked controller program.
        conn: Child end of the duplex pipe.
        params: Gains copied into the context.
        dt_s: Control period, seconds.
        stdout_path: File that receives the controller's stdout and stderr.
        cpu_s: CPU limit, seconds.
    """
    apply_limits(cpu_s)
    _capture_stdio(stdout_path)
    ctx = ControllerContext(dt_s, params)
    controller, phase, error = _construct(source, ctx)
    if controller is None:
        conn.send({"cmd": "error", "phase": phase, "traceback": error})
        return
    _serve(conn, controller, ctx)


def _construct(source: str, ctx: ControllerContext) -> tuple[Controller | None, str, str]:
    try:
        controller = _build(source)
    except Exception:
        return None, "IMPORT", traceback.format_exc()
    try:
        controller.setup(ctx)
    except Exception:
        return None, "SETUP", traceback.format_exc()
    return controller, "", ""


def _serve(conn: Connection, controller: Controller, ctx: ControllerContext) -> None:
    conn.send({"cmd": "ready", "assertions": _assertions(ctx)})
    phase = "STEP"
    try:
        while True:
            message = conn.recv()
            if message.get("cmd") == "stop":
                phase = "TEARDOWN"
                controller.teardown(ctx)
                conn.send({"cmd": "done", "assertions": _assertions(ctx)})
                return
            ctx.time_s = float(message["time_s"])
            action = controller.step(ctx, cast(Mapping[str, float], message["obs"]))
            conn.send(_action_message(ctx, action))
    except Exception:
        _send_error(conn, phase)


def _build(source: str) -> Controller:
    compiled = compile(source, "<controller>", "exec")
    namespace: dict[str, object] = {"__name__": "<controller>"}
    exec(compiled, namespace)
    found = [
        value
        for value in namespace.values()
        if isinstance(value, type) and issubclass(value, Controller) and value is not Controller
    ]
    if len(found) != 1:
        raise ImportError(f"expected one Controller subclass, found {len(found)}")
    return found[0]()


def _action_message(ctx: ControllerContext, action: Mapping[str, float]) -> dict[str, object]:
    numeric = {str(key): float(value) for key, value in action.items()}
    return {
        "cmd": "action",
        "action": numeric,
        "logs": ctx.drain_logs(),
        "assertions": _assertions(ctx),
    }


def _assertions(ctx: ControllerContext) -> list[tuple[float, str]]:
    rows = list(ctx.assertion_failures)
    ctx.assertion_failures.clear()
    return rows


def _send_error(conn: Connection, phase: str) -> None:
    conn.send({"cmd": "error", "phase": phase, "traceback": traceback.format_exc()})


def _capture_stdio(path: str) -> None:
    # Held until the child exits so prints land in the parent's stdout tail.
    handle = open(path, "w", buffering=1)  # noqa: SIM115
    sys.stdout = handle
    sys.stderr = handle
