"""Print-service protocol and the build, check, submit pipeline."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from physlab.fabrication.cad import build_part
from physlab.fabrication.mesh_check import PrinterProfile, check_printable
from physlab.fabrication.models import (
    FabricationResult,
    InventoryItem,
    PrintJob,
    PrintState,
    PrintStatus,
)

_TERMINAL = {PrintState.DONE, PrintState.FAILED, PrintState.CANCELLED}


@runtime_checkable
class PrintService(Protocol):
    """Something that accepts a mesh and reports print progress."""

    def submit(self, job: PrintJob) -> str:
        """Queue ``job`` and return its id."""

    def status(self, job_id: str) -> PrintStatus:
        """Return the latest status of ``job_id``."""

    def cancel(self, job_id: str) -> None:
        """Cancel ``job_id`` if it has not finished."""


def fabricate(
    part_name: str,
    params: Mapping[str, object] | None,
    service: PrintService,
    printer_profile: PrinterProfile,
    *,
    timeout_s: float = 120.0,
) -> FabricationResult:
    """Build a part, reject it when it will not print, otherwise submit it.

    Args:
        part_name: Library name.
        params: Parameter overrides.
        service: Where the job is queued.
        printer_profile: Bed and nozzle limits.
        timeout_s: How long to poll before giving up.

    Returns:
        A result. ``issues`` is set when the mesh check fails or the job does
        not end in ``DONE``.
    """
    artifact = build_part(part_name, params)
    report = check_printable(artifact.stl_path, printer_profile)
    if not report.ok:
        return FabricationResult(ok=False, part=artifact, issues=list(report.issues))
    job_id = service.submit(PrintJob(part=artifact))
    status = _poll(service, job_id, timeout_s)
    entry = _inventory(service, job_id)
    ok = status.state is PrintState.DONE
    issues = [] if ok else [status.message or status.state.value]
    return FabricationResult(
        ok=ok,
        part=artifact,
        job_id=job_id,
        status=status,
        inventory=entry,
        issues=issues,
    )


def _poll(service: PrintService, job_id: str, timeout_s: float) -> PrintStatus:
    started = time.monotonic()
    while True:
        status = service.status(job_id)
        if status.state in _TERMINAL:
            return status
        if time.monotonic() - started > timeout_s:
            return PrintStatus(
                state=PrintState.FAILED, progress=status.progress, message="timed out"
            )
        time.sleep(0.01)


def _inventory(service: PrintService, job_id: str) -> InventoryItem | None:
    lookup = getattr(service, "inventory_entry", None)
    if lookup is None:
        return None
    found = lookup(job_id)
    if isinstance(found, InventoryItem):
        return found
    return None
