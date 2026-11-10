"""Background jobs for tool calls that can run longer than two seconds.

A tool returns ``{"job_id": ...}`` immediately. The work runs on a daemon
thread. ``get_job`` reports status, progress, and the result.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal

from physlab.core.errors import ConfigError
from physlab.mcp_server.envelope import failure, jsonable

Progress = Callable[[float], None]
JobStatus = Literal["queued", "running", "done", "error"]


class JobManager:
    """Run callables on worker threads and keep their latest snapshot."""

    def __init__(self) -> None:
        """Start with an empty table."""
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._next = 1

    def submit(self, work: Callable[[Progress], object]) -> str:
        """Start ``work`` and return its id.

        Args:
            work: Called with a progress function that accepts a fraction
                from 0 to 1. The return value is stored as the job result.

        Returns:
            New job id, ``job-1``, ``job-2``, and so on.
        """
        with self._lock:
            job_id = f"job-{self._next}"
            self._next += 1
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "result": None,
                "error": None,
            }
        thread = threading.Thread(
            target=self._run,
            args=(job_id, work),
            name=f"physlab-{job_id}",
            daemon=True,
        )
        thread.start()
        return job_id

    def get(self, job_id: str) -> dict[str, object]:
        """Return a copy of the snapshot for ``job_id``.

        Args:
            job_id: Id from :meth:`submit`.

        Returns:
            Status, progress in ``[0, 1]``, result, and error.

        Raises:
            ConfigError: ``job_id`` was never submitted.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise ConfigError("Unknown job", details={"job_id": job_id})
            return dict(record)

    def _run(self, job_id: str, work: Callable[[Progress], object]) -> None:
        self._update(job_id, status="running", progress=0.0)

        def progress(fraction: float) -> None:
            self._update(job_id, progress=min(1.0, max(0.0, float(fraction))))

        try:
            value = work(progress)
        except Exception as exc:
            error = failure(exc)["error"]
            self._update(job_id, status="error", progress=1.0, error=error)
            return
        self._update(job_id, status="done", progress=1.0, result=jsonable(value))

    def _update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.update(fields)
