"""Print service that calls a printer MCP server.

Tool names and argument names are read from ``configs/printers/printer_mcp.yaml``.
This module does not contain printer tool names.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import ClassVar, TypeVar

import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, ConfigDict, ValidationError

from physlab.core.config import load_config
from physlab.core.errors import ConfigError, HardwareError
from physlab.fabrication.inventory import Inventory
from physlab.fabrication.models import InventoryItem, PrintJob, PrintState, PrintStatus

_T = TypeVar("_T")


class McpServerSpec(BaseModel):
    """How to start the printer MCP server."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    command: str
    args: list[str] = []
    env: dict[str, str] = {}


class McpToolNames(BaseModel):
    """Role to server tool name. ``None`` means that role is not mapped."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    slice: str | None = None
    upload: str | None = None
    start: str | None = None
    status: str | None = None
    cancel: str | None = None


class McpResultFields(BaseModel):
    """Keys in a tool's JSON object for our status fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    state: str
    progress: str
    message: str
    eta_s: str


class PrinterMcpConfig(BaseModel):
    """YAML mapping for :class:`McpPrintService`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    server: McpServerSpec
    tools: McpToolNames
    arg_templates: dict[str, dict[str, str]]
    result_fields: McpResultFields


def load_printer_mcp(path: Path | None = None) -> PrinterMcpConfig:
    """Load the printer MCP mapping.

    Args:
        path: YAML file. Defaults to ``configs/printers/printer_mcp.yaml``.

    Returns:
        Validated mapping.

    Raises:
        ConfigError: The file is missing or does not match the schema.
    """
    config_path = path or (load_config().configs_dir / "printers" / "printer_mcp.yaml")
    if not config_path.is_file():
        raise ConfigError("Printer MCP config was not found", details={"path": str(config_path)})
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(
            "Could not read the printer MCP config",
            details={"path": str(config_path), "reason": str(exc)},
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(
            "Printer MCP YAML is invalid",
            details={"path": str(config_path), "reason": str(exc)},
        ) from exc
    try:
        return PrinterMcpConfig.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError(
            "Printer MCP config is invalid",
            details={"path": str(config_path), "errors": str(exc)},
        ) from exc


class McpPrintService:
    """Submit prints by calling the mapped MCP tools over stdio."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Load the mapping. The server starts on the first call.

        Args:
            config_path: Mapping YAML. Defaults to the config in ``configs``.
        """
        self.config = load_printer_mcp(config_path)
        self.inventory = Inventory()
        self._tools: dict[str, dict[str, object]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._names: dict[str, str] = {}

    def connect(self) -> dict[str, dict[str, object]]:
        """Start the server, list tools, and check the mapping.

        Returns:
            Server tool name to its input schema.

        Raises:
            ConfigError: A mapped tool name is not in the server's tool list.
                The message includes the names the server actually advertised.
        """
        self._ensure()
        return dict(self._tools)

    def submit(self, job: PrintJob) -> str:
        """Slice (when mapped), upload, and start ``job``.

        Args:
            job: Part and print settings.

        Returns:
            Job id taken from the mapped result field.
        """
        self._ensure()
        return self._call(self._submit(job))

    def status(self, job_id: str) -> PrintStatus:
        """Query the mapped status tool.

        Args:
            job_id: Id returned by :meth:`submit`.

        Returns:
            Parsed status.
        """
        self._ensure()
        return self._call(self._status(job_id))

    def cancel(self, job_id: str) -> None:
        """Call the mapped cancel tool.

        Args:
            job_id: Id returned by :meth:`submit`.
        """
        self._ensure()
        self._call(self._cancel(job_id))

    def inventory_entry(self, job_id: str) -> InventoryItem | None:
        """Return the row recorded when a job reports ``DONE``.

        Args:
            job_id: Id returned by :meth:`submit`.

        Returns:
            The row, or ``None`` before the job is done.
        """
        return self.inventory.get(job_id)

    def close(self) -> None:
        """Stop the server subprocess."""
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and self._loop is not None
            and self._stop is not None
        ):
            self._loop.call_soon_threadsafe(self._stop.set)
            thread.join(timeout=5)
        self._thread = None
        self._session = None

    def _ensure(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._thread_main, name="printer-mcp", daemon=True
            )
            self._thread.start()
        if not self._ready.wait(timeout=30):
            raise ConfigError("Printer MCP server did not connect")
        if self._error is not None:
            raise self._error
        if self._session is None:
            raise ConfigError("Printer MCP server is not connected")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())
        loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        server = self.config.server
        env = dict(os.environ)
        env.update(server.env)
        params = StdioServerParameters(command=server.command, args=list(server.args), env=env)
        try:
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                listed = await session.list_tools()
                self._remember_tools(listed)
                self._reject_missing()
                if self._error is not None:
                    self._ready.set()
                    return
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            if self._error is None:
                self._error = exc
            self._ready.set()
            if not isinstance(exc, Exception):
                raise

    def _remember_tools(self, listed: object) -> None:
        tools = getattr(listed, "tools", [])
        found: dict[str, dict[str, object]] = {}
        for tool in tools:
            schema = getattr(tool, "input_schema", None)
            if isinstance(schema, dict):
                found[str(tool.name)] = dict(schema)
        self._tools = found

    def _reject_missing(self) -> None:
        actual = sorted(self._tools)
        missing = [name for name in _mapped_names(self.config.tools) if name not in self._tools]
        if not missing:
            return
        listed = ", ".join(actual) if actual else "(none)"
        self._error = ConfigError(
            f"Printer MCP tool is not on the server. Server tools: {listed}",
            details={"missing": missing, "tools": actual},
        )

    def _call(self, coro: Coroutine[object, object, _T]) -> _T:
        if self._loop is None:
            raise ConfigError("Printer MCP server is not connected")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _submit(self, job: PrintJob) -> str:
        fields = _job_fields(job)
        if self.config.tools.slice:
            await self._invoke("slice", fields)
        uploaded = await self._invoke("upload", fields)
        job_id = str(uploaded[self.config.result_fields.job_id])
        self._names[job_id] = job.part.name
        started = dict(fields)
        started["job_id"] = job_id
        await self._invoke("start", started)
        return job_id

    async def _status(self, job_id: str) -> PrintStatus:
        payload = await self._invoke("status", {"job_id": job_id})
        status = _status_from(payload, self.config.result_fields)
        if status.state is PrintState.DONE and self.inventory.get(job_id) is None:
            self.inventory.add(
                InventoryItem(
                    part_instance_id=job_id,
                    part_name=self._names.get(job_id, ""),
                    aruco_id=None,
                    location="printer",
                    pose=None,
                )
            )
        return status

    async def _cancel(self, job_id: str) -> None:
        await self._invoke("cancel", {"job_id": job_id})

    async def _invoke(self, role: str, fields: dict[str, object]) -> dict[str, object]:
        if self._session is None:
            raise ConfigError("Printer MCP server is not connected")
        tool_name = getattr(self.config.tools, role)
        if not isinstance(tool_name, str) or not tool_name:
            raise ConfigError("Print tool is not mapped", details={"tool": role})
        arguments = _apply_template(self.config.arg_templates.get(role, {}), fields, role)
        result = await self._session.call_tool(tool_name, arguments)
        if result.is_error:
            raise HardwareError("Printer tool failed", details={"tool": role, "name": tool_name})
        return _payload(result)


def _mapped_names(tools: McpToolNames) -> list[str]:
    names: list[str] = []
    for role in ("slice", "upload", "start", "status", "cancel"):
        value = getattr(tools, role)
        if isinstance(value, str) and value:
            names.append(value)
    return names


def _job_fields(job: PrintJob) -> dict[str, object]:
    return {
        "stl_path": str(job.part.stl_path),
        "step_path": str(job.part.step_path),
        "part_name": job.part.name,
        "material": job.material,
        "quality": job.quality,
        "copies": job.copies,
    }


def _apply_template(
    template: dict[str, str],
    fields: dict[str, object],
    role: str,
) -> dict[str, object]:
    if not template:
        return dict(fields)
    payload: dict[str, object] = {}
    for our_name, arg_name in template.items():
        if our_name not in fields:
            raise ConfigError(
                "Argument template field is unknown",
                details={"tool": role, "field": our_name},
            )
        payload[arg_name] = fields[our_name]
    return payload


def _payload(result: object) -> dict[str, object]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return dict(structured)
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return dict(loaded)
    raise HardwareError("Printer tool returned no JSON object")


def _status_from(payload: dict[str, object], fields: McpResultFields) -> PrintStatus:
    try:
        state = PrintState(str(payload[fields.state]))
        progress = _number(payload[fields.progress])
    except (KeyError, ValueError) as exc:
        raise HardwareError(
            "Printer status could not be read",
            details={"keys": sorted(str(key) for key in payload)},
        ) from exc
    message = payload.get(fields.message, "")
    eta = payload.get(fields.eta_s)
    return PrintStatus(
        state=state,
        progress=progress,
        message=message if isinstance(message, str) else str(message),
        eta_s=None if eta is None else _number(eta),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("not a number")
    return float(value)
