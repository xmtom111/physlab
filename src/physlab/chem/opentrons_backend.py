"""Opentrons protocol text and the robot HTTP client.

Protocol API level defaults to ``2.28``, the maximum OT-2 level documented
for robot software 10.0.0. The HTTP client talks to port 31950 and sends
``Opentrons-Version``. Paths were taken from the official HTTP API reference.
"""

from __future__ import annotations

import httpx

from physlab.chem.labware import PIPETTE_MAX_UL, LabwareKind
from physlab.chem.protocol import (
    ChemProtocol,
    ChemStep,
    IncubateStep,
    MixStep,
    ReadAbsorbanceStep,
    ReadPhStep,
    TransferStep,
)
from physlab.core.errors import HardwareError

# https://docs.opentrons.com/python-api/versioning/ (retrieved 2026-11-02):
# Flex supports 2.15-2.30; OT-2 supports 2.0-2.28. apiLevel belongs in
# requirements or metadata, not both.
OT2_API_LEVEL = "2.28"
# https://docs.opentrons.com/http/api_reference.html (retrieved 2026-11-02):
# the robot serves this API on port 31950. Opentrons-Version must be 2 or
# higher, or "*" for the latest. POST /protocols takes multipart field
# "files". POST /runs takes {"data": {"protocolId": ...}}. GET /runs/{runId}
# returns data.status.
OT_HTTP_PORT = 31950
OT_VERSION_HEADER = "Opentrons-Version"

_LOAD_NAMES = {
    LabwareKind.WELL_PLATE_96: "corning_96_wellplate_360ul_flat",
    LabwareKind.TUBE_RACK_15ML: "opentrons_15_tuberack_falcon_15ml_conical",
    LabwareKind.TUBE_RACK_50ML: "opentrons_6_tuberack_falcon_50ml_conical",
    LabwareKind.RESERVOIR_12: "nest_12_reservoir_15ml",
    LabwareKind.TIPRACK_300UL: "opentrons_96_tiprack_300ul",
}


def to_opentrons_protocol(protocol: ChemProtocol, *, api_level: str = OT2_API_LEVEL) -> str:
    """Render a ChemProtocol as an OT-2 Python protocol.

    Args:
        protocol: Deck and steps to render.
        api_level: ``requirements["apiLevel"]``. The OT-2 maximum is ``2.28``.

    Returns:
        Python source. ``ast.parse`` accepts it. The file is not executed.
    """
    lines = [
        "from opentrons import protocol_api",
        "",
        "metadata = {",
        f"    'protocolName': {protocol.id!r},",
        "    'author': 'physlab',",
        "    'description': 'Generated chemistry protocol',",
        "}",
        f"requirements = {{'robotType': 'OT-2', 'apiLevel': {api_level!r}}}",
        "",
        "def run(protocol: protocol_api.ProtocolContext) -> None:",
    ]
    lines.extend(_load_lines(protocol))
    lines.extend(_step_lines(protocol))
    return "\n".join(lines) + "\n"


class OpentronsHttpClient:
    """Upload a protocol and read run status. Tests pass a MockTransport."""

    def __init__(self, client: httpx.Client, *, api_version: str = "*") -> None:
        """Store the HTTP client.

        Args:
            client: Client whose base URL is ``http://<robot>:31950``. The
                caller owns it. Tests use :class:`httpx.MockTransport`.
            api_version: Value of the ``Opentrons-Version`` header.
        """
        self._client = client
        self._headers = {OT_VERSION_HEADER: api_version}

    def upload_protocol(self, filename: str, source: str) -> str:
        """POST ``/protocols`` with the protocol file.

        Args:
            filename: Multipart filename.
            source: Protocol text.

        Returns:
            Protocol id from ``data.id``.
        """
        files = {"files": (filename, source.encode("utf-8"), "text/x-python")}
        response = self._client.post("/protocols", files=files, headers=self._headers)
        return _field(response, "id")

    def create_run(self, protocol_id: str) -> str:
        """POST ``/runs`` for an uploaded protocol.

        Args:
            protocol_id: Id returned by :meth:`upload_protocol`.

        Returns:
            Run id from ``data.id``.
        """
        body = {"data": {"protocolId": protocol_id}}
        response = self._client.post("/runs", json=body, headers=self._headers)
        return _field(response, "id")

    def run_status(self, run_id: str) -> str:
        """GET ``/runs/{runId}`` and return ``data.status``.

        Args:
            run_id: Id returned by :meth:`create_run`.

        Returns:
            Status string such as ``idle`` or ``succeeded``.
        """
        response = self._client.get(f"/runs/{run_id}", headers=self._headers)
        return _field(response, "status")


def _load_lines(protocol: ChemProtocol) -> list[str]:
    lines = ["    tip_racks = []"]
    for slot, kind in protocol.labware.items():
        load = _LOAD_NAMES[kind]
        ident = f"labware_{slot}"
        lines.append(f"    {ident} = protocol.load_labware({load!r}, {slot!r})")
        if kind is LabwareKind.TIPRACK_300UL:
            lines.append(f"    tip_racks.append({ident})")
    lines.append(
        "    pipette = protocol.load_instrument('p300_single_gen2', 'left', tip_racks=tip_racks)"
    )
    return lines


def _step_lines(protocol: ChemProtocol) -> list[str]:
    lines: list[str] = []
    for step in protocol.steps:
        lines.extend(_one_step(step))
    return lines


def _one_step(step: ChemStep) -> list[str]:
    if isinstance(step, TransferStep):
        return _transfer_lines(step)
    if isinstance(step, MixStep):
        labware, well = _ref(step.well)
        return [f"    pipette.mix({step.cycles}, {step.volume_ul}, {labware}[{well!r}])"]
    if isinstance(step, IncubateStep):
        return [f"    protocol.delay(seconds={step.duration_s})"]
    if isinstance(step, ReadAbsorbanceStep):
        wells = ", ".join(step.wells)
        text = f"read_absorbance {step.wavelength_nm} nm at {wells}"
        return [f"    protocol.comment({text!r})"]
    if isinstance(step, ReadPhStep):
        text = f"read_ph {step.well}"
        return [f"    protocol.comment({text!r})"]
    return []


def _transfer_lines(step: TransferStep) -> list[str]:
    source_labware, source_well = _ref(step.src)
    dest_labware, dest_well = _ref(step.dst)
    remaining = step.volume_ul
    lines: list[str] = []
    while remaining > 1e-9:
        chunk = min(PIPETTE_MAX_UL, remaining)
        remaining -= chunk
        mix = ""
        if step.mix_after and remaining <= 1e-9:
            mix = f", mix_after=(3, {chunk})"
        lines.append(
            f"    pipette.transfer({chunk}, {source_labware}[{source_well!r}], "
            f"{dest_labware}[{dest_well!r}], new_tip='always'{mix})"
        )
    return lines


def _ref(location: str) -> tuple[str, str]:
    slot, well = location.split(":", maxsplit=1)
    return f"labware_{slot}", well


def _field(response: httpx.Response, name: str) -> str:
    body: object = response.json()
    if not isinstance(body, dict):
        raise HardwareError("Opentrons response is not an object")
    data = body.get("data")
    if not isinstance(data, dict):
        raise HardwareError("Opentrons response has no data object")
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise HardwareError("Opentrons response missing field", details={"field": name})
    return value
