"""Safety denials, dilution, titration, Beer-Lambert, and Opentrons text."""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from physlab.chem.labware import DECK_SLOTS, LabwareKind, empty_labware
from physlab.chem.opentrons_backend import OT2_API_LEVEL, OpentronsHttpClient, to_opentrons_protocol
from physlab.chem.protocol import ChemProtocol, ReagentFill, TransferStep
from physlab.chem.reactions import advance_kinetics, ph_of
from physlab.chem.safety import ChemSafetyPolicy, KineticPair
from physlab.chem.virtual_lab import VirtualChemLab
from physlab.chem.workflows import EQUIVALENCE_UL, dye_calibration_protocol, titration_protocol
from physlab.core.errors import SafetyViolation
from physlab.core.events import EventLog
from physlab.core.types import Mode
from physlab.experiments.analysis import fit_models
from physlab.experiments.runner import ExperimentRunner


def test_allowlist_is_only_the_named_reagents() -> None:
    policy = ChemSafetyPolicy.load()
    assert set(policy.allowlist.reagents) == {
        "water",
        "food_dye",
        "sodium_chloride",
        "acetic_acid",
        "sodium_hydroxide",
        "phenolphthalein",
    }
    assert policy.allowlist.reagents["acetic_acid"].max_concentration_m == pytest.approx(1.0)
    assert policy.allowlist.reagents["sodium_hydroxide"].max_concentration_m == pytest.approx(0.1)


def test_package_never_writes_the_allowlist() -> None:
    for path in Path("src/physlab").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "reagents.yaml" not in text:
            continue
        assert "open(" not in text or '"w"' not in text
        assert "write_text" not in text


def test_safety_denies_unknown_over_concentrated_and_incompatible() -> None:
    policy = ChemSafetyPolicy.load()
    unknown = _fill_protocol("sulfuric_acid", 0.01)
    strong = _fill_protocol("sodium_hydroxide", 0.2)
    mixed = _mix_dye_and_base()
    assert policy.check_protocol(unknown).allowed is False
    assert any("unknown reagent" in item for item in policy.check_protocol(unknown).violations)
    assert any("above" in item for item in policy.check_protocol(strong).violations)
    assert any("incompatible" in item for item in policy.check_protocol(mixed).violations)
    real = policy.check_protocol(_fill_protocol("water", 0.0), mode=Mode.REAL, approval_token=None)
    assert real.allowed is False
    hot = policy.check_protocol(_hot_protocol())
    assert any("exceeds" in item for item in hot.violations)


def test_serial_dilution_concentrations() -> None:
    lab = VirtualChemLab(seed=0)
    readings = lab.execute(dye_calibration_protocol())
    found = [reading.values["concentration_m"] for reading in readings]
    expected = [1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5]
    assert found == pytest.approx(expected, rel=1e-9)


def test_titration_equivalence_within_two_percent() -> None:
    readings = VirtualChemLab(seed=0).execute(titration_protocol())
    volumes = np.array([reading.values["titrant_ul"] for reading in readings])
    ph = np.array([reading.measurements["ph"] for reading in readings])
    index = int(np.argmax(np.diff(ph) / np.diff(volumes)))
    found = 0.5 * (volumes[index] + volumes[index + 1])
    assert abs(found - EQUIVALENCE_UL) / EQUIVALENCE_UL <= 0.02


def test_runner_recovers_epsilon_within_three_percent(tmp_path: Path) -> None:
    policy = ChemSafetyPolicy.load()
    result = ExperimentRunner(EventLog(tmp_path, seed=0)).run_chem(
        dye_calibration_protocol(),
        question="Beer-Lambert calibration",
        hypothesis="Absorbance is linear in concentration.",
        seed=0,
    )
    frame = pd.DataFrame([{**trial.values, **trial.measurements} for trial in result.trials])
    linear = next(
        item for item in fit_models(frame, "concentration_m", "absorbance") if item.name == "linear"
    )
    epsilon = linear.params["b"] / policy.allowlist.path_cm
    expected = policy.allowlist.reagents["food_dye"].epsilon_l_mol_cm[520]
    assert abs(epsilon - expected) / expected <= 0.03
    assert (result.directory / "protocol.json").is_file()
    assert (result.directory / "trials.parquet").is_file()


def test_opentrons_protocol_parses_and_client_uses_documented_paths() -> None:
    source = to_opentrons_protocol(dye_calibration_protocol(), api_level="2.22")
    ast.parse(source)
    assert "apiLevel': '2.22'" in source or 'apiLevel": "2.22"' in source
    default = to_opentrons_protocol(titration_protocol())
    ast.parse(default)
    assert OT2_API_LEVEL in default
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Opentrons-Version"] == "*"
        seen.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/protocols":
            return httpx.Response(201, json={"data": {"id": "protocol-1", "status": "ok"}})
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(201, json={"data": {"id": "run-1", "status": "idle"}})
        if request.method == "GET" and request.url.path == "/runs/run-1":
            return httpx.Response(200, json={"data": {"id": "run-1", "status": "succeeded"}})
        return httpx.Response(404, json={"data": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://robot:31950")
    api = OpentronsHttpClient(client)
    protocol_id = api.upload_protocol("dye.py", source)
    run_id = api.create_run(protocol_id)
    assert api.run_status(run_id) == "succeeded"
    assert seen == [("POST", "/protocols"), ("POST", "/runs"), ("GET", "/runs/run-1")]


def test_charge_balance_and_allowlisted_kinetics() -> None:
    policy = ChemSafetyPolicy.load()
    water = ph_of({}, 1000.0, policy.allowlist)
    assert water == pytest.approx(7.0, abs=0.01)
    acid = {"acetic_acid": 0.05 * 0.001}
    assert ph_of(acid, 1000.0, policy.allowlist) == pytest.approx(3.02, abs=0.05)
    pair = KineticPair(reactant="food_dye", product="water", k_per_s=1.0)
    updated = advance_kinetics(
        {"food_dye": 1.0}, 10.0, 25.0, [pair], set(policy.allowlist.reagents)
    )
    assert updated["food_dye"] < 1.0e-4
    blocked = advance_kinetics({"food_dye": 1.0}, 10.0, 25.0, [pair], {"food_dye"})
    assert blocked["food_dye"] == pytest.approx(1.0)


def test_deck_catalog_and_denied_protocol_does_not_run() -> None:
    assert tuple(str(index) for index in range(1, 12)) == DECK_SLOTS
    plate = empty_labware(LabwareKind.WELL_PLATE_96)
    assert plate.wells["A1"].max_volume_ul == 300.0
    assert len(plate.wells) == 96
    assert empty_labware(LabwareKind.TIPRACK_300UL).tips_remaining == 96
    with pytest.raises(SafetyViolation):
        VirtualChemLab(seed=0).execute(_fill_protocol("sulfuric_acid", 0.01))


def _fill_protocol(reagent: str, concentration_m: float) -> ChemProtocol:
    return ChemProtocol(
        id="fill",
        labware={"1": LabwareKind.TIPRACK_300UL, "2": LabwareKind.TUBE_RACK_15ML},
        reagents={
            "2:A1": ReagentFill(reagent=reagent, concentration_m=concentration_m, volume_ul=1000)
        },
        steps=[TransferStep(src="2:A1", dst="2:B1", volume_ul=100)],
    )


def _mix_dye_and_base() -> ChemProtocol:
    return ChemProtocol(
        id="clash",
        labware={"2": LabwareKind.TUBE_RACK_15ML, "1": LabwareKind.TIPRACK_300UL},
        reagents={
            "2:A1": ReagentFill(reagent="food_dye", concentration_m=1.0e-5, volume_ul=500),
            "2:B1": ReagentFill(reagent="sodium_hydroxide", concentration_m=0.05, volume_ul=500),
        },
        steps=[TransferStep(src="2:A1", dst="2:B1", volume_ul=100)],
    )


def _hot_protocol() -> ChemProtocol:
    from physlab.chem.protocol import IncubateStep

    return ChemProtocol(
        id="hot",
        labware={"2": LabwareKind.TUBE_RACK_15ML},
        reagents={"2:A1": ReagentFill(reagent="water", concentration_m=0.0, volume_ul=1000)},
        steps=[IncubateStep(duration_s=10, temp_c=90)],
    )
