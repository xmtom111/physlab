"""The two simulated chemistry experiments shipped with the package."""

from __future__ import annotations

from physlab.chem.labware import LabwareKind
from physlab.chem.protocol import (
    ChemProtocol,
    ReadAbsorbanceStep,
    ReadPhStep,
    ReagentFill,
    TransferStep,
)

DYE_WAVELENGTH_NM = 520.0
DYE_STOCK_M = 1.0e-4
ACID_M = 0.05
ACID_UL = 2000.0
BASE_M = 0.1
# n_acid / C_base = 0.05 mol/L * 0.002 L / 0.1 mol/L = 0.001 L.
EQUIVALENCE_UL = 1000.0


def dye_calibration_protocol() -> ChemProtocol:
    """Serial-dilute a food-dye stock and read it at 520 nm.

    Returns:
        Protocol whose wells are 1.0e-4, 5.0e-5, and 2.5e-5 M.
    """
    wells = ["2:A1", "2:A2", "2:A3", "2:A4"]
    steps: list[TransferStep | ReadAbsorbanceStep] = [
        TransferStep(src="3:A1", dst="2:A1", volume_ul=200),
        TransferStep(src="3:A1", dst="2:A2", volume_ul=100),
        TransferStep(src="3:A2", dst="2:A2", volume_ul=100, mix_after=True),
        TransferStep(src="2:A2", dst="2:A3", volume_ul=100),
        TransferStep(src="3:A2", dst="2:A3", volume_ul=100, mix_after=True),
        TransferStep(src="2:A3", dst="2:A4", volume_ul=100),
        TransferStep(src="3:A2", dst="2:A4", volume_ul=100, mix_after=True),
        ReadAbsorbanceStep(wells=wells, wavelength_nm=DYE_WAVELENGTH_NM),
    ]
    return ChemProtocol(
        id="dye_calibration",
        labware={
            "1": LabwareKind.TIPRACK_300UL,
            "2": LabwareKind.WELL_PLATE_96,
            "3": LabwareKind.RESERVOIR_12,
        },
        reagents={
            "3:A1": ReagentFill(reagent="food_dye", concentration_m=DYE_STOCK_M, volume_ul=1000),
            "3:A2": ReagentFill(reagent="water", concentration_m=0.0, volume_ul=2000),
        },
        steps=list(steps),
    )


def titration_protocol() -> ChemProtocol:
    """Titrate 0.05 M acetic acid with 0.1 M sodium hydroxide.

    Returns:
        Protocol that crosses the equivalence volume of 1000 µL.
    """
    steps: list[TransferStep | ReadPhStep] = [
        TransferStep(src="3:A1", dst="2:A1", volume_ul=800),
        ReadPhStep(well="2:A1"),
    ]
    for _ in range(20):
        steps.append(TransferStep(src="3:A1", dst="2:A1", volume_ul=20))
        steps.append(ReadPhStep(well="2:A1"))
    return ChemProtocol(
        id="acetic_titration",
        labware={
            "1": LabwareKind.TIPRACK_300UL,
            "2": LabwareKind.TUBE_RACK_15ML,
            "3": LabwareKind.TUBE_RACK_15ML,
        },
        reagents={
            "2:A1": ReagentFill(reagent="acetic_acid", concentration_m=ACID_M, volume_ul=ACID_UL),
            "3:A1": ReagentFill(reagent="sodium_hydroxide", concentration_m=BASE_M, volume_ul=5000),
        },
        steps=list(steps),
    )
