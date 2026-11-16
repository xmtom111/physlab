"""96-well titration microprotocols and check-only safety demonstrations."""

from physlab.chem.labware import LabwareKind
from physlab.chem.protocol import ChemProtocol, IncubateStep, ReadPhStep, ReagentFill, TransferStep

DECK = {
    "1": LabwareKind.TIPRACK_300UL,
    "2": LabwareKind.WELL_PLATE_96,
    "3": LabwareKind.RESERVOIR_12,
}
EQUIVALENCE_UL = 40.0  # 80 uL * 0.05 M / 0.1 M


def initial_protocol() -> ChemProtocol:
    """Build the initial acid and indicator load.

    Returns:
        Safe protocol with 90 uL in the titration well.
    """
    return ChemProtocol(
        id="visual_initial",
        labware=DECK,
        reagents={
            "2:A1": ReagentFill(reagent="acetic_acid", concentration_m=0.05, volume_ul=80),
            "3:A1": ReagentFill(reagent="sodium_hydroxide", concentration_m=0.1, volume_ul=1000),
            "3:A2": ReagentFill(reagent="phenolphthalein", concentration_m=0.001, volume_ul=100),
        },
        steps=[TransferStep(src="3:A2", dst="2:A1", volume_ul=10), ReadPhStep(well="2:A1")],
    )


def dose_protocol(volume_ul: float) -> ChemProtocol:
    """Build one dose and pH read without repeating any fill.

    Args:
        volume_ul: Coarse 10 uL or fine 2 uL dose.

    Returns:
        Incremental microprotocol.

    Raises:
        ValueError: The requested dose is outside the demo choices.
    """
    if volume_ul not in (2, 10):
        raise ValueError("Choose 10 uL coarse or 2 uL fine")
    return ChemProtocol(
        id="visual_dose",
        labware=DECK,
        reagents={},
        steps=[TransferStep(src="3:A1", dst="2:A1", volume_ul=volume_ul), ReadPhStep(well="2:A1")],
    )


def unsafe_protocols() -> dict[str, ChemProtocol]:
    """Build deliberately invalid protocols for checks only.

    Returns:
        Overflow, concentrated base, excessive heat, and incompatible mixture.
    """
    initial = initial_protocol()
    overflow = initial.model_copy(
        update={"steps": [TransferStep(src="3:A1", dst="2:A1", volume_ul=500)]}
    )
    strong = initial.model_copy(
        update={
            "reagents": {
                **initial.reagents,
                "3:A1": ReagentFill(reagent="sodium_hydroxide", concentration_m=5, volume_ul=1000),
            }
        }
    )
    hot = initial.model_copy(update={"steps": [IncubateStep(duration_s=10, temp_c=80)]})
    mixture = initial.model_copy(
        update={
            "reagents": {
                **initial.reagents,
                "2:A1": ReagentFill(reagent="food_dye", concentration_m=0.0001, volume_ul=80),
            },
            "steps": [TransferStep(src="3:A1", dst="2:A1", volume_ul=10)],
        }
    )
    return {"overflow": overflow, "5 M base": strong, "80 C": hot, "dye + base": mixture}
