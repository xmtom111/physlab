"""Ideal mixing, weak-acid charge balance, Beer-Lambert, and first-order decay.

The charge balance is ``[H+] + C_base = K_w/[H+] + [A-]`` with
``[A-] = C_acid K_a / ([H+] + K_a)``. ``scipy.optimize.brentq`` finds ``[H+]``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from physlab.chem.safety import Allowlist, KineticPair
from physlab.core.errors import ConfigError

_KW = 1.0e-14
_R_J_MOL_K = 8.314462618


def moles_from_fill(concentration_m: float, volume_ul: float) -> float:
    """Return solute moles in a fill.

    Args:
        concentration_m: Molarity.
        volume_ul: Volume, microliters.

    Returns:
        Moles.
    """
    return concentration_m * volume_ul * 1.0e-6


def transfer_contents(
    source: dict[str, float],
    volume_source_ul: float,
    volume_ul: float,
) -> dict[str, float]:
    """Return the moles moved by an ideal transfer.

    Args:
        source: Moles in the source well before the transfer.
        volume_source_ul: Source volume before the transfer.
        volume_ul: Volume moved.

    Returns:
        Moles removed from the source. Concentration is unchanged.
    """
    if volume_source_ul <= 0.0:
        return {}
    fraction = volume_ul / volume_source_ul
    return {name: amount * fraction for name, amount in source.items()}


def ph_of(contents: dict[str, float], volume_ul: float, allowlist: Allowlist) -> float:
    """Return the equilibrium pH of a well.

    Args:
        contents: Moles of each species.
        volume_ul: Liquid volume, microliters.
        allowlist: Reagent properties, including ``K_a`` and strong-base flags.

    Returns:
        pH.

    Raises:
        ConfigError: The well is empty or the charge balance has no root.
    """
    if volume_ul <= 0.0:
        raise ConfigError("Cannot read pH of an empty well")
    volume_l = volume_ul * 1.0e-6

    def residual(hydrogen: float) -> float:
        return _charge(hydrogen, contents, volume_l, allowlist)

    try:
        hydrogen = float(brentq(residual, 1.0e-14, 1.0))
    except ValueError as exc:
        raise ConfigError("Charge balance has no root in (1e-14, 1)") from exc
    return -math.log10(hydrogen)


def absorbance_of(
    contents: dict[str, float],
    volume_ul: float,
    wavelength_nm: int,
    allowlist: Allowlist,
) -> float:
    """Return the Beer-Lambert absorbance of a well.

    Args:
        contents: Moles of each species.
        volume_ul: Liquid volume, microliters.
        wavelength_nm: Wavelength, nanometers.
        allowlist: Supplies molar absorptivities and the path length.

    Returns:
        Absorbance. Species without an entry at this wavelength contribute 0.
    """
    if volume_ul <= 0.0:
        return 0.0
    volume_l = volume_ul * 1.0e-6
    total = 0.0
    for name, amount in contents.items():
        spec = allowlist.reagents.get(name)
        epsilon = 0.0 if spec is None else spec.epsilon_l_mol_cm.get(wavelength_nm, 0.0)
        total += epsilon * allowlist.path_cm * (amount / volume_l)
    return total


def advance_kinetics(
    contents: dict[str, float],
    duration_s: float,
    temp_c: float,
    pairs: list[KineticPair],
    allowed: set[str],
) -> dict[str, float]:
    """Apply allowlisted first-order conversions.

    Args:
        contents: Moles before the incubate step.
        duration_s: Seconds.
        temp_c: Celsius.
        pairs: Pairs from the reagent file.
        allowed: Reagent ids. A pair is skipped unless both names are in it.

    Returns:
        Updated moles. Pairs that are not fully allowlisted do nothing.
    """
    updated = dict(contents)
    for pair in pairs:
        if pair.reactant not in allowed or pair.product not in allowed:
            continue
        amount = updated.get(pair.reactant, 0.0)
        if amount <= 0.0:
            continue
        converted = amount * (1.0 - math.exp(-_rate(pair, temp_c) * duration_s))
        updated[pair.reactant] = amount - converted
        updated[pair.product] = updated.get(pair.product, 0.0) + converted
    return updated


def _charge(
    hydrogen: float, contents: dict[str, float], volume_l: float, allowlist: Allowlist
) -> float:
    cations = hydrogen
    anions = _KW / hydrogen
    for name, amount in contents.items():
        spec = allowlist.reagents.get(name)
        if spec is None:
            continue
        concentration = amount / volume_l
        if spec.strong_base:
            cations += concentration
        elif spec.ka is not None and spec.ka > 0.0:
            anions += concentration * spec.ka / (spec.ka + hydrogen)
    return cations - anions


def _rate(pair: KineticPair, temp_c: float) -> float:
    if pair.a_per_s is None or pair.ea_j_mol is None:
        return pair.k_per_s
    kelvin = temp_c + 273.15
    return pair.a_per_s * math.exp(-pair.ea_j_mol / (_R_J_MOL_K * kelvin))
