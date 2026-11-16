"""Compact chemistry deck and illustrative, kinematic pipetting animation."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from physlab.chem.protocol import TransferStep
from physlab.chem.reactions import absorbance_of, ph_of
from physlab.chem.virtual_lab import VirtualChemLab
from physlab.sim.world import SimWorld

INDICATOR_PKA = 9.3
SLOTS = {"1": (0.04, 0.04), "2": (0.23, 0.04), "3": (0.04, 0.20)}


def well_position(location: str) -> tuple[float, float, float]:
    """Map a compact deck location to meters in world coordinates.

    Args:
        location: Slot and well, separated by a colon.

    Returns:
        Reachable XYZ position at the well bottom.
    """
    slot, well = location.split(":")
    x, y = SLOTS[slot]
    return x + (int(well[1:]) - 1) * 0.009, y + (ord(well[0]) - 65) * 0.009, 0.765


def liquid_rgba(ph: float, absorbance: float = 0.0, indicator: bool = True) -> np.ndarray:
    """Map indicator and dye to approximate channel transmittance.

    Args:
        ph: Equilibrium pH.
        absorbance: Beer-Lambert absorbance at 520 nm.
        indicator: Whether phenolphthalein is present.

    Returns:
        RGBA approximation, not a spectral rendering.
    """
    fraction = 1 / (1 + 10 ** (INDICATOR_PKA - ph)) if indicator else 0.0
    weight = 1 - 10 ** (-max(0.0, absorbance))
    solvent = np.array([0.96, 0.98, 1.0])
    color = np.array([1.0, 0.12, 0.45])
    ind = solvent * (1 - fraction) + color * fraction
    dye = solvent * (1 - weight) + color * weight
    return np.r_[ind * dye, np.clip(0.20 + 0.75 * (1 - (1 - fraction) * (1 - weight)), 0.20, 0.95)]


def liquid_height(volume_ul: float, capacity_ul: float = 300.0) -> float:
    """Compute liquid height in meters.

    Args:
        volume_ul: Current volume.
        capacity_ul: Well capacity.

    Returns:
        Height for a 12 mm deep well.
    """
    return max(0.0001, 0.92 * volume_ul / capacity_ul * 0.012)


def add_chem_deck(spec: mujoco.MjSpec) -> None:
    """Add fixed labware, independent liquid geoms, and a fixed tip.

    Args:
        spec: Uncompiled gantry scene.
    """
    for slot, (x, y) in SLOTS.items():
        spec.worldbody.add_geom(
            name=f"deck_{slot}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x + 0.0495, y + 0.0315, 0.756],
            size=[0.059, 0.041, 0.006],
            rgba=[0.75, 0.78, 0.85, 1],
            contype=0,
            conaffinity=0,
        )
        rows = "ABCDEFGH" if slot != "3" else "A"
        for row in rows:
            for col in range(1, 13):
                loc = f"{slot}:{row}{col}"
                pos = well_position(loc)
                spec.worldbody.add_geom(
                    name=f"rim_{loc}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=list(pos),
                    size=[0.0038, 0.001],
                    rgba=[0.25, 0.3, 0.4, 1],
                    contype=0,
                    conaffinity=0,
                )
                spec.worldbody.add_geom(
                    name=f"liquid_{loc}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=list(pos),
                    size=[0.0032, 0.0001],
                    rgba=[1, 1, 1, 0],
                    contype=0,
                    conaffinity=0,
                )
    spec.body("gantry/z_carriage").add_geom(
        name="pipette_tip",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=[0, 0, -0.0675],
        size=[0.002, 0.0675],
        rgba=[0.9, 0.9, 0.95, 0],
        contype=0,
        conaffinity=0,
    )


class GantryBridge:
    """Synchronize chemistry and retain animation frames for offline playback."""

    def __init__(self, world: SimWorld) -> None:
        """Attach a bridge.

        Args:
            world: Compiled chemistry gantry world.
        """
        self.world = world
        self.frames: list[np.ndarray] = []

    def sync(self, lab: VirtualChemLab) -> None:
        """Update liquid sizes/colors and refresh forward kinematics.

        Args:
            lab: Authoritative live chemistry state.
        """
        m = self.world.model
        for gid in range(m.ngeom):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith("liquid_"):
                m.geom_rgba[gid, 3] = 0
        for loc, well in lab._wells.items():
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"liquid_{loc}")
            if gid < 0:
                continue
            h = liquid_height(well.volume_ul, well.max_volume_ul)
            m.geom_size[gid, 1] = h / 2
            m.geom_pos[gid, 2] = well_position(loc)[2] + h / 2
            m.geom_rgba[gid] = liquid_rgba(
                ph_of(well.contents, well.volume_ul, lab.policy.allowlist)
                if well.volume_ul > 0
                else 7.0,
                absorbance_of(well.contents, well.volume_ul, 520, lab.policy.allowlist),
                well.contents.get("phenolphthalein", 0) > 0,
            )
            if well.volume_ul <= 0:
                m.geom_rgba[gid, 3] = 0
        mujoco.mj_forward(m, self.world.data)

    def animate(self, step: TransferStep, before: VirtualChemLab, after: VirtualChemLab) -> None:
        """Play pure XYZ source/target motion in meters, without a gripper.

        Args:
            step: Completed transfer.
            before: Chemistry before the transfer.
            after: Chemistry after the transfer.
        """
        self.sync(before)
        m, d = self.world.model, self.world.data
        tip = m.geom("pipette_tip").id
        m.geom_rgba[tip, 3] = 1
        aspirated = copy.deepcopy(before)
        aspirated._wells[step.src] = copy.deepcopy(after._wells[step.src])
        src, dst = well_position(step.src), well_position(step.dst)
        for index, (x, y, z) in enumerate(
            [
                (src[0], src[1], 1.04),
                (src[0], src[1], 0.90),
                (src[0], src[1], 1.04),
                (dst[0], dst[1], 1.04),
                (dst[0], dst[1], 0.90),
                (dst[0], dst[1], 1.04),
            ]
        ):
            start = d.qpos.copy()
            target = np.array([x, y, z - 0.9])
            for fraction in (0.33, 0.67, 1.0):
                for axis, value in zip(
                    "xyz", start[:3] * (1 - fraction) + target * fraction, strict=True
                ):
                    self.world.set_joint_position(f"gantry/{axis}", float(value))
                if index == 1:
                    self.sync(aspirated)
                if index == 4:
                    self.sync(after)
                self.frames.append(self.capture())
        m.geom_rgba[tip, 3] = 0
        # Park beside the deck so the endpoint well is visible from overhead.
        self.world.set_joint_position("gantry/x", 0.40)
        self.world.set_joint_position("gantry/y", 0.30)
        self.sync(after)

    def capture(self) -> np.ndarray:
        """Capture the refreshed overhead view.

        Returns:
            RGB pixels from the chemistry camera.
        """
        with mujoco.Renderer(self.world.model, height=240, width=320) as renderer:
            mujoco.mj_forward(self.world.model, self.world.data)
            renderer.update_scene(self.world.data, camera="chem_overhead")
            return renderer.render().copy()
