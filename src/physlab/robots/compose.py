"""Build MuJoCo specs from catalog entries and attach arms to mobile bases.

Menagerie models are loaded with ``MjSpec.from_file``. A gripper listed in
``GripperSpec.attach_from`` is attached with ``MjSpec.attach`` and the prefix
``"{folder}/"``. Procedural robots are built with ``MjSpec`` only.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from physlab.core.errors import AssetMissing, ConfigError
from physlab.robots.models import RobotKind, RobotSpec, SimSource

RUNTIME_EE_SITE = "physlab_ee"
"""Site added on ``ee_body`` when the source model has no end-effector site."""

_MENAGERIE_DIR = "mujoco_menagerie"
_GAIN_SIZE = 10


def load_robot_spec(spec: RobotSpec, assets_dir: Path) -> mujoco.MjSpec:
    """Load ``spec`` as a MuJoCo spec, including an attached gripper.

    Args:
        spec: Catalog entry. Unavailable robots are rejected.
        assets_dir: Directory that contains ``mujoco_menagerie``.

    Returns:
        Spec ready to compile. A missing end-effector site is added on
        ``ee_body`` at ``ee_offset_m`` and named :data:`RUNTIME_EE_SITE`.

    Raises:
        AssetMissing: The robot is unavailable, the XML is missing, or MuJoCo
            cannot parse it.
        ConfigError: The procedural builder name is unknown.
    """
    if not spec.available:
        reason = spec.unavailable_reason or "robot is unavailable"
        raise AssetMissing(reason, details={"robot": spec.id})
    if spec.sim.source is SimSource.PROCEDURAL:
        model = _procedural(spec)
    else:
        model = _load_menagerie(spec, assets_dir)
        _attach_gripper(model, spec, assets_dir)
    _ensure_ee_site(model, spec)
    return model


def compose_mobile_manipulator(
    base: RobotSpec,
    arm: RobotSpec,
    name: str,
    assets_dir: Path,
) -> tuple[RobotSpec, mujoco.MjSpec]:
    """Attach ``arm`` to ``base.sim.mount_site`` and describe the combination.

    The arm is loaded first, so its gripper is already attached. The arm spec
    is then attached with an empty prefix, which keeps the arm's joint names.
    This is used for ``generic_diffdrive`` plus ``trs_so_arm100`` and
    ``ufactory_xarm7``.

    Args:
        base: Mobile base. ``sim.mount_site`` is required.
        arm: Arm to mount.
        name: Id of the composed robot.
        assets_dir: Menagerie root parent.

    Returns:
        The composed catalog entry and the MuJoCo spec.

    Raises:
        ConfigError: The base has no mount site.
        AssetMissing: Either model fails to load or compile.
    """
    if not base.sim.mount_site:
        raise ConfigError(
            "Base has no mount_site",
            details={"base": base.id},
        )
    base_model = load_robot_spec(base, assets_dir)
    arm_model = load_robot_spec(arm, assets_dir)
    site = base_model.site(base.sim.mount_site)
    if site is None or site.name != base.sim.mount_site:
        raise AssetMissing(
            "Mount site was not found on the base",
            details={"base": base.id, "site": base.sim.mount_site},
        )
    base_model.attach(arm_model, prefix="", site=site)
    _compile(base_model, name)
    return _composed_spec(base, arm, name), base_model


def gripper_prefix(spec: RobotSpec) -> str:
    """Return the ``MjSpec.attach`` prefix for this robot's gripper.

    Args:
        spec: Catalog entry.

    Returns:
        ``"{folder}/"`` when a separate gripper is attached, else ``""``.
    """
    gripper = spec.sim.gripper
    if gripper is None or gripper.attach_from is None:
        return ""
    return f"{gripper.attach_from}/"


def element_exists(model: mujoco.MjSpec, kind: str, name: str) -> bool:
    """Return whether ``name`` is a joint, actuator, site, body, or geom.

    Args:
        model: Spec to query.
        kind: One of ``joint``, ``actuator``, ``site``, ``body``, ``geom``.
        name: Element name.

    Returns:
        True when the spec lookup returns an element with that name.

    Raises:
        ConfigError: ``kind`` is not a known element type.
    """
    finders = {
        "joint": model.joint,
        "actuator": model.actuator,
        "site": model.site,
        "body": model.body,
        "geom": model.geom,
    }
    finder = finders.get(kind)
    if finder is None:
        raise ConfigError("Unknown MuJoCo element kind", details={"kind": kind})
    found = finder(name)
    return found is not None and getattr(found, "name", None) == name


def _procedural(spec: RobotSpec) -> mujoco.MjSpec:
    builder_name = spec.sim.procedural_builder or ""
    builders = {
        "generic_diffdrive": _build_diffdrive,
        "gantry_xyz": _build_gantry,
    }
    builder = builders.get(builder_name)
    if builder is None:
        raise ConfigError(
            "Unknown procedural robot builder",
            details={"builder": builder_name, "robot": spec.id},
        )
    return builder()


def _load_menagerie(spec: RobotSpec, assets_dir: Path) -> mujoco.MjSpec:
    path = _xml_path(assets_dir, spec.sim.menagerie_dir or "", spec.sim.xml_file or "")
    return _from_file(path)


def _attach_gripper(model: mujoco.MjSpec, spec: RobotSpec, assets_dir: Path) -> None:
    gripper = spec.sim.gripper
    if gripper is None or gripper.attach_from is None:
        return
    folder = assets_dir / _MENAGERIE_DIR / gripper.attach_from
    xml_file = _sole_model_xml(folder)
    child = _from_file(folder / xml_file)
    site_name = spec.sim.ee_site
    site = model.site(site_name) if site_name else None
    if site is None or site.name != site_name:
        raise AssetMissing(
            "Cannot attach a gripper without an arm site",
            details={"robot": spec.id, "site": site_name},
        )
    model.attach(child, prefix=gripper_prefix(spec), site=site)


def _ensure_ee_site(model: mujoco.MjSpec, spec: RobotSpec) -> None:
    if spec.sim.ee_site is not None or spec.sim.ee_body is None:
        return
    body = model.body(spec.sim.ee_body)
    if body is None or body.name != spec.sim.ee_body:
        raise AssetMissing(
            "End-effector body was not found",
            details={"robot": spec.id, "body": spec.sim.ee_body},
        )
    body.add_site(
        name=RUNTIME_EE_SITE,
        pos=list(spec.sim.ee_offset_m),
        size=[0.01, 0.01, 0.01],
    )


def _xml_path(assets_dir: Path, folder: str, xml_file: str) -> Path:
    path = assets_dir / _MENAGERIE_DIR / folder / xml_file
    if not path.is_file():
        raise AssetMissing(
            "Robot XML is not in the fetched menagerie checkout",
            details={"path": str(path)},
        )
    return path


def _sole_model_xml(folder: Path) -> str:
    if not folder.is_dir():
        raise AssetMissing(
            "Gripper menagerie folder was not found",
            details={"path": str(folder)},
        )
    xml_files = sorted(
        path.name
        for path in folder.glob("*.xml")
        if not path.name.startswith("scene") and "mjx" not in path.name
    )
    if len(xml_files) != 1:
        raise AssetMissing(
            "Gripper folder must contain one model XML",
            details={"path": str(folder), "files": xml_files},
        )
    return xml_files[0]


def _from_file(path: Path) -> mujoco.MjSpec:
    try:
        return mujoco.MjSpec.from_file(str(path))
    except (OSError, ValueError) as exc:
        raise AssetMissing(
            "Could not parse robot XML",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def _compile(model: mujoco.MjSpec, name: str) -> None:
    try:
        model.compile()
    except (OSError, ValueError) as exc:
        raise AssetMissing(
            "Composed robot failed to compile",
            details={"robot": name, "reason": str(exc)},
        ) from exc


def _composed_spec(base: RobotSpec, arm: RobotSpec, name: str) -> RobotSpec:
    return RobotSpec(
        id=name,
        display_name=name,
        kind=RobotKind.MOBILE_MANIPULATOR,
        license=f"{base.license}; {arm.license}",
        sim=base.sim.model_copy(
            update={
                "source": SimSource.PROCEDURAL,
                "procedural_builder": None,
                "arm_joints": arm.sim.arm_joints,
                "arm_actuators": arm.sim.arm_actuators,
                "ee_site": arm.sim.ee_site,
                "ee_body": arm.sim.ee_body,
                "ee_offset_m": arm.sim.ee_offset_m,
                "home_qpos_rad": arm.sim.home_qpos_rad,
                "gripper": arm.sim.gripper,
                "base_type": base.sim.base_type,
                "base_joints": base.sim.base_joints,
                "base_actuators": base.sim.base_actuators,
                "mount_site": base.sim.mount_site,
            }
        ),
        real=arm.real,
        available=True,
    )


def _build_diffdrive() -> mujoco.MjSpec:
    """0.40 x 0.30 x 0.12 m chassis, two driven wheels, one passive caster."""
    spec = mujoco.MjSpec()
    spec.modelname = "generic_diffdrive"
    # Box size in MuJoCo is the half-extent. Full size is 0.40 x 0.30 x 0.12 m.
    chassis = spec.worldbody.add_body(name="chassis", pos=[0.0, 0.0, 0.11])
    chassis.add_geom(
        name="chassis_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.20, 0.15, 0.06],
    )
    chassis.add_site(name="mount", pos=[0.0, 0.0, 0.06], size=[0.01, 0.01, 0.01])
    _wheel(spec, chassis, "left_wheel", 0.15)
    _wheel(spec, chassis, "right_wheel", -0.15)
    caster = chassis.add_body(name="caster", pos=[-0.12, 0.0, -0.04])
    caster.add_joint(name="caster_ball", type=mujoco.mjtJoint.mjJNT_BALL)
    caster.add_geom(name="caster_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0.0, 0.0])
    return spec


def _wheel(spec: mujoco.MjSpec, chassis: mujoco.MjsBody, name: str, y_m: float) -> None:
    # Radius 0.05 m. The cylinder axis is rotated from z to y.
    wheel = chassis.add_body(name=name, pos=[0.0, y_m, -0.06])
    wheel.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 1.0, 0.0])
    wheel.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.05, 0.01, 0.0],
        quat=[0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
    )
    _velocity_actuator(spec, f"{name}_vel", name)


def _build_gantry() -> mujoco.MjSpec:
    """Three prismatic joints spanning 0.5 m, 0.5 m, and 0.3 m."""
    spec = mujoco.MjSpec()
    spec.modelname = "gantry_xyz"
    x_body = spec.worldbody.add_body(name="x_carriage")
    _slide(x_body, "x", [1.0, 0.0, 0.0], 0.5)
    y_body = x_body.add_body(name="y_carriage")
    _slide(y_body, "y", [0.0, 1.0, 0.0], 0.5)
    z_body = y_body.add_body(name="z_carriage")
    _slide(z_body, "z", [0.0, 0.0, 1.0], 0.3)
    z_body.add_site(name="tool", pos=[0.0, 0.0, 0.0], size=[0.01, 0.01, 0.01])
    _position_actuator(spec, "x", "x", 0.5)
    _position_actuator(spec, "y", "y", 0.5)
    _position_actuator(spec, "z", "z", 0.3)
    return spec


def _slide(body: mujoco.MjsBody, name: str, axis: list[float], travel_m: float) -> None:
    body.add_joint(
        name=name,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=axis,
        limited=1,
        range=[0.0, travel_m],
    )
    body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02])


def _velocity_actuator(spec: mujoco.MjSpec, name: str, joint: str) -> None:
    # Matches a MJCF ``<velocity kv="10" ctrlrange="-20 20"/>`` expansion.
    spec.add_actuator(
        name=name,
        type=mujoco.mjtActuator.mjACTUATOR_VELOCITY,
        target=joint,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_AFFINE,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=_pad([10.0]),
        biasprm=_pad([0.0, 0.0, -10.0]),
        ctrllimited=1,
        ctrlrange=[-20.0, 20.0],
    )


def _position_actuator(spec: mujoco.MjSpec, name: str, joint: str, travel_m: float) -> None:
    spec.add_actuator(
        name=name,
        type=mujoco.mjtActuator.mjACTUATOR_POSITION,
        target=joint,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_AFFINE,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=_pad([200.0]),
        biasprm=_pad([0.0, -200.0, -20.0]),
        ctrllimited=1,
        ctrlrange=[0.0, travel_m],
    )


def _pad(values: list[float]) -> list[float]:
    return values + [0.0] * (_GAIN_SIZE - len(values))
