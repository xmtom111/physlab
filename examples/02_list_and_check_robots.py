"""List the robot catalog and check every available model."""

from __future__ import annotations

from physlab.core.config import load_config
from physlab.core.logging import configure_logging, get_logger
from physlab.robots.catalog import RobotCatalog


def main() -> None:
    """Print each robot, then validate models when Menagerie is present.

    Raises:
        SystemExit: An available robot failed its name or compile check.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.02_list_and_check_robots")
    loaded = load_config()
    catalog = RobotCatalog(loaded.configs_dir, loaded.assets_dir)
    for spec in catalog.list():
        gripper = spec.sim.gripper.actuator if spec.sim.gripper is not None else "-"
        dof = len(spec.sim.arm_joints) + len(spec.sim.base_joints)
        log.info(
            "robot",
            robot=spec.id,
            kind=spec.kind.value,
            available=spec.available,
            dof=dof,
            gripper=gripper,
        )
    if not (loaded.assets_dir / "mujoco_menagerie" / ".git").is_dir():
        log.info("assets_missing", hint="physlab assets fetch")
        return
    failed = False
    for spec in catalog.list():
        if not spec.available:
            continue
        result = catalog.check(spec.id)
        log.info("check", robot=result.robot_id, ok=result.ok, messages=list(result.messages))
        failed = failed or not result.ok
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
