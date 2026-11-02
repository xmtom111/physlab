"""Mock-print a microscope, assemble it, and save the focused image."""

from __future__ import annotations

from physlab.assembly.bench import assemble_microscope
from physlab.core.config import load_config
from physlab.core.logging import configure_logging, get_logger


def main() -> None:
    """Run seed 0 and write the microscope image plus an MP4.

    Raises:
        SystemExit: Menagerie is missing or the assembly fails.
    """
    configure_logging("INFO", False)
    log = get_logger("examples.07_assemble_microscope")
    loaded = load_config()
    if not (loaded.assets_dir / "mujoco_menagerie" / ".git").is_dir():
        log.info("assets_missing", hint="physlab assets fetch")
        raise SystemExit(1)
    directory = loaded.run_dir / "07_assemble_microscope"
    report = assemble_microscope(0, save_dir=directory)
    image = directory / "microscope.png"
    video = directory / "assembly.mp4"
    log.info(
        "finished",
        success=report.success,
        failed_step=report.failed_step,
        image=str(image),
        video=str(video),
    )
    if not report.success or not image.is_file() or image.stat().st_size == 0:
        raise SystemExit(1)
    if not video.is_file() or video.stat().st_size == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
