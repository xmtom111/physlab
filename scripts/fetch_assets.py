"""Download MuJoCo Menagerie into ``assets/mujoco_menagerie``.

The commit comes from ``configs/assets.lock``. ``menagerie_commit: latest``
resolves ``main`` and writes the hash back. ``physlab assets fetch`` calls the
same function.
"""

from __future__ import annotations

from physlab.core.logging import configure_logging, get_logger
from physlab.robots.fetch import fetch_assets

_logger = get_logger(__name__)


def main() -> None:
    """Clone or update Menagerie and log the checked-out commit."""
    configure_logging("INFO", False)
    commit = fetch_assets()
    _logger.info("menagerie_ready", commit=commit)


if __name__ == "__main__":
    main()
