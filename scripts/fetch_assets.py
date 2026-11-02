"""Download third-party assets declared in ``configs/assets.lock``.

The downloader is implemented in task T2. This module is a placeholder so the
repository layout matches the project map.
"""

from __future__ import annotations


def fetch_assets() -> None:
    """Download pinned assets into the local ``assets`` directory.

    Raises:
        NotImplementedError: Always. Implemented in T2.
    """
    raise NotImplementedError("Implemented in T2")


def main() -> None:
    """Run the asset downloader from the command line."""
    fetch_assets()


if __name__ == "__main__":
    main()
