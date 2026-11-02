"""Track printed copies by instance id."""

from __future__ import annotations

from physlab.fabrication.models import InventoryItem


class Inventory:
    """Map ``part_instance_id`` to the copy's name, marker, and location."""

    def __init__(self) -> None:
        """Start with an empty inventory."""
        self._items: dict[str, InventoryItem] = {}

    def add(self, item: InventoryItem) -> InventoryItem:
        """Store ``item``.

        Args:
            item: Copy to record. Replaces an item with the same id.

        Returns:
            The stored item.
        """
        self._items[item.part_instance_id] = item
        return item

    def get(self, part_instance_id: str) -> InventoryItem | None:
        """Return the copy named ``part_instance_id``.

        Args:
            part_instance_id: Id assigned when the copy was recorded.

        Returns:
            The item, or ``None`` when it is absent.
        """
        return self._items.get(part_instance_id)

    def __len__(self) -> int:
        """Return how many copies are tracked."""
        return len(self._items)
