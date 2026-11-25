"""Queue helpers shared by the pipeline threads."""
from __future__ import annotations

import queue
from typing import TypeVar

T = TypeVar("T")


def put_drop_oldest(q: queue.Queue[T], item: T) -> bool:
    """Put item without blocking; if full, discard the oldest item first.

    Returns True if something had to be dropped (the consumer is lagging).
    """
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass  # another producer refilled it; this item is dropped instead
    return True
