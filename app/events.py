"""In-process pub/sub for pushing job progress to SSE clients.

The pipeline runs on the background job thread; browsers watch a session via an
``EventSource`` (Server-Sent Events). This broker bridges the two: the worker
``publish()``es stage/status events keyed by session id, and each open SSE
request holds a ``subscribe()``d queue it drains to the client. Purely in-memory
— the SQLite row stays the durable source of truth (used for the initial state
on connect and after a reload), this only carries the live deltas.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, Set


class Broker:
    def __init__(self) -> None:
        self._subs: Dict[str, Set[queue.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, session_id: str) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._subs.setdefault(session_id, set()).add(q)
        return q

    def unsubscribe(self, session_id: str, q: "queue.Queue") -> None:
        with self._lock:
            subs = self._subs.get(session_id)
            if subs is not None:
                subs.discard(q)
                if not subs:
                    del self._subs[session_id]

    def publish(self, session_id: str, event: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(session_id, ()))
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:  # slow/dead client — drop rather than block the worker
                pass
