"""Server-Sent Events transport: the in-process pub/sub broker plus the Flask
streaming-response helper every SSE endpoint is built on.

The pipeline runs on background threads; browsers watch a channel via an
``EventSource``. ``Broker`` bridges the two — a worker ``publish()``es events
keyed by channel, each open request drains a ``subscribe()``d queue. It is
purely in-memory: the SQLite row stays the durable source of truth (used for the
initial state on connect and after a reload), the broker only carries live
deltas.

``sse_response()`` owns the parts every endpoint must get right (subscribe, the
``finally: unsubscribe``, the keepalive frame, the no-buffering headers), so a
new stream only supplies what varies. Event-shaping and the route handlers stay
in ``server.py`` where the app state lives.
"""

from __future__ import annotations

import json
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


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def sse_response(broker: Broker, channel, *, initial=None, terminal=None,
                 pending=None, keepalive=15):
    """Build a streaming ``text/event-stream`` Response for a broker channel.

    Centralizes the parts every SSE endpoint must get right — subscribe, the
    ``finally: unsubscribe``, the keepalive comment frame, and the no-buffering
    headers — so a new stream only supplies what varies:

    - ``initial``:  ``callable() -> dict | None`` — state emitted right after
      subscribe (so it can't miss an event racing the subscribe). ``None`` skips.
    - ``terminal``: ``callable(event) -> bool`` — close after this event.
      ``None`` = persistent stream (no terminal; runs until the client leaves).
    - ``pending``:  ``callable() -> dict | None`` — late-subscriber replay; if it
      returns an event, emit it and close immediately (the work already finished
      before this client connected).
    """
    from flask import Response  # lazy: keeps Broker/sse_format importable without Flask

    def stream():
        q = broker.subscribe(channel)
        try:
            if pending is not None:
                replay = pending()
                if replay is not None:
                    yield sse_format(replay)
                    return
            if initial is not None:
                first = initial()
                if first is not None:
                    yield sse_format(first)
                    if terminal is not None and terminal(first):
                        return  # already terminal on connect (e.g. job done before subscribe)
            while True:
                try:
                    event = q.get(timeout=keepalive)
                except queue.Empty:
                    yield ": keepalive\n\n"  # comment frame keeps the connection warm
                    continue
                yield sse_format(event)
                if terminal is not None and terminal(event):
                    return
        finally:
            broker.unsubscribe(channel, q)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
