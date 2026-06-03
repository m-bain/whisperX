"""Single-worker background queue driving the transcription pipeline.

The SessionStore is the source of truth for status/results; this queue only
runs work on a serialized executor (max_workers=1, so CPU isn't oversubscribed
and uploads queue) and records state transitions on the store.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, store, run_session: Callable[[str], None]):
        """run_session(session_id): execute the pipeline and persist artifacts.

        It must call store.mark_done(...) itself (it has the result metadata);
        the queue handles mark_running and error capture.
        """
        self._store = store
        self._run_session = run_session
        self._executor = ThreadPoolExecutor(max_workers=1)

    def submit(self, session_id: str) -> None:
        self._executor.submit(self._run, session_id)

    def _run(self, session_id: str) -> None:
        self._store.mark_running(session_id)
        try:
            self._run_session(session_id)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("Session %s failed", session_id)
            self._store.mark_error(session_id, str(exc))
