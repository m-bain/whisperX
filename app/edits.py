"""Turn-level transcript editing: pure, Flask-free segment transforms.

A *turn* is the on-screen unit the transcript view renders: a contiguous run of
``segments`` that share a speaker (``render.py`` groups them into one bubble via
:func:`group_turns`). Editing a turn **collapses** its segment range into a single
segment (start = first.start, end = last.end, ``words: []``, same speaker key) and
**splices** it back into the list. Per-word timestamps are lost on an edited turn;
the result view falls back to segment-level highlighting.

Nothing here mutates the original result — callers persist the returned state and
delta as a non-destructive overlay (see :class:`app.store.SessionStore`). A *delta*
records exactly what a single edit replaced so :func:`undo_last` can reverse it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# Keep at most this many deltas; older edits roll off (current state stays
# materialized, so they simply become non-undoable).
HISTORY_LIMIT = 100


@dataclass
class Turn:
    """A contiguous run of same-speaker segments — one transcript bubble."""

    index: int                 # position in the turn list (the editable id)
    speaker: Optional[str]     # raw key e.g. "SPEAKER_00", or None (undiarized)
    start: Optional[float]     # first segment's start
    end: Optional[float]       # last segment's end
    seg_indices: list[int]     # contiguous indices into segments[] this turn covers
    text: str                  # segments' text joined (editor prefill)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seg_key(seg: dict) -> Optional[str]:
    """Grouping key: the raw speaker, with any falsy value (None/"") normalized to
    None so undiarized segments form their own run — matching render's ``raw or
    'Speaker'`` collapse."""
    return seg.get("speaker") or None


def group_turns(segments: list[dict]) -> list[Turn]:
    """Group consecutive same-speaker segments into turns.

    Grouping spans *all* segments (including empty-text ones) so every turn covers a
    contiguous ``[i..j]`` index range — which keeps splicing trivial and keeps turn
    indices identical between rendering and editing.
    """
    turns: list[Turn] = []
    for i, seg in enumerate(segments):
        key = _seg_key(seg)
        if turns and turns[-1].speaker == key:
            turns[-1].seg_indices.append(i)
        else:
            turns.append(Turn(index=len(turns), speaker=key, start=None, end=None,
                              seg_indices=[i], text=""))

    for t in turns:
        first, last = t.seg_indices[0], t.seg_indices[-1]
        t.start = segments[first].get("start")
        t.end = segments[last].get("end")
        parts = [(segments[k].get("text") or "").strip() for k in t.seg_indices]
        t.text = " ".join(p for p in parts if p)
    return turns


def apply_turn_edit(segments: list[dict], turn_index: int, new_text: str):
    """Replace turn ``turn_index``'s text, collapsing its segment range.

    Returns ``(new_segments, delta)``. An empty/whitespace ``new_text`` deletes the
    turn (its segments are spliced out). Raises ``IndexError`` for an unknown turn.
    """
    # Reject negatives explicitly — a stray -1 would silently edit the last turn.
    if turn_index < 0:
        raise IndexError(f"turn_index out of range: {turn_index}")
    turn = group_turns(segments)[turn_index]
    i, j = turn.seg_indices[0], turn.seg_indices[-1]
    old_segments = copy.deepcopy(segments[i:j + 1])

    text = (new_text or "").strip()
    if text:
        new_seg: dict = {"start": turn.start, "end": turn.end, "text": text, "words": []}
        if turn.speaker is not None:
            new_seg["speaker"] = turn.speaker
        replacement = [new_seg]
        new_segment = copy.deepcopy(new_seg)
    else:
        replacement = []
        new_segment = None

    new_segments = list(segments[:i]) + replacement + list(segments[j + 1:])
    delta = {
        "ts": _now(),
        "turn_index": turn_index,
        "seg_range": [i, j],
        "old_segments": old_segments,
        "new_segment": new_segment,
    }
    return new_segments, delta


def undo_last(segments: list[dict], history: list[dict]):
    """Reverse the most recent edit (strict LIFO).

    Returns ``(new_segments, new_history)``. Empty history is a no-op. The delta's
    replacement currently occupies ``[i : i+repl_len]`` (``repl_len`` is 1, or 0 for a
    deletion); its ``old_segments`` are spliced back in their place.
    """
    if not history:
        return list(segments), list(history)
    history = list(history)
    delta = history.pop()
    i, _j = delta["seg_range"]
    repl_len = 1 if delta.get("new_segment") is not None else 0
    restored = copy.deepcopy(delta["old_segments"])
    new_segments = list(segments[:i]) + restored + list(segments[i + repl_len:])
    return new_segments, history
