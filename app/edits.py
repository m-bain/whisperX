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
import difflib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# Keep at most this many deltas; older edits roll off (current state stays
# materialized, so they simply become non-undoable).
HISTORY_LIMIT = 100

# Target width (seconds) for a typed word with no audio timing of its own. When the
# gap between its timed neighbours is smaller than this, the word borrows the shortfall
# from those neighbours (which shrink, but never below this same floor).
MIN_WORD_WIDTH = 0.1

# A segment shorter than this is coalesced into a same-speaker neighbour (see
# coalesce_segments). Sub-second utterances/false-starts the recognizer split off get
# folded back into their turn so the stored transcript isn't littered with slivers.
SEGMENT_MIN_DURATION = 0.2


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


class NoChange(Exception):
    """Raised by an edit that would be a no-op, so callers skip a useless history
    entry (e.g. reassigning a turn to the speaker it already has)."""


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


def distinct_speakers(segments: list[dict]) -> list[str]:
    """Ordered unique non-null speaker keys, in first-appearance order."""
    seen: list[str] = []
    for seg in segments:
        key = _seg_key(seg)
        if key is not None and key not in seen:
            seen.append(key)
    return seen


def next_speaker_key(existing_keys) -> str:
    """Mint the next ``SPEAKER_<n>`` key (zero-padded to 2, matching pyannote).

    ``n`` is one past the highest ``SPEAKER_<n>`` index in ``existing_keys``;
    non-conforming keys are ignored. Falls back to ``SPEAKER_00`` when none match.
    """
    highest = -1
    for key in existing_keys or ():
        m = re.fullmatch(r"SPEAKER_(\d+)", str(key))
        if m:
            highest = max(highest, int(m.group(1)))
    return f"SPEAKER_{highest + 1:02d}"


def _interpolate_gaps(words: list[dict], start: Optional[float],
                      end: Optional[float]) -> None:
    """Give newly typed words a (rough) timestamp by spreading each run of untimed
    words evenly across the gap between its timed neighbours.

    Anchors: the previous timed word's ``end`` (or the turn ``start`` for a leading
    run) and the next timed word's ``start`` (or the turn ``end`` for a trailing run).
    When that gap is too narrow to give each typed word ``MIN_WORD_WIDTH``, the run
    *borrows* the shortfall from its neighbour words — split between the two sides —
    so a word inserted between tightly-packed words still gets a dwellable span. A
    lending neighbour never shrinks below ``MIN_WORD_WIDTH``. Mutates in place; a run
    with no usable anchor (or a backwards gap) is left untimed.
    """
    n = len(words)
    i = 0
    while i < n:
        if "start" in words[i]:
            i += 1
            continue
        j = i
        while j < n and "start" not in words[j]:
            j += 1
        run = j - i
        left_word = words[i - 1] if i > 0 and "end" in words[i - 1] else None
        right_word = words[j] if j < n and "start" in words[j] else None
        left = left_word["end"] if left_word else start
        right = right_word["start"] if right_word else end
        if left is None or right is None or right < left:
            i = j
            continue

        deficit = run * MIN_WORD_WIDTH - (right - left)
        if deficit > 0:
            # Each neighbour can lend down to MIN_WORD_WIDTH; aim for an even split,
            # then let either side cover whatever the other can't.
            cap_l = max(0.0, (left_word["end"] - left_word["start"]) - MIN_WORD_WIDTH) if left_word else 0.0
            cap_r = max(0.0, (right_word["end"] - right_word["start"]) - MIN_WORD_WIDTH) if right_word else 0.0
            take_l = min(deficit / 2, cap_l)
            take_r = min(deficit / 2, cap_r)
            take_l += min(cap_l - take_l, deficit - take_l - take_r)
            take_r += min(cap_r - take_r, deficit - take_l - take_r)
            if left_word:
                left_word["end"] -= take_l
            if right_word:
                right_word["start"] += take_r
            left -= take_l
            right += take_r

        step = (right - left) / run
        for k in range(run):
            words[i + k]["start"] = left + k * step
            words[i + k]["end"] = left + (k + 1) * step
        i = j


def realign_words(old_words: list[dict], new_text: str,
                  start: Optional[float] = None, end: Optional[float] = None) -> list[dict]:
    """Map edited text back onto a turn's words, keeping timestamps for tokens that
    survive the edit and interpolating timing for the ones the user typed.

    A token-level diff (whitespace tokens) between the turn's existing words and the
    new text: unchanged tokens keep their original ``start``/``end``; deleted tokens
    drop out; inserted tokens are given an interpolated span between their timed
    neighbours (see :func:`_interpolate_gaps`) so they stay lightly seekable. ``start``
    / ``end`` are the turn bounds, used as anchors for leading/trailing new words.

    Returns ``[]`` when no token keeps a real timestamp (e.g. a full rewrite, or the
    turn had no word timing to begin with) so the caller falls back to a single
    segment-timed span rather than a row of words timed from nothing.
    """
    new_tokens = new_text.split()
    old_tokens = [(w.get("word") or "").strip() for w in old_words]
    out: list[dict] = []
    kept_timing = False
    sm = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for oi, nj in zip(range(i1, i2), range(j1, j2)):
                ow = old_words[oi]
                w = {"word": new_tokens[nj]}
                wstart, wend = ow.get("start"), ow.get("end")
                if wstart is not None and wend is not None:
                    w["start"], w["end"] = wstart, wend
                    kept_timing = True
                out.append(w)
        elif tag in ("replace", "insert"):
            out.extend({"word": new_tokens[nj]} for nj in range(j1, j2))
        # "delete": the old token (and its timing) is gone — skip it.
    if not kept_timing:
        return []
    _interpolate_gaps(out, start, end)
    return out


def _seg_dur(seg: dict) -> float:
    start, end = seg.get("start"), seg.get("end")
    return (end - start) if start is not None and end is not None else 0.0


def _merge_segments(a: dict, b: dict) -> dict:
    """Fuse ``b`` onto ``a`` (a precedes b, same speaker): span both, concatenate text
    and words. Builds a fresh dict so inputs are never mutated."""
    merged = {
        "start": a.get("start") if a.get("start") is not None else b.get("start"),
        "end": b.get("end") if b.get("end") is not None else a.get("end"),
        "text": " ".join(t for t in ((a.get("text") or "").strip(),
                                      (b.get("text") or "").strip()) if t),
        "words": list(a.get("words") or []) + list(b.get("words") or []),
    }
    if a.get("speaker") is not None:
        merged["speaker"] = a["speaker"]
    return merged


def _coalesce_run(run: list[dict], threshold: float) -> list[dict]:
    """Coalesce one same-speaker run so every output segment clears ``threshold`` —
    unless the whole run is shorter, in which case a single sub-threshold segment
    remains (nothing same-speaker to merge it with)."""
    out: list[dict] = []
    cur: Optional[dict] = None
    for seg in run:
        cur = dict(seg) if cur is None else _merge_segments(cur, seg)
        if _seg_dur(cur) >= threshold:
            out.append(cur)
            cur = None
    if cur is not None:
        if out:
            out[-1] = _merge_segments(out[-1], cur)  # trailing sliver joins previous
        else:
            out.append(cur)                          # whole run < threshold: unavoidable
    return out


def coalesce_segments(segments: list[dict],
                      threshold: float = SEGMENT_MIN_DURATION) -> list[dict]:
    """Merge consecutive same-speaker segments shorter than ``threshold`` until each
    clears it (a lone short segment bordered by other speakers is left as-is).

    Same-speaker only, so the speaker-grouped turns and their order are unchanged — a
    turn that was N segments may become fewer (or one). Pure; returns a new list and
    never mutates the input. Idempotent: ``coalesce(coalesce(x)) == coalesce(x)``.
    """
    out: list[dict] = []
    n = len(segments)
    i = 0
    while i < n:
        key = segments[i].get("speaker") or None
        j = i
        while j < n and (segments[j].get("speaker") or None) == key:
            j += 1
        out.extend(_coalesce_run(segments[i:j], threshold))
        i = j
    return out


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
        old_words: list[dict] = []
        for k in turn.seg_indices:
            old_words.extend(segments[k].get("words") or [])
        # Keep timing for unchanged words; interpolate it for typed ones.
        words = realign_words(old_words, text, start=turn.start, end=turn.end)
        new_seg: dict = {"start": turn.start, "end": turn.end, "text": text, "words": words}
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


def apply_turn_reassign(segments: list[dict], turn_index: int, new_speaker: str):
    """Reassign turn ``turn_index`` to ``new_speaker``, rewriting the ``speaker`` key
    on its segment range (and on those segments' words, for export/JSON parity).

    Timing, text and word timestamps are preserved. After reassignment the turn may
    group with adjacent same-speaker turns (``group_turns`` regroups by speaker), so
    callers re-render the whole body. Returns ``(new_segments, delta)``.

    Raises ``IndexError`` for an unknown turn and :class:`NoChange` when the turn
    already has ``new_speaker`` (nothing to record).
    """
    # Reject negatives explicitly — a stray -1 would silently reassign the last turn.
    if turn_index < 0:
        raise IndexError(f"turn_index out of range: {turn_index}")
    turn = group_turns(segments)[turn_index]
    if turn.speaker == new_speaker:
        raise NoChange
    i, j = turn.seg_indices[0], turn.seg_indices[-1]
    old_segments = copy.deepcopy(segments[i:j + 1])

    replacement: list[dict] = []
    for seg in old_segments:
        new_seg = copy.deepcopy(seg)
        new_seg["speaker"] = new_speaker
        for word in new_seg.get("words") or []:
            word["speaker"] = new_speaker
        replacement.append(new_seg)

    new_segments = list(segments[:i]) + replacement + list(segments[j + 1:])
    delta = {
        "ts": _now(),
        "turn_index": turn_index,
        "seg_range": [i, j],
        "old_segments": copy.deepcopy(old_segments),
        "new_len": len(replacement),
    }
    return new_segments, delta


def undo_last(segments: list[dict], history: list[dict]):
    """Reverse the most recent edit (strict LIFO).

    Returns ``(new_segments, new_history)``. Empty history is a no-op. The delta's
    replacement currently occupies ``[i : i+repl_len]``; its ``old_segments`` are
    spliced back in their place. ``repl_len`` comes from the delta's ``new_len`` when
    present (a reassign replaces N segments with N), else falls back to the text-edit
    shape (1 segment, or 0 for a deletion) for older deltas without ``new_len``.
    """
    if not history:
        return list(segments), list(history)
    history = list(history)
    delta = history.pop()
    i, _j = delta["seg_range"]
    repl_len = delta.get("new_len")
    if repl_len is None:
        repl_len = 1 if delta.get("new_segment") is not None else 0
    restored = copy.deepcopy(delta["old_segments"])
    new_segments = list(segments[:i]) + restored + list(segments[i + repl_len:])
    return new_segments, history
