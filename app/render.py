"""Render a whisperx result dict into transcript HTML.

Produces speaker-grouped *turns*; within a turn every word becomes a
``<span class="seg" data-start data-end>`` so the transcript view can highlight
words live against the real audio's currentTime. Words without timestamps
(punctuation, numerals the aligner dropped) render as plain spans.
"""

from __future__ import annotations

import re
from html import escape
from typing import Optional


def _fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _speaker_label(raw: Optional[str]) -> str:
    """SPEAKER_00 -> 'Speaker 1'; pass through anything else."""
    if not raw:
        return "Speaker"
    m = re.fullmatch(r"SPEAKER_(\d+)", str(raw))
    return f"Speaker {int(m.group(1)) + 1}" if m else str(raw)


def _word_spans(seg: dict) -> str:
    """Render a segment's words as timed spans; fall back to the raw text."""
    words = seg.get("words") or []
    if not words:
        text = (seg.get("text") or "").strip()
        if not text:
            return ""
        start, end = seg.get("start"), seg.get("end")
        attrs = ""
        if start is not None and end is not None:
            attrs = f' data-start="{float(start):.3f}" data-end="{float(end):.3f}"'
        return f'<span class="seg"{attrs}>{escape(text)}</span> '

    out = []
    for w in words:
        token = (w.get("word") or "").strip()
        if not token:
            continue
        start, end = w.get("start"), w.get("end")
        if start is not None and end is not None:
            attrs = f' data-start="{float(start):.3f}" data-end="{float(end):.3f}"'
        else:
            attrs = ""
        out.append(f'<span class="seg"{attrs}>{escape(token)}</span> ')
    return "".join(out)


def render_transcript(result: dict) -> str:
    """Group consecutive segments by speaker into turn blocks of timed words."""
    segments = result.get("segments", [])
    if not segments:
        return '<p class="tr__empty">No speech detected.</p>'

    turns: list[dict] = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker"
        spans = _word_spans(seg)
        if not spans:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["html"] += spans
        else:
            turns.append({"speaker": speaker, "html": spans, "start": seg.get("start")})

    rows = []
    for t in turns:
        rows.append(
            '<div class="turn">'
            '<div class="turn__who">'
            f'<div class="turn__speaker">{escape(_speaker_label(t["speaker"]))}</div>'
            f'<div class="turn__time">{_fmt_ts(t["start"])}</div>'
            "</div>"
            f'<div class="turn__text">{t["html"]}</div>'
            "</div>"
        )
    return "\n".join(rows)
