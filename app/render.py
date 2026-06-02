"""Render a whisperx result dict into transcript HTML (speaker-grouped turns)."""

from __future__ import annotations

from html import escape
from typing import Optional


def _fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def render_transcript(result: dict) -> str:
    """Group consecutive segments by speaker into turn blocks."""
    segments = result.get("segments", [])
    if not segments:
        return '<p class="empty">No speech detected.</p>'

    turns: list[dict] = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg.get("end", turns[-1]["end"])
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                }
            )

    rows = []
    for t in turns:
        rows.append(
            '<div class="turn">'
            f'<div class="meta"><span class="speaker">{escape(str(t["speaker"]))}</span>'
            f'<span class="ts">{_fmt_ts(t["start"])} – {_fmt_ts(t["end"])}</span></div>'
            f'<div class="text">{escape(t["text"])}</div>'
            "</div>"
        )
    return "\n".join(rows)
