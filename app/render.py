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

from app.edits import group_turns


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


def resolve_label(raw: Optional[str], names: Optional[dict] = None) -> str:
    """User-assigned name for a speaker key if set, else the default label."""
    if names and raw and names.get(raw):
        return names[raw]
    return _speaker_label(raw)


# lucide "pencil" icon (https://lucide.dev/icons/pencil), inlined to avoid a dep.
_PENCIL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" '
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83'
    'l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/>'
    '<path d="m15 5 4 4"/></svg>'
)


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


def render_markdown(result: dict, names: Optional[dict] = None,
                    title: Optional[str] = None) -> str:
    """Render a result as a Markdown transcript: a title heading, then one block
    per speaker turn — a bold speaker tag with the turn's ``[start - end]`` span,
    followed by the turn text. ``names`` maps raw speaker keys to display names.

    Mirrors :func:`render_transcript`'s turn grouping so the export matches the
    on-screen transcript (including edits, since callers pass the overlaid segments).
    """
    lines: list[str] = [f"# {title.strip()}" if title and title.strip() else "# Transcript", ""]
    segments = result.get("segments", [])
    if not segments:
        lines.append("_No speech detected._")
        return "\n".join(lines) + "\n"

    for t in group_turns(segments):
        text = t.text.strip()
        if not text:
            continue
        label = resolve_label(t.speaker, names)
        span = f"{_fmt_ts(t.start)} – {_fmt_ts(t.end)}"
        lines.append(f"**{label}** [{span}]")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines) + "\n"


def render_transcript(result: dict, names: Optional[dict] = None) -> str:
    """Group consecutive segments by speaker into turn blocks of timed words.

    ``names`` maps a raw speaker key (e.g. ``SPEAKER_00``) to a user-assigned
    display name; overrides are applied here only, never written to the result.
    """
    segments = result.get("segments", [])
    if not segments:
        return '<p class="tr__empty">No speech detected.</p>'

    rows = []
    for t in group_turns(segments):
        # Turn indices come straight from group_turns so the edit endpoint and the
        # rendered DOM agree on what each `data-turn` refers to. Turns with no
        # visible text are skipped from the DOM but never renumber the rest.
        html = "".join(_word_spans(segments[k]) for k in t.seg_indices)
        if not html.strip():
            continue
        raw = t.speaker
        label = resolve_label(raw, names)
        # Only diarized turns (a real speaker key) get the rename affordance.
        if raw:
            key = escape(str(raw), quote=True)
            speaker_attr = f' data-speaker="{key}"'
            edit = (
                f'<button class="turn__edit" type="button" data-speaker="{key}" '
                f'data-name="{escape(label, quote=True)}" '
                'title="Edit speaker name" aria-label="Edit speaker name">'
                f'{_PENCIL_SVG}</button>'
            )
        else:
            speaker_attr = ""
            edit = ""
        rows.append(
            f'<div class="turn" data-turn="{t.index}">'
            '<div class="turn__who">'
            f'<div class="turn__speaker"{speaker_attr}>{escape(label)}</div>'
            f'{edit}'
            f'<div class="turn__time">{_fmt_ts(t.start)}</div>'
            "</div>"
            f'<div class="turn__text" data-text="{escape(t.text, quote=True)}">{html}</div>'
            "</div>"
        )
    return "\n".join(rows)
