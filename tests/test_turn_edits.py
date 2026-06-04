"""Turn-level transcript editing: collapse / splice / undo behaviour.

The transcript view lets a user edit a whole speaker *turn* (a contiguous run of
same-speaker segments). Editing collapses that range into one segment and splices
it back; the original result is never mutated, and an overlay keeps the edited state
plus the last 100 deltas for undo. These tests pin the structural transform
(`app.edits`), its persistence (`app.store`), and the rendered markup (`app.render`).
"""

from __future__ import annotations

import copy

import pytest
from pytest import approx

from app.edits import (
    HISTORY_LIMIT,
    MIN_WORD_WIDTH,
    SEGMENT_MIN_DURATION,
    apply_turn_edit,
    coalesce_segments,
    group_turns,
    realign_words,
    undo_last,
)
from app.render import render_transcript
from app.store import SessionStore


# --- fixtures / helpers ------------------------------------------------------

def _seg(start, end, text, speaker=None, words=None):
    s = {"start": start, "end": end, "text": text}
    if speaker is not None:
        s["speaker"] = speaker
    if words is not None:
        s["words"] = words
    return s


def _word(token, start, end):
    return {"word": token, "start": start, "end": end}


@pytest.fixture
def diarized():
    """Two speakers, A spanning two segments then B then A again -> 3 turns."""
    return [
        _seg(0.0, 1.0, "Hello there.", "SPEAKER_00",
             [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)]),
        _seg(1.0, 2.0, "How are you?", "SPEAKER_00",
             [_word("How", 1.0, 1.4), _word("are", 1.4, 1.7), _word("you?", 1.7, 2.0)]),
        _seg(2.0, 3.0, "Fine thanks.", "SPEAKER_01",
             [_word("Fine", 2.0, 2.5), _word("thanks.", 2.5, 3.0)]),
        _seg(3.0, 4.0, "Good.", "SPEAKER_00", [_word("Good.", 3.0, 4.0)]),
    ]


# --- group_turns -------------------------------------------------------------

def test_group_single_speaker_is_one_turn():
    segs = [_seg(0, 1, "a", "SPEAKER_00"), _seg(1, 2, "b", "SPEAKER_00")]
    turns = group_turns(segs)
    assert len(turns) == 1
    t = turns[0]
    assert t.seg_indices == [0, 1]
    assert (t.start, t.end) == (0, 2)       # first.start, last.end
    assert t.text == "a b"
    assert t.speaker == "SPEAKER_00"


def test_group_alternating_speakers_ranges(diarized):
    turns = group_turns(diarized)
    assert [t.seg_indices for t in turns] == [[0, 1], [2], [3]]
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [t.index for t in turns] == [0, 1, 2]
    assert turns[0].start == 0.0 and turns[0].end == 2.0


def test_group_undiarized_run_is_one_turn():
    # No speaker key (and falsy "") collapse to a single None-speaker run.
    segs = [_seg(0, 1, "a"), _seg(1, 2, "b", ""), _seg(2, 3, "c")]
    turns = group_turns(segs)
    assert len(turns) == 1
    assert turns[0].speaker is None
    assert turns[0].seg_indices == [0, 1, 2]


def test_group_keeps_empty_segments_inside_contiguous_range():
    # An empty-text segment between two same-speaker segments stays in the range, so
    # seg_indices remain contiguous (clean splice).
    segs = [_seg(0, 1, "a", "SPEAKER_00"), _seg(1, 2, "", "SPEAKER_00"),
            _seg(2, 3, "c", "SPEAKER_00")]
    turns = group_turns(segs)
    assert turns[0].seg_indices == [0, 1, 2]
    assert turns[0].text == "a c"           # empty dropped only from the joined text


def test_group_empty_list_is_no_turns():
    assert group_turns([]) == []


def test_group_preserves_none_timestamps():
    turns = group_turns([_seg(None, None, "a", "SPEAKER_00")])
    assert turns[0].start is None and turns[0].end is None


# --- apply_turn_edit: collapse ----------------------------------------------

def test_edit_collapses_multi_segment_turn(diarized):
    # A full rewrite (no surviving words) collapses the turn's segment range into one
    # segment and falls back to segment-level timing (words == []).
    new_segments, _ = apply_turn_edit(diarized, 0, "Brand new sentence.")
    # turn 0 spanned segments [0,1] -> now one segment; total count drops by one.
    assert len(new_segments) == len(diarized) - 1
    edited = new_segments[0]
    assert edited["text"] == "Brand new sentence."
    assert edited["words"] == []                       # nothing survived -> fallback
    assert (edited["start"], edited["end"]) == (0.0, 2.0)  # spans first..last
    assert edited["speaker"] == "SPEAKER_00"           # speaker key preserved


def test_edit_full_rewrite_single_segment_falls_back_to_empty_words(diarized):
    new_segments, _ = apply_turn_edit(diarized, 1, "Not fine.")  # turn 1 == seg [2]
    assert len(new_segments) == len(diarized)          # 1 -> 1, no count change
    assert new_segments[2]["text"] == "Not fine."
    assert new_segments[2]["words"] == []              # no token survived
    assert new_segments[2]["speaker"] == "SPEAKER_01"


# --- apply_turn_edit: word-timing preservation (token diff) ------------------

def test_edit_appending_word_keeps_other_word_timing(diarized):
    # turn 0 words: Hello(0,.5) there.(.5,1) How(1,1.4) are(1.4,1.7) you?(1.7,2)
    new_segments, _ = apply_turn_edit(diarized, 0, "Hello there. How are you? Right.")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["Hello", "there.", "How", "are", "you?", "Right."]
    # non-adjacent originals keep their exact timestamps...
    assert (w[0]["start"], w[0]["end"]) == (0.0, 0.5)
    assert (w[3]["start"], w[3]["end"]) == (1.4, 1.7)   # "are" untouched
    # ...and the trailing typed word borrows width from "you?" (no room at the end).
    assert w[4]["start"] == 1.7 and w[4]["end"] == approx(1.9)
    assert (w[5]["start"], w[5]["end"]) == (approx(1.9), approx(2.0))
    assert (new_segments[0]["start"], new_segments[0]["end"]) == (0.0, 2.0)


def test_edit_inserting_word_midway_borrows_from_both_neighbours(diarized):
    new_segments, _ = apply_turn_edit(diarized, 0, "Hello there. How are really you?")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["Hello", "there.", "How", "are", "really", "you?"]
    # zero gap between are(.,1.7) and you?(1.7,.) -> each lends 0.05 to "really".
    assert (w[3]["start"], w[3]["end"]) == (1.4, approx(1.65))   # "are" lent a slice
    assert (w[4]["start"], w[4]["end"]) == (approx(1.65), approx(1.75))  # "really"
    assert (w[5]["start"], w[5]["end"]) == (approx(1.75), 2.0)   # "you?" lent a slice


def test_edit_deleting_words_keeps_survivors_timing(diarized):
    new_segments, _ = apply_turn_edit(diarized, 0, "Hello there. you?")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["Hello", "there.", "you?"]
    assert (w[2]["start"], w[2]["end"]) == (1.7, 2.0)   # survivor keeps timestamp


def test_edit_interpolates_inserted_word_across_a_gap():
    # A real gap between A(0..1) and B(2..3): the inserted word splits it evenly.
    segs = [_seg(0.0, 3.0, "A B", "SPEAKER_00",
                 [_word("A", 0.0, 1.0), _word("B", 2.0, 3.0)])]
    new_segments, _ = apply_turn_edit(segs, 0, "A new B")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["A", "new", "B"]
    assert (w[1]["start"], w[1]["end"]) == (1.0, 2.0)   # interpolated across the gap


def test_edit_interpolates_two_inserted_words_evenly():
    segs = [_seg(0.0, 4.0, "A B", "SPEAKER_00",
                 [_word("A", 0.0, 1.0), _word("B", 3.0, 4.0)])]
    new_segments, _ = apply_turn_edit(segs, 0, "A one two B")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["A", "one", "two", "B"]
    # gap 1.0..3.0 split across the two typed words -> 1.0-2.0, 2.0-3.0
    assert (w[1]["start"], w[1]["end"]) == (1.0, 2.0)
    assert (w[2]["start"], w[2]["end"]) == (2.0, 3.0)


def test_edit_interpolates_leading_and_trailing_words_from_turn_bounds():
    # Turn spans 0..3 but its one word sits at 1..2, leaving room either side.
    segs = [_seg(0.0, 3.0, "core", "SPEAKER_00", [_word("core", 1.0, 2.0)])]
    new_segments, _ = apply_turn_edit(segs, 0, "pre core post")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["pre", "core", "post"]
    assert (w[0]["start"], w[0]["end"]) == (0.0, 1.0)   # leading: turn start -> core
    assert (w[2]["start"], w[2]["end"]) == (2.0, 3.0)   # trailing: core -> turn end


def test_edit_borrows_width_for_word_in_zero_gap():
    # Adjacent words (A ends where B starts). The inserted word makes each lend a
    # sliver so it gets a real, dwellable span instead of zero width.
    segs = [_seg(0.0, 1.0, "A B", "SPEAKER_00",
                 [_word("A", 0.0, 0.5), _word("B", 0.5, 1.0)])]
    new_segments, _ = apply_turn_edit(segs, 0, "A mid B")
    w = new_segments[0]["words"]
    assert [x["word"] for x in w] == ["A", "mid", "B"]
    assert (w[0]["start"], w[0]["end"]) == (0.0, approx(0.45))   # A lent half
    assert (w[1]["start"], w[1]["end"]) == (approx(0.45), approx(0.55))
    assert w[1]["end"] - w[1]["start"] == approx(MIN_WORD_WIDTH)  # full dwellable width
    assert (w[2]["start"], w[2]["end"]) == (approx(0.55), 1.0)   # B lent half


def test_edit_borrow_respects_neighbour_floor():
    # B is only MIN_WORD_WIDTH wide already, so it can't lend; A covers the deficit.
    segs = [_seg(0.0, 0.6, "A B", "SPEAKER_00",
                 [_word("A", 0.0, 0.5), _word("B", 0.5, 0.6)])]
    new_segments, _ = apply_turn_edit(segs, 0, "A mid B")
    w = new_segments[0]["words"]
    assert w[2]["start"] == 0.5                       # B unchanged (at the floor)
    assert w[0]["end"] == approx(0.4)                 # A lent the full 0.1
    assert (w[1]["start"], w[1]["end"]) == (approx(0.4), approx(0.5))


def test_realign_words_returns_empty_without_old_timing():
    # A turn that never had word timing can't preserve any -> fall back to [].
    assert realign_words([], "some new text") == []
    assert realign_words([{"word": "x"}], "x y") == []


def test_edit_undiarized_turn_emits_no_speaker_key():
    segs = [_seg(0, 1, "a"), _seg(1, 2, "b")]
    new_segments, _ = apply_turn_edit(segs, 0, "merged")
    assert "speaker" not in new_segments[0]


def test_edit_does_not_mutate_input(diarized):
    snapshot = copy.deepcopy(diarized)
    apply_turn_edit(diarized, 0, "changed")
    assert diarized == snapshot                        # original list untouched


def test_edit_out_of_range_turn_raises(diarized):
    with pytest.raises(IndexError):
        apply_turn_edit(diarized, 5, "nope")


def test_edit_negative_turn_index_raises(diarized):
    # Guard: a stray -1 must NOT silently edit the last turn.
    with pytest.raises(IndexError):
        apply_turn_edit(diarized, -1, "nope")


def test_edit_on_empty_segments_raises():
    with pytest.raises(IndexError):
        apply_turn_edit([], 0, "nope")


def test_edit_same_turn_twice_overwrites(diarized):
    segs, _ = apply_turn_edit(diarized, 0, "first pass")
    segs, _ = apply_turn_edit(segs, 0, "second pass")   # re-edit the collapsed turn
    assert segs[0]["text"] == "second pass"
    assert segs[0]["words"] == []
    assert len(segs) == len(diarized) - 1               # still just one collapse


# --- apply_turn_edit: splice / insertion ------------------------------------

def test_edit_leaves_siblings_untouched(diarized):
    new_segments, _ = apply_turn_edit(diarized, 1, "Replaced.")  # middle turn, seg [2]
    # segment before the range (indices 0,1) and after (index 3) are value-identical.
    assert new_segments[0] == diarized[0]
    assert new_segments[1] == diarized[1]
    assert new_segments[-1] == diarized[3]
    # the replacement sits exactly where the old segment was.
    assert new_segments[2]["text"] == "Replaced."


def test_edit_inserts_at_correct_position(diarized):
    new_segments, _ = apply_turn_edit(diarized, 0, "X")
    assert new_segments[0]["text"] == "X"
    assert new_segments[1] == diarized[2]              # old seg[2] now follows directly
    assert new_segments[2] == diarized[3]


# --- empty edit == deletion (and re-merge) ----------------------------------

def test_empty_edit_deletes_turn(diarized):
    new_segments, delta = apply_turn_edit(diarized, 1, "   ")  # whitespace == delete
    assert len(new_segments) == len(diarized) - 1
    assert delta["new_segment"] is None
    assert all(s.get("speaker") != "SPEAKER_01" for s in new_segments)


def test_deleting_middle_turn_remerges_neighbours(diarized):
    # Deleting the SPEAKER_01 turn makes the surrounding SPEAKER_00 segments adjacent;
    # group_turns must then fuse them into a single turn.
    new_segments, _ = apply_turn_edit(diarized, 1, "")
    turns = group_turns(new_segments)
    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].seg_indices == [0, 1, 2]


def test_deleting_only_turn_yields_empty_and_undo_restores():
    segs = [_seg(0, 1, "lonely", "SPEAKER_00")]
    original = copy.deepcopy(segs)
    new_segments, delta = apply_turn_edit(segs, 0, "")
    assert new_segments == []
    assert group_turns(new_segments) == []
    restored, _ = undo_last(new_segments, [delta])
    assert restored == original


# --- delta integrity ---------------------------------------------------------

def test_delta_old_segments_is_independent_deepcopy(diarized):
    new_segments, delta = apply_turn_edit(diarized, 0, "X")
    assert delta["seg_range"] == [0, 1]
    assert delta["old_segments"] == diarized[0:2]
    # mutating the source afterwards must not bleed into the captured delta.
    diarized[0]["text"] = "MUTATED"
    assert delta["old_segments"][0]["text"] == "Hello there."


# --- coalesce_segments (small-segment second pass) --------------------------

def test_coalesce_merges_two_short_same_speaker_segments():
    segs = [_seg(0.0, 0.1, "uh", "SPEAKER_00", [_word("uh", 0.0, 0.1)]),
            _seg(0.1, 0.5, "hello there", "SPEAKER_00",
                 [_word("hello", 0.1, 0.3), _word("there", 0.3, 0.5)])]
    out = coalesce_segments(segs, threshold=0.2)
    assert len(out) == 1
    assert (out[0]["start"], out[0]["end"]) == (0.0, 0.5)
    assert out[0]["text"] == "uh hello there"
    assert [w["word"] for w in out[0]["words"]] == ["uh", "hello", "there"]
    assert out[0]["speaker"] == "SPEAKER_00"


def test_coalesce_until_threshold_then_stops():
    # 0.1 + 0.1 -> 0.2 (clears), the next 0.3 stands alone.
    segs = [_seg(0.0, 0.1, "a", "S", []), _seg(0.1, 0.2, "b", "S", []),
            _seg(0.2, 0.5, "c", "S", [])]
    out = coalesce_segments(segs, threshold=0.2)
    assert [s["text"] for s in out] == ["a b", "c"]
    assert all(s["end"] - s["start"] >= 0.2 for s in out)


def test_coalesce_does_not_cross_speakers():
    # A lone short segment bordered by other speakers can't merge -> stays short.
    segs = [_seg(0.0, 1.0, "long one", "SPEAKER_00", []),
            _seg(1.0, 1.1, "hi", "SPEAKER_01", []),
            _seg(1.1, 2.1, "long two", "SPEAKER_00", [])]
    out = coalesce_segments(segs, threshold=0.2)
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert out[1]["text"] == "hi"                       # untouched (isolated)


def test_coalesce_trailing_sliver_joins_previous():
    segs = [_seg(0.0, 0.3, "big", "S", []), _seg(0.3, 0.35, "x", "S", [])]
    out = coalesce_segments(segs, threshold=0.2)
    assert len(out) == 1 and out[0]["text"] == "big x"  # sliver folded back
    assert (out[0]["start"], out[0]["end"]) == (0.0, 0.35)


def test_coalesce_whole_run_under_threshold_stays_one():
    segs = [_seg(0.0, 0.05, "a", "S", []), _seg(0.05, 0.1, "b", "S", [])]
    out = coalesce_segments(segs, threshold=0.2)
    assert len(out) == 1                                # can't reach threshold, but one
    assert out[0]["text"] == "a b"


def test_coalesce_leaves_large_segments_untouched_and_is_idempotent():
    segs = [_seg(0.0, 1.0, "a", "S", []), _seg(1.0, 2.0, "b", "S", [])]
    once = coalesce_segments(segs, threshold=0.2)
    assert [s["text"] for s in once] == ["a", "b"]
    assert coalesce_segments(once, threshold=0.2) == once   # idempotent


def test_coalesce_default_threshold_is_segment_min_duration():
    segs = [_seg(0.0, 0.1, "a", "S", []), _seg(0.1, 0.9, "b", "S", [])]
    assert SEGMENT_MIN_DURATION == 0.2
    assert len(coalesce_segments(segs)) == 1            # uses the module default


def test_coalesce_does_not_mutate_input():
    segs = [_seg(0.0, 0.1, "a", "S", [_word("a", 0.0, 0.1)]),
            _seg(0.1, 0.5, "b", "S", [_word("b", 0.1, 0.5)])]
    snapshot = copy.deepcopy(segs)
    coalesce_segments(segs, threshold=0.2)
    assert segs == snapshot


# --- undo --------------------------------------------------------------------

def test_undo_round_trips_single_edit(diarized):
    original = copy.deepcopy(diarized)
    segs, delta = apply_turn_edit(diarized, 0, "Hi.")
    restored, history = undo_last(segs, [delta])
    assert restored == original
    assert history == []


def test_undo_two_edits_returns_original(diarized):
    original = copy.deepcopy(diarized)
    segs, d1 = apply_turn_edit(diarized, 0, "first")
    segs, d2 = apply_turn_edit(segs, 1, "second")   # turn 1 of the post-edit state
    history = [d1, d2]
    segs, history = undo_last(segs, history)        # undo second
    segs, history = undo_last(segs, history)        # undo first
    assert segs == original
    assert history == []


def test_undo_restores_a_deletion(diarized):
    original = copy.deepcopy(diarized)
    segs, delta = apply_turn_edit(diarized, 1, "")  # delete middle turn
    restored, _ = undo_last(segs, [delta])
    assert restored == original


def test_undo_empty_history_is_noop(diarized):
    segs, history = undo_last(diarized, [])
    assert segs == diarized
    assert history == []


def test_undo_two_edits_on_same_turn(diarized):
    original = copy.deepcopy(diarized)
    segs, d1 = apply_turn_edit(diarized, 0, "first")
    segs, d2 = apply_turn_edit(segs, 0, "second")
    segs, hist = undo_last(segs, [d1, d2])
    assert segs[0]["text"] == "first"                  # back to the first edit
    segs, hist = undo_last(segs, hist)
    assert segs == original                            # and back to pristine


def test_undo_after_index_shift_resolves_correctly(diarized):
    # Collapse turn 0 (4 segs -> 3), then edit a later turn against the shifted state.
    original = copy.deepcopy(diarized)
    segs, d1 = apply_turn_edit(diarized, 0, "merged")   # now segs: [merged, seg01, seg00b]
    segs, d2 = apply_turn_edit(segs, 2, "last edit")    # edit the trailing SPEAKER_00 turn
    assert segs[2]["text"] == "last edit"
    segs, hist = undo_last(segs, [d1, d2])
    segs, hist = undo_last(segs, hist)
    assert segs == original


# --- whitespace --------------------------------------------------------------

def test_edit_trims_outer_whitespace_keeps_inner(diarized):
    new_segments, _ = apply_turn_edit(diarized, 0, "  a   b  ")
    assert new_segments[0]["text"] == "a   b"


# --- render integration ------------------------------------------------------

def test_render_emits_data_turn(diarized):
    html = render_transcript({"segments": diarized})
    assert 'data-turn="0"' in html
    assert 'data-turn="1"' in html
    assert 'data-turn="2"' in html


def test_render_edited_turn_is_single_seg_span(diarized):
    segs, _ = apply_turn_edit(diarized, 0, "Collapsed line.")
    html = render_transcript({"segments": segs})
    # The edited turn renders as ONE timed span carrying the segment bounds.
    assert html.count('data-start="0.000"') == 1
    assert 'data-start="0.000" data-end="2.000">Collapsed line.</span>' in html


def test_render_appended_word_keeps_timed_spans(diarized):
    # Adding a word must NOT collapse the turn to one untimed span: the surviving
    # words keep their data-start spans and only the new word is plain.
    segs, _ = apply_turn_edit(diarized, 0, "Hello there. How are you? Right.")
    html = render_transcript({"segments": [segs[0]]})  # just the edited turn
    # All six words are timed: five originals + the interpolated typed word.
    assert html.count("data-start=") == 6
    assert 'data-start="0.000"' in html and 'data-start="1.700"' in html
    assert ">Right.</span>" in html


def test_render_speaker_override_survives_edit(diarized):
    segs, _ = apply_turn_edit(diarized, 0, "Edited.")
    html = render_transcript({"segments": segs}, {"SPEAKER_00": "Alice"})
    assert "Alice" in html                              # rename overlay still applies


def test_render_skips_empty_turn_without_renumbering():
    # A diarized-but-empty middle turn is omitted from the DOM, but the turns after it
    # keep their group_turns index — so editing by that index still hits the right
    # segment. This is the render/edit-index alignment guarantee.
    segs = [_seg(0, 1, "hi", "SPEAKER_00"),
            _seg(1, 2, "", "SPEAKER_01"),       # empty -> not rendered, but is turn 1
            _seg(2, 3, "bye", "SPEAKER_00")]    # -> turn 2
    html = render_transcript({"segments": segs})
    assert 'data-turn="0"' in html
    assert 'data-turn="1"' not in html          # skipped (no visible text)
    assert 'data-turn="2"' in html
    # The index the DOM exposes (2) edits the trailing segment, not the empty one.
    new_segments, _ = apply_turn_edit(segs, 2, "BYE")
    assert new_segments[2]["text"] == "BYE"
    assert new_segments[0] == segs[0] and new_segments[1] == segs[1]


def test_render_empty_transcript():
    assert "No speech detected" in render_transcript({"segments": []})


def test_render_escapes_edited_text(diarized):
    # Edited text is user input and must be HTML-escaped, not injected raw.
    segs, _ = apply_turn_edit(diarized, 0, '<script>alert("x")</script>')
    html = render_transcript({"segments": segs})
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


# --- store: overlay persistence ---------------------------------------------

def _make_store(tmp_path, segments):
    store = SessionStore(str(tmp_path))
    sid = "sess1"
    import json
    import os
    os.makedirs(store.session_dir(sid), exist_ok=True)
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f)
    return store, sid


def test_store_save_turn_edit_writes_overlay(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    assert store.load_edits(sid) is None                # none until first edit
    store.save_turn_edit(sid, 0, "Hi there.")
    edits = store.load_edits(sid)
    assert edits["version"] == 1
    assert len(edits["history"]) == 1
    cur = store.current_segments(sid, diarized)
    assert cur[0]["text"] == "Hi there."
    assert len(cur) == len(diarized) - 1


def test_store_original_result_is_never_mutated(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    store.save_turn_edit(sid, 0, "changed")
    assert store.load_result(sid)["segments"] == diarized


def test_store_current_segments_falls_back_to_original(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    assert store.current_segments(sid, diarized) == diarized


def test_store_undo_reverts_and_drops_overlay_when_pristine(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    store.save_turn_edit(sid, 0, "edit one")
    store.undo_turn_edit(sid)
    assert store.load_edits(sid) is None                # fully reverted -> file removed
    assert store.current_segments(sid, diarized) == diarized


def test_store_undo_noop_when_no_history(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    assert store.undo_turn_edit(sid) == diarized
    assert store.edit_history_len(sid) == 0


def test_store_history_capped_but_state_reflects_all_edits(tmp_path):
    # A long single-speaker turn we can edit many times in a row.
    segs = [_seg(0, 1, "start", "SPEAKER_00")]
    store, sid = _make_store(tmp_path, segs)
    for n in range(HISTORY_LIMIT + 1):                  # 101 edits
        store.save_turn_edit(sid, 0, f"edit {n}")
    edits = store.load_edits(sid)
    assert len(edits["history"]) == HISTORY_LIMIT       # oldest rolled off
    assert edits["segments"][0]["text"] == f"edit {HISTORY_LIMIT}"   # latest wins


def test_store_baseline_is_coalesced(tmp_path):
    # Two tiny same-speaker segments in the stored result -> current_segments merges
    # them, but the original transcript.json is left untouched.
    segs = [_seg(0.0, 0.1, "uh", "SPEAKER_00", [_word("uh", 0.0, 0.1)]),
            _seg(0.1, 0.6, "hello world", "SPEAKER_00",
                 [_word("hello", 0.1, 0.3), _word("world", 0.3, 0.6)])]
    store, sid = _make_store(tmp_path, segs)
    cur = store.current_segments(sid, segs)
    assert len(cur) == 1 and cur[0]["text"] == "uh hello world"
    assert store.load_result(sid)["segments"] == segs   # original intact


def test_store_edit_and_undo_roundtrip_with_coalescing(tmp_path):
    segs = [_seg(0.0, 0.1, "uh", "SPEAKER_00", [_word("uh", 0.0, 0.1)]),
            _seg(0.1, 0.6, "hello world", "SPEAKER_00",
                 [_word("hello", 0.1, 0.3), _word("world", 0.3, 0.6)]),
            _seg(0.6, 1.6, "fine thanks", "SPEAKER_01",
                 [_word("fine", 0.6, 1.0), _word("thanks", 1.0, 1.6)])]
    store, sid = _make_store(tmp_path, segs)
    baseline = store.current_segments(sid, segs)         # coalesced: 2 segments
    assert len(baseline) == 2
    store.save_turn_edit(sid, 0, "Hey hello world")      # edit the merged turn 0
    assert store.current_segments(sid, segs)[0]["text"] == "Hey hello world"
    store.undo_turn_edit(sid)
    assert store.current_segments(sid, segs) == baseline  # back to the coalesced base
    assert store.load_edits(sid) is None                  # overlay dropped


def test_store_unknown_turn_raises(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    with pytest.raises(IndexError):
        store.save_turn_edit(sid, 99, "nope")


def test_store_history_cap_limits_undo_reach(tmp_path):
    # After the 100-cap rolls off the first edit, undo can no longer reach the
    # pristine original — but every later edit stays reversible and the materialized
    # current state is always correct.
    segs = [_seg(0, 1, "start", "SPEAKER_00")]
    store, sid = _make_store(tmp_path, segs)
    for n in range(HISTORY_LIMIT + 1):                  # edits "edit 0" .. "edit 100"
        store.save_turn_edit(sid, 0, f"edit {n}")
    for _ in range(HISTORY_LIMIT):                      # exhaust the retained deltas
        store.undo_turn_edit(sid)
    edits = store.load_edits(sid)
    assert edits is not None and edits["history"] == []
    # "edit 0"'s delta was dropped, so this is as far back as undo can go.
    assert store.current_segments(sid, segs)[0]["text"] == "edit 0"
    # A further undo is a harmless no-op that reports the live (non-pristine) state.
    assert store.undo_turn_edit(sid)[0]["text"] == "edit 0"


def test_store_unicode_round_trips(tmp_path, diarized):
    store, sid = _make_store(tmp_path, diarized)
    store.save_turn_edit(sid, 0, "Привіт, як справи? 😀")
    assert store.current_segments(sid, diarized)[0]["text"] == "Привіт, як справи? 😀"


# --- end-to-end: a realistic editing session --------------------------------

def test_end_to_end_editing_session(tmp_path, diarized):
    """Walk the full stack (store + edits + render, no Flask) through the common
    operations a user performs: edit, re-render, rename a speaker, undo, delete a
    turn (with re-merge), undo the delete — and verify the original is never touched."""
    store, sid = _make_store(tmp_path, diarized)
    original = copy.deepcopy(diarized)

    def body():
        result = store.load_result(sid)
        segs = store.current_segments(sid, result["segments"])
        return render_transcript({"segments": segs}, store.get_speaker_names(sid))

    # 0) Pristine view: original text, three turns, no overlay yet.
    html = body()
    assert "Hello there." in html and "Fine thanks." in html
    assert store.load_edits(sid) is None

    # 1) Edit the opening (multi-segment) turn -> collapses to one timed span.
    store.save_turn_edit(sid, 0, "Hi, how is it going?")
    html = body()
    assert "Hi, how is it going?" in html
    assert "Hello there." not in html
    assert 'data-start="0.000" data-end="2.000">' in html   # segment-level timing
    assert store.edit_history_len(sid) == 1

    # 2) Edit the SPEAKER_01 turn. Indices shifted (4 segs -> 3): it's now turn 1.
    store.save_turn_edit(sid, 1, "All good here.")
    assert "All good here." in body()
    assert store.edit_history_len(sid) == 2

    # 3) Rename a speaker — the override still resolves on edited turns.
    store.set_speaker_name(sid, "SPEAKER_00", "Alice")
    assert "Alice" in body()

    # 4) Undo the last edit -> SPEAKER_01 text comes back; one delta left.
    store.undo_turn_edit(sid)
    html = body()
    assert "Fine thanks." in html
    assert "All good here." not in html
    assert store.edit_history_len(sid) == 1

    # 5) Delete a turn: empty-edit the trailing "Good." turn (now turn 2).
    store.save_turn_edit(sid, 2, "")
    html = body()
    assert "Good." not in html
    assert store.edit_history_len(sid) == 2

    # 6) Undo the delete -> "Good." restored.
    store.undo_turn_edit(sid)
    assert "Good." in body()

    # Throughout, the original transcript.json was never mutated.
    assert store.load_result(sid)["segments"] == original


def test_end_to_end_delete_merges_then_undo_splits(tmp_path, diarized):
    """Deleting the middle speaker fuses the two SPEAKER_00 turns into one bubble;
    undo splits them back apart."""
    store, sid = _make_store(tmp_path, diarized)

    def turn_count():
        segs = store.current_segments(sid, store.load_result(sid)["segments"])
        return len(group_turns(segs))

    assert turn_count() == 3
    store.save_turn_edit(sid, 1, "")        # delete SPEAKER_01 -> SPEAKER_00 runs merge
    assert turn_count() == 1
    store.undo_turn_edit(sid)
    assert turn_count() == 3
    assert store.load_edits(sid) is None    # fully reverted -> overlay dropped
