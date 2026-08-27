"""Test sentence splitting for languages written without spaces."""

import torch
from unittest.mock import MagicMock

from whisperx import alignment
from whisperx.alignment import align, LANGUAGES_WITHOUT_SPACES
from whisperx.utils import CJKSentenceSplitter, PUNKT_LANGUAGES

ZH = "我要给他一个教训。人家帮你是人情。你已经死了。"
JA = "私は彼に教訓を与えます。手伝うのは人情です。もう終わりです。"


class TestCJKSentenceSplitter:
    """The splitter itself, without loading an alignment model."""

    def setup_method(self):
        self.splitter = CJKSentenceSplitter()

    def spans(self, text):
        return list(self.splitter.span_tokenize(text))

    def test_chinese_splits_on_terminators(self):
        assert len(self.spans(ZH)) == 3

    def test_japanese_splits_on_terminators(self):
        assert len(self.spans(JA)) == 3

    def test_spans_slice_back_to_the_sentences(self):
        assert [ZH[a:b] for a, b in self.spans(ZH)] == [
            "我要给他一个教训。",
            "人家帮你是人情。",
            "你已经死了。",
        ]

    def test_terminator_stays_with_its_sentence(self):
        for _, end in self.spans(ZH):
            assert ZH[end - 1] == "。"

    def test_spans_are_contiguous_when_there_is_no_whitespace(self):
        prev_end = 0
        for start, end in self.spans(ZH):
            assert start == prev_end
            assert end > start
            prev_end = end
        assert prev_end == len(ZH)

    def test_whitespace_between_sentences_is_skipped(self):
        """Punkt leaves the separator out of its spans, so this does too."""
        assert self.spans("これは。 次の文。") == [(0, 4), (5, 9)]
        assert self.spans("これは。\n次の文。") == [(0, 4), (5, 9)]

    def test_fullwidth_period_does_not_split_fullwidth_numbers(self):
        """． is a terminator in the 「，．」 style but also a decimal point."""
        text = "値は３．１４です．次の文．"
        assert [text[a:b] for a, b in self.spans(text)] == ["値は３．１４です．", "次の文．"]

    def test_fullwidth_period_still_ends_a_sentence(self):
        text = "これは文です．次の文です．"
        assert len(self.spans(text)) == 2

    def test_terminator_inside_a_quotation_still_splits(self):
        """Deliberate: tracking bracket depth would let one unclosed quote
        swallow every later sentence, which is the bug being fixed."""
        text = "彼は「もう終わりです。まだです」と言った。"
        assert len(self.spans(text)) > 1

    def test_run_of_terminators_ends_one_sentence(self):
        text = "本当に！？信じられない。"
        assert [text[a:b] for a, b in self.spans(text)] == ["本当に！？", "信じられない。"]

    def test_closing_bracket_stays_with_its_sentence(self):
        text = "彼は「もう終わりです。」と言った。"
        assert [text[a:b] for a, b in self.spans(text)] == [
            "彼は「もう終わりです。」",
            "と言った。",
        ]

    def test_halfwidth_terminators(self):
        text = "そうですか?はい!"
        assert [text[a:b] for a, b in self.spans(text)] == ["そうですか?", "はい!"]

    def test_halfwidth_period_does_not_split_numbers(self):
        text = "値は3.14です。次の文。"
        assert [text[a:b] for a, b in self.spans(text)] == ["値は3.14です。", "次の文。"]

    def test_halfwidth_period_followed_by_space_still_splits(self):
        """The English punkt fallback used to split these, so not splitting
        them would be a regression for CJK text punctuated with ASCII."""
        text = "你好世界. 再见朋友."
        assert [text[a:b] for a, b in self.spans(text)] == ["你好世界.", "再见朋友."]

    def test_halfwidth_period_without_a_following_space_does_not_split(self):
        """Punkt does not split this either, so match it."""
        text = "你好世界.再见朋友."
        assert len(self.spans(text)) == 1

    def test_halfwidth_bang_inside_a_latin_word_does_not_split(self):
        text = "Yahoo!ニュースを見た。"
        assert len(self.spans(text)) == 1

    def test_question_mark_inside_a_url_does_not_split(self):
        text = "https://example.com/a?b=1を開く。"
        assert len(self.spans(text)) == 1

    def test_halfwidth_bang_after_kana_still_splits(self):
        text = "はい!そうです。"
        assert [text[a:b] for a, b in self.spans(text)] == ["はい!", "そうです。"]

    def test_ascii_quote_is_not_treated_as_a_closer(self):
        """The same character opens and closes, so absorbing it would take the
        next sentence's opening quote."""
        text = '終わりです。"次は何ですか。'
        assert [text[a:b] for a, b in self.spans(text)] == [
            "終わりです。", '"次は何ですか。']

    def test_text_without_a_terminator_is_one_sentence(self):
        text = "これはテストです"
        assert self.spans(text) == [(0, len(text))]

    def test_empty_text_has_no_sentences(self):
        assert self.spans("") == []

    def test_trailing_whitespace_is_excluded(self):
        """Punkt leaves trailing whitespace out of its spans, so this does too."""
        assert self.spans("これはテストです。  ") == [(0, 9)]
        assert self.spans("これはテストです  ") == [(0, 8)]


class TestSplitterSelection:
    """Which splitter each language gets. These monkeypatch the loader so they
    assert which tokenizer is requested without downloading NLTK data."""

    def _requested(self, monkeypatch, lang):
        """The punkt model align() asks for, or the CJK splitter it picks."""
        asked = []

        def fake_load(path):
            asked.append(path)
            return object()

        monkeypatch.setattr(alignment, "nltk_load", fake_load)
        if lang in LANGUAGES_WITHOUT_SPACES:
            return CJKSentenceSplitter, asked
        return type(alignment._load_punkt(lang)), asked

    def test_cjk_languages_do_not_load_a_punkt_model(self, monkeypatch):
        for lang in ("zh", "ja"):
            _, asked = self._requested(monkeypatch, lang)
            assert asked == [], f"{lang} should not need punkt, asked for {asked}"

    def test_each_mapped_language_asks_for_its_own_model(self, monkeypatch):
        for lang, expected in PUNKT_LANGUAGES.items():
            _, asked = self._requested(monkeypatch, lang)
            assert asked == [f"tokenizers/punkt_tab/{expected}.pickle"]

    def test_unmapped_language_asks_for_english(self, monkeypatch):
        for lang in ("xx", "ko", "ar"):
            _, asked = self._requested(monkeypatch, lang)
            assert asked == ["tokenizers/punkt_tab/english.pickle"]

    def test_align_uses_the_cjk_splitter_for_cjk_only(self):
        """The list align() branches on is the one the splitter is built for."""
        assert set(LANGUAGES_WITHOUT_SPACES) == {"ja", "zh"}
        assert not set(LANGUAGES_WITHOUT_SPACES) & set(PUNKT_LANGUAGES)


class TestAlignSplitsCJKSegments:
    """align() end-to-end, so the fix is checked where it actually runs."""

    # Chinese alignment models are phoneme-based, so the dictionary here stands
    # in for one: no CJK character is in it and every one takes the wildcard
    # path. That is enough to exercise sentence splitting, which is what
    # decides how many segments come back.
    DICTIONARY = {"<pad>": 0, "a": 1, "|": 2}

    def _run_align(self, text, language, duration=6.0, num_frames=120,
                   return_char_alignments=False):
        torch.manual_seed(0)
        emission = torch.full((num_frames, 3), -5.0)
        emission[:, 0] = -1.0
        model = MagicMock()
        model.return_value = (emission.unsqueeze(0), None)

        return align(
            transcript=[{"text": text, "start": 0.0, "end": duration}],
            model=model,
            align_model_metadata={
                "language": language,
                "dictionary": self.DICTIONARY,
                "type": "torchaudio",
            },
            audio=torch.randn(int(duration * 16000)),
            device="cpu",
            return_char_alignments=return_char_alignments,
        )

    def test_chinese_segment_splits_into_sentences(self):
        """Before the fix this returned one segment holding all three sentences."""
        result = self._run_align(ZH, "zh")
        assert len(result["segments"]) == 3

    def test_japanese_segment_splits_into_sentences(self):
        result = self._run_align(JA, "ja")
        assert len(result["segments"]) == 3

    def test_each_chinese_sentence_keeps_its_own_text(self):
        result = self._run_align(ZH, "zh")
        assert [s["text"] for s in result["segments"]] == [
            "我要给他一个教训。",
            "人家帮你是人情。",
            "你已经死了。",
        ]

    def test_words_do_not_leak_across_the_sentence_boundary(self):
        """CJK spans are adjacent, so an inclusive end would take the next
        sentence's first character. The words of each segment should be exactly
        its own text."""
        for text in (ZH, " " + ZH):
            result = self._run_align(text, "zh")
            for seg in result["segments"]:
                assert "".join(w["word"] for w in seg["words"]) == seg["text"].strip()

    def test_english_sentences_still_carry_their_own_words(self):
        """The same boundary check for the punkt path, which must be unchanged."""
        dictionary = {"<pad>": 0, "|": 1}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", 2):
            dictionary[c] = i
        torch.manual_seed(0)
        emission = torch.full((160, 28), -5.0)
        emission[:, 0] = -1.0
        model = MagicMock()
        model.return_value = (emission.unsqueeze(0), None)
        result = align(
            transcript=[{"text": "I teach him. Helping is kind. You are done.",
                         "start": 0.0, "end": 8.0}],
            model=model,
            align_model_metadata={"language": "en", "dictionary": dictionary,
                                  "type": "torchaudio"},
            audio=torch.randn(128000),
            device="cpu",
        )
        assert [s["text"] for s in result["segments"]] == [
            "I teach him.", "Helping is kind.", "You are done."]
        for seg in result["segments"]:
            assert " ".join(w["word"] for w in seg["words"]) == seg["text"]

    def test_char_alignments_cover_the_whole_cjk_segment(self):
        result = self._run_align(ZH, "zh", return_char_alignments=True)
        rebuilt = "".join(c["char"] for s in result["segments"] for c in s["chars"])
        assert rebuilt == ZH
        for seg in result["segments"]:
            assert "".join(c["char"] for c in seg["chars"]) == seg["text"]

    def test_sentence_timestamps_are_ordered_and_in_range(self):
        result = self._run_align(ZH, "zh", duration=6.0)
        prev_end = 0.0
        for seg in result["segments"]:
            assert seg["start"] >= prev_end
            assert seg["end"] >= seg["start"]
            assert seg["end"] <= 6.0
            prev_end = seg["end"]


class TestPunktPathBoundaries:
    """The exclusive span end also touches the punkt path, so pin it here."""

    DICTIONARY = {"<pad>": 0, "|": 1}
    for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz", 2):
        DICTIONARY[_c] = _i
    del _i, _c

    def _run_align(self, text, return_char_alignments=False):
        torch.manual_seed(0)
        num_frames = 220
        emission = torch.full((num_frames, 28), -5.0)
        emission[:, 0] = -1.0
        for j, ch in enumerate(text.lower()):
            if ch in self.DICTIONARY:
                centre = int((j + 1) * num_frames / (len(text) + 1))
                for t in range(max(0, centre - 2), min(num_frames, centre + 2)):
                    emission[t, self.DICTIONARY[ch]] = 2.0
                    emission[t, 0] = -3.0
        model = MagicMock()
        model.return_value = (emission.unsqueeze(0), None)
        return align(
            transcript=[{"text": text, "start": 0.0, "end": 8.0}],
            model=model,
            align_model_metadata={"language": "en", "dictionary": self.DICTIONARY,
                                  "type": "torchaudio"},
            audio=torch.randn(128000),
            device="cpu",
            return_char_alignments=return_char_alignments,
        )

    def test_char_alignments_cover_every_character(self):
        """Punkt leaves the separating whitespace out of its spans. Those
        characters still have to appear in the char output."""
        text = "One apple here. Two pears there. Three plums now."
        result = self._run_align(text, return_char_alignments=True)
        rebuilt = "".join(c["char"] for s in result["segments"] for c in s["chars"])
        assert rebuilt == text

    def test_char_alignments_cover_multiple_separating_spaces(self):
        text = "One apple here.  Two pears there.  Three plums now."
        result = self._run_align(text, return_char_alignments=True)
        rebuilt = "".join(c["char"] for s in result["segments"] for c in s["chars"])
        assert rebuilt == text

    def test_newline_separated_sentences_do_not_touch(self):
        """A separator that is not a plain space used to be counted as part of
        the sentence before it, so one sentence ended exactly where the next
        began."""
        result = self._run_align("One apple here.\nTwo pears there.\nThree plums now.")
        segments = result["segments"]
        assert len(segments) == 3
        for earlier, later in zip(segments, segments[1:]):
            assert earlier["end"] < later["start"]

    def test_adjacent_punkt_spans_do_not_leak_words(self):
        """Punkt returns adjacent spans when a boundary has no whitespace after
        it, e.g. 'Hello!!!Goodbye.' -> [(0, 7), (7, 16)]. Those hit the same
        boundary problem as CJK: the first sentence took the next one's opening
        character."""
        result = self._run_align("Hello!!!Goodbye.")
        assert len(result["segments"]) == 2
        for seg in result["segments"]:
            assert " ".join(w["word"] for w in seg["words"]) == seg["text"]

    def test_single_space_sentences_keep_their_own_words(self):
        result = self._run_align("One apple here. Two pears there.")
        assert [s["text"] for s in result["segments"]] == [
            "One apple here.", "Two pears there."]
        for seg in result["segments"]:
            assert " ".join(w["word"] for w in seg["words"]) == seg["text"]
