"""The pandas-free word assembly must give exactly the pandas result whenever it applies."""
import random
from types import SimpleNamespace

import pytest

from whisperx.alignment import _assemble_segment, _assemble_segment_fast, _assemble_segment_pandas

ALPHABET = list("abcdefghij") * 4 + [" "] * 8 + list(".,!?-") + list("0123")


def sentence_spans(text, rng):
    """Spans like NLTK's span_tokenize (end exclusive), sometimes touching or overlapping."""
    spans, pos = [], 0
    while pos < len(text):
        end = min(len(text), pos + rng.randint(1, 25))
        spans.append((pos, end))
        pos = end + rng.choice([0, 0, 1, 1, 2])
    return spans or [(0, len(text))]


def make_case(rng, drop_chars):
    text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(3, 80)))
    text = text.strip() or "ab"
    clean = [i for i, c in enumerate(text) if c != " " and not (drop_chars and rng.random() < 0.15)]
    if not clean:
        clean = [0]
    t = 0.0
    chars = []
    for _ in clean:
        start = t
        t += rng.randint(1, 6)
        chars.append(SimpleNamespace(start=int(start), end=int(t), score=rng.random()))
    return dict(
        text=text, char_segments=chars, clean_cdx_index={cdx: i for i, cdx in enumerate(clean)},
        t1=round(rng.uniform(0, 100), 3), ratio=rng.uniform(0.01, 0.05), sentence_spans=sentence_spans(text, rng),
        model_lang=rng.choice(["en", "en", "ru", "ja"]), avg_logprob=rng.choice([None, -0.25]),
        return_char_alignments=rng.choice([False, True]),
    )


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("drop_chars", [False, True])
def test_fast_assembly_equals_pandas(drop_chars):
    rng = random.Random(1234)
    used = 0
    for _ in range(150):
        case = make_case(rng, drop_chars)
        expected = _assemble_segment_pandas(**case, interpolate_method="nearest")
        fast = _assemble_segment_fast(**{k: v for k, v in case.items()})
        if fast is not None:
            used += 1
            assert same(fast, expected), case
        assert same(_assemble_segment(**case, interpolate_method="nearest"), expected)
    assert used > 25  # the fast path is the common one when every character is aligned


def test_float_types_of_records():
    case = make_case(random.Random(5), False)
    record = _assemble_segment(**case, interpolate_method="nearest")[0]
    assert type(record["start"]) is float and type(record["end"]) is float
