"""Strict text metrics on hand-computed examples, Unicode normalization and a 2,000+ line page."""

import random

import pytest

from ocrbench.metrics import (
    EditCounts,
    Overlap,
    bigrams,
    edit_counts,
    normalize,
    overlap,
    score_text,
    words,
)


def levenshtein(a, b):
    """Plain dynamic-programming edit distance, the reference for the rapidfuzz alignment."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def counts(scores):
    c = scores
    return (c.reference, c.hypothesis, c.substitutions, c.deletions, c.insertions)


def test_substitutions_deletions_insertions_and_denominators():
    kitten = edit_counts("kitten", "sitting")
    assert counts(kitten) == (6, 7, 2, 0, 1)
    assert kitten.errors == 3
    assert kitten.rate == pytest.approx(3 / 6)
    s = score_text("the cat sat on the mat", "the cat sat on mat")
    assert counts(s.words) == (6, 5, 0, 1, 0)
    assert s.words.rate == pytest.approx(1 / 6)
    assert counts(s.chars) == (22, 18, 0, 4, 0)  # "the " is missing


def test_strict_scores_keep_case_diacritics_and_punctuation():
    s = score_text("Café, X", "cafe X")
    assert counts(s.chars) == (7, 6, 2, 1, 0)  # C->c, é->e, comma deleted
    assert counts(s.words) == (2, 2, 1, 0, 0)  # "Café," vs "cafe"
    assert s.bag_of_words == Overlap(2, 2, 1)


def test_hallucinated_text_counts_as_insertions_and_lowers_precision():
    s = score_text("a b", "a b c d")
    assert counts(s.chars) == (3, 7, 0, 0, 4)
    assert s.chars.rate == pytest.approx(4 / 3)  # rates are not capped at 1
    assert counts(s.words) == (2, 4, 0, 0, 2)
    bow = s.bag_of_words
    assert (bow.precision, bow.recall, bow.f1) == (0.5, 1.0, pytest.approx(2 / 3))


def test_repetition_is_counted_by_every_metric():
    s = score_text("one two three", "one two three one two three")
    assert counts(s.chars) == (13, 27, 0, 0, 14)
    assert counts(s.words) == (3, 6, 0, 0, 3)
    assert s.bag_of_words == Overlap(3, 6, 3)
    assert s.rouge2 == Overlap(2, 5, 2)  # repeated bigrams match only once each
    assert s.rouge2.f1 == pytest.approx(4 / 7)


def test_missing_text_is_all_deletions():
    s = score_text("lost words here", "")
    assert counts(s.chars) == (15, 0, 0, 15, 0)
    assert s.chars.rate == 1.0
    assert counts(s.words) == (3, 0, 0, 3, 0)
    assert s.bag_of_words == Overlap(3, 0, 0)
    bow = s.bag_of_words
    assert (bow.precision, bow.recall, bow.f1) == (None, 0.0, 0.0)


def test_empty_reference_and_empty_pages():
    extra = score_text("", "ghost text")
    assert counts(extra.chars) == (0, 10, 0, 0, 10)
    assert extra.chars.rate is None  # undefined, but the insertions stay in the counts
    assert extra.words.rate is None
    bow = extra.bag_of_words
    assert (bow.precision, bow.recall, bow.f1) == (0.0, None, 0.0)
    empty = score_text(" \n", "")
    assert empty.chars == EditCounts(0, 0, 0, 0, 0)
    assert empty.chars.rate is None
    assert (empty.bag_of_words.f1, empty.rouge2.f1) == (None, None)


def test_line_and_region_breaks_are_one_space():
    s = score_text("abc def\nghi", "abc\ndef\n\n  ghi \t")
    assert s.chars.errors == 0
    assert s.words.errors == 0
    assert normalize(" a \n\n b ") == "a b"  # no-break space is whitespace too


def test_split_and_merged_words():
    split = score_text("abcdef", "abc def")
    assert counts(split.chars) == (6, 7, 0, 0, 1)
    assert counts(split.words) == (1, 2, 1, 0, 1)
    merged = score_text("abc def", "abcdef")
    assert counts(merged.chars) == (7, 6, 0, 1, 0)
    assert counts(merged.words) == (2, 1, 1, 1, 0)


def test_unicode_normalization():
    composed, decomposed = "été", "été"
    assert score_text(composed, decomposed).chars.errors == 0  # NFC by default
    assert normalize(decomposed) == composed
    assert score_text("ſ ﬁ", "s fi").chars.errors == 3  # strict: long s and ligature differ
    assert score_text("ſ ﬁ", "s fi", "NFKC").chars.errors == 0
    combining = score_text("q̇", "q")  # no precomposed form: two code points
    assert counts(combining.chars) == (2, 1, 0, 1, 0)


def test_bag_of_words_ignores_order_but_rouge2_does_not():
    s = score_text("a b c", "c b a")
    assert s.bag_of_words.f1 == 1.0
    assert s.rouge2 == Overlap(2, 2, 0)
    assert s.words.errors == 2
    assert bigrams(words("x")) == []
    assert score_text("x", "x").rouge2.f1 is None  # one word has no bigram


def test_overlap_is_a_multiset_intersection():
    assert overlap(["a", "a", "b"], ["a", "b", "b", "c"]) == Overlap(3, 4, 2)


def test_counts_add_up():
    total = edit_counts("ab", "b") + edit_counts("", "xy")
    assert total == EditCounts(2, 3, 0, 1, 2)
    assert total.rate == 1.5
    assert (Overlap(2, 0, 0) + Overlap(0, 2, 0)).f1 == 0.0


def test_alignment_matches_the_reference_distance():
    rng = random.Random(8)
    for _ in range(300):
        a = "".join(rng.choices("abcž ", k=rng.randint(0, 12)))
        b = "".join(rng.choices("abcž ", k=rng.randint(0, 12)))
        c = edit_counts(a, b)
        assert c.errors == levenshtein(a, b)
        assert c.hypothesis == c.reference - c.deletions + c.insertions
        tokens = edit_counts(a.split(), b.split())
        assert tokens.errors == levenshtein(a.split(), b.split())


def test_newspaper_page_with_2500_lines():
    lines = [f"line {i:04d} text ž{i % 7}" for i in range(2500)]
    dropped = range(100, 110)
    predicted = [line for i, line in enumerate(lines) if i not in dropped] + lines[:3]
    s = score_text("\n".join(lines), "\n".join(predicted))
    assert s.chars.reference == sum(len(line) for line in lines) + len(lines) - 1
    deleted = sum(len(lines[i]) + 1 for i in dropped)
    inserted = sum(len(line) + 1 for line in lines[:3])
    assert counts(s.chars)[2:] == (0, deleted, inserted)
    assert counts(s.words)[2:] == (0, 40, 12)
    assert s.bag_of_words == Overlap(10000, 9972, 9969)
