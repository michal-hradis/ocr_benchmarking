"""Flexible CER/WER: hand-computed accounting, the exhaustive reference against an independent
enumeration, the scalable search against the exhaustive optimum, and a 2,000+ line page."""

import itertools
import random

import pytest

from ocrbench import alignment
from ocrbench.alignment import (
    Group,
    _Side,
    exhaustive_groups,
    prediction_lines,
    score_flexible,
    solution_counts,
)
from ocrbench.metrics import score_text
from ocrbench.prediction import Line, Prediction, Region


def counts(alignment_):
    c = alignment_.counts
    return (c.reference, c.hypothesis, c.substitutions, c.deletions, c.insertions)


def chars(ref, hyp):
    return score_flexible(ref, hyp).chars


def test_reordered_columns_cost_nothing_but_count_as_an_order_break():
    a = chars(["ab", "cd"], ["cd", "ab"])
    assert counts(a) == (5, 5, 0, 0, 0)  # "ab cd": the separator between the groups is matched
    assert (len(a.groups), a.order_breaks, a.exact) == (2, 1, True)
    assert score_text("ab cd", "cd ab").chars.errors == 4  # ordered


def test_missing_and_extra_lines_cost_their_length_and_one_separator():
    missing = chars(["ab", "cd", "ef"], ["cd", "ab"])
    assert counts(missing) == (8, 5, 0, 3, 0)  # "ef" and its separator are deleted
    assert missing.unmatched_reference == 1
    extra = chars(["ab", "cd"], ["cd", "xyz", "ab"])
    assert counts(extra) == (5, 9, 0, 0, 4)  # "xyz" and its separator are inserted


def test_empty_sides_match_the_ordered_score():
    assert counts(chars(["ab", "cd"], [])) == (5, 0, 0, 5, 0)  # a failure: all deleted
    assert counts(chars([], ["ab", "c"])) == (0, 4, 0, 0, 4)
    assert counts(chars([], [])) == (0, 0, 0, 0, 0)
    assert chars([], []).counts.rate is None
    assert counts(score_flexible(["ab cd"], []).words) == (2, 0, 0, 2, 0)


def test_segmentation_differences_are_free():
    merged = chars(["erste Zeile", "zweite Zeile"], ["erste Zeile zweite Zeile"])  # N:1
    split = chars(["erste Zeile zweite Zeile"], ["erste Zeile", "zweite Zeile"])  # 1:N
    assert merged.counts.errors == split.counts.errors == 0
    assert len(merged.groups) == len(split.groups) == 1


def test_order_inside_a_group_still_counts():
    # One prediction unit cannot be cut: lines reordered inside it cost edits.
    a = chars(["ab", "cd"], ["cd ab"])
    assert a.counts.errors == score_text("ab cd", "cd ab").chars.errors


def test_repeated_lines_are_not_credited_twice():
    assert counts(chars(["Anzeige", "Anzeige"], ["Anzeige"])) == (15, 7, 0, 8, 0)
    assert counts(chars(["Anzeige"], ["Anzeige", "Anzeige"])) == (7, 15, 0, 0, 8)
    with pytest.raises(ValueError, match="more than one group"):
        solution_counts(
            _Side(["a"], True), _Side(["a", "a"], True), [Group(0, 1, 0, 1), Group(0, 1, 1, 2)]
        )


def test_substitutions_inside_groups_and_word_level_alignment():
    s = score_flexible(["Die Zeitung", "erste Zeile"], ["erste Zeils", "Die Zeitunq"])
    assert counts(s.chars) == (23, 23, 2, 0, 0)
    assert counts(s.words) == (4, 4, 2, 0, 0)
    assert s.words.order_breaks == 1


def test_groups_are_reported_as_ranges_of_input_lines():
    s = score_flexible(["alpha beta", "  ", "gamma delta"], ["gamma delta", "", "alpha beta"])
    # Empty lines are no units, so unit indices differ from line indices.
    assert set(s.chars.groups) == {Group(1, 2, 0, 1), Group(0, 1, 1, 2)}
    assert s.line_ranges(s.chars) == [(2, 3, 0, 1), (0, 1, 2, 3)]  # in prediction order
    assert s.line_ranges(s.words) == [(2, 3, 0, 1), (0, 1, 2, 3)]
    empty = score_flexible(["alpha"], [])
    assert empty.line_ranges(empty.chars) == []


def test_unicode_form_is_applied_to_units():
    s = score_flexible(["ﬁsch"], ["fisch"], form="NFKC")
    assert s.chars.counts.errors == 0
    assert score_flexible(["ﬁsch"], ["fisch"]).chars.counts.errors == 2


# The exhaustive reference -------------------------------------------------------------------


def compositions(n):
    """Every split of range(n) into consecutive runs."""
    for cuts in itertools.product((False, True), repeat=max(n - 1, 0)):
        runs, start = [], 0
        for i, cut in enumerate(cuts, 1):
            if cut:
                runs.append((start, i))
                start = i
        yield [*runs, (start, n)] if n else []


def enumerate_optimum(ref, hyp):
    """Independent brute force: every pair of run splits and every partial matching of runs."""
    best = None
    for ref_runs in compositions(len(ref)):
        for hyp_runs in compositions(len(hyp)):
            for k in range(min(len(ref_runs), len(hyp_runs)) + 1):
                for chosen in itertools.combinations(ref_runs, k):
                    for partners in itertools.permutations(hyp_runs, k):
                        groups = [Group(*r, *h) for r, h in zip(chosen, partners, strict=True)]
                        errors = solution_counts(ref, hyp, groups).errors
                        best = errors if best is None else min(best, errors)
    return best


def random_units(rng, n, alphabet="abc"):
    return [
        "".join(rng.choice(alphabet + " ") for _ in range(rng.randint(1, 4))).strip() or "a"
        for _ in range(n)
    ]


@pytest.mark.parametrize("seed", range(40))
def test_exhaustive_search_matches_an_independent_enumeration(seed):
    rng = random.Random(seed)
    for separator in (True, False):
        ref_units = random_units(rng, rng.randint(0, 4))
        hyp_units = random_units(rng, rng.randint(0, 4))
        if not separator:
            ref_units = [tuple(u.split()) for u in ref_units]
            hyp_units = [tuple(u.split()) for u in hyp_units]
        ref, hyp = _Side(ref_units, separator), _Side(hyp_units, separator)
        found = solution_counts(ref, hyp, exhaustive_groups(ref, hyp)).errors
        assert found == enumerate_optimum(ref, hyp)
        assert alignment.lower_bound(ref, hyp) <= found


@pytest.mark.parametrize("seed", range(30))
def test_scalable_search_is_valid_and_never_worse_than_ordered(seed, monkeypatch):
    rng = random.Random(seed)
    words_ = ["Zeitung", "erste", "Zeile", "Stadt", "Rathaus", "Wetter", "heute", "Bericht"]
    ref = [" ".join(rng.choice(words_) for _ in range(rng.randint(1, 4))) for _ in range(6)]
    hyp = ref[3:] + ref[:3]
    if seed % 3 == 0:
        hyp[1] = hyp[1].replace("e", "c")
    if seed % 3 == 1:
        hyp.insert(2, "Anzeigen")
    if seed % 3 == 2:
        del hyp[4]
    exact = score_flexible(ref, hyp)
    monkeypatch.setattr(alignment, "EXHAUSTIVE_UNITS", 0)
    scalable = score_flexible(ref, hyp)
    ordered = score_text(" ".join(ref), " ".join(hyp))
    for key in ("chars", "words"):
        found, optimum = getattr(scalable, key), getattr(exact, key)
        assert optimum.method == "exhaustive"
        assert found.lower_bound <= optimum.counts.errors <= found.counts.errors
        assert found.counts.errors <= getattr(ordered, key).errors
        assert found.exact == (found.counts.errors == found.lower_bound)


def test_inexact_results_are_flagged_with_a_lower_bound(monkeypatch):
    monkeypatch.setattr(alignment, "EXHAUSTIVE_UNITS", 0)
    shuffled = chars(["aaa bbb", "ccc ddd", "eee fff"], ["eee fff", "aaa bbb", "ccc ddd"])
    assert (shuffled.counts.errors, shuffled.exact, shuffled.method) == (0, True, "reordered")
    # A missing line next to an unrelated extra line: the bag bound cannot prove optimality.
    noisy = chars(["aaa bbb", "ccc ddd", "eee fff"], ["eee fff", "xyz", "aaa bbb"])
    assert noisy.lower_bound < noisy.counts.errors
    assert not noisy.exact


# Prediction units and text-only output --------------------------------------------------------


def test_prediction_units_are_the_lines_of_the_structure_text():
    regions = (
        Region("r1", lines=(Line("erste Zeile"), Line("zweite Zeile"))),
        Region("r2", text="ein Absatz\nmit Umbruch"),
        Region("r3", type="figure"),
    )
    with_regions = Prediction("s", "erste Zeile zweite Zeile ein Absatz mit Umbruch", regions)
    assert prediction_lines(with_regions) == [
        "erste Zeile",
        "zweite Zeile",
        "",
        "ein Absatz",
        "mit Umbruch",
    ]
    assert prediction_lines(Prediction("s", "eine Seite\nohne Regionen")) == [
        "eine Seite",
        "ohne Regionen",
    ]


def make_page(rng, columns, lines_per_column):
    letters = "abcdefghijklmnopqrstuvwxyzäöü"
    vocab = ["".join(rng.choice(letters) for _ in range(rng.randint(1, 9))) for _ in range(2000)]

    def line():
        return " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 8)))

    cols = [[line() for _ in range(lines_per_column)] for _ in range(columns)]
    for col in cols:  # repeated identical lines in every column
        for i in range(0, lines_per_column, 97):
            col[i] = "Anzeigen und Bekanntmachungen"
    return cols


def noisy(rng, text):
    return "".join(rng.choice("xyz") if rng.random() < 0.02 else c for c in text)


def test_whole_page_paragraphs_without_lines_are_evaluated_meaningfully():
    rng = random.Random(3)
    cols = make_page(rng, 3, 60)
    ref = [line for col in cols for line in col]
    # A text-only model: one paragraph per 10 lines, columns in a different order.
    hyp = [" ".join(cols[c][i : i + 10]) for c in (2, 0, 1) for i in range(0, 60, 10)]
    s = score_flexible(ref, hyp)
    assert s.chars.counts.errors == 0
    assert len(s.chars.groups) == 2  # columns 0 and 1 stay adjacent: one group
    assert score_text(" ".join(ref), "\n\n".join(hyp)).chars.rate > 0.5
    # A page without any line break is a single unit: it is scored in order.
    single = score_flexible(ref, [" ".join(hyp)])
    assert single.chars.counts == score_text(" ".join(ref), " ".join(hyp)).chars


def test_large_page_without_dense_comparison(monkeypatch):
    rng = random.Random(7)
    cols = make_page(rng, 4, 550)
    ref = [line for col in cols for line in col]
    predicted = [[noisy(rng, line) for line in col] for col in cols]
    predicted[1] = [line for i, line in enumerate(predicted[1]) if i % 50 != 3]  # missing lines
    predicted[2].insert(300, "hallucinated extra text here")
    hyp = [line for c in (2, 0, 3, 1) for line in predicted[c]]
    calls = {"distance": 0, "opcodes": 0, "editops": 0}
    real = alignment.Levenshtein

    class Counting:
        @staticmethod
        def distance(a, b):
            calls["distance"] += 1
            return real.distance(a, b)

        @staticmethod
        def opcodes(a, b):
            calls["opcodes"] += 1
            return real.opcodes(a, b)

    monkeypatch.setattr(alignment, "Levenshtein", Counting)
    s = score_flexible(ref, hyp)
    assert len(ref) == 2200
    assert sum(calls.values()) < 1000  # versus 2200 × 2189 pairs for a dense matrix
    ordered = score_text(" ".join(ref), "\n".join(hyp))
    for key in ("chars", "words"):
        flexible = getattr(s, key)
        assert flexible.method == "reordered"
        assert flexible.lower_bound <= flexible.counts.errors <= getattr(ordered, key).errors
        assert len(flexible.groups) <= 8
    assert s.chars.counts.rate < 0.03  # about 2 % substitutions plus the missing lines
    assert ordered.chars.rate > 0.5
    assert s.anchored_units >= 0.99 * len(hyp)  # a few short noisy lines have no anchor
