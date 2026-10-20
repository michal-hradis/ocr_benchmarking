"""Strict text metrics: ordered CER/WER, bag-of-words P/R/F1 and ROUGE-2.

The rules are documented in docs/EVALUATION.md; the decisions are in ADR 0008. Text is
Unicode-normalized (NFC by default), every whitespace run, including line and region breaks,
becomes one space, and the ends are stripped. Characters are code points; words are the
space-separated tokens of that text, and every metric uses the same tokens. Capitalization,
diacritics and punctuation are kept. Any change to these rules increments METRICS_VERSION.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, fields

from rapidfuzz.distance import Levenshtein

METRICS_VERSION = 1
UNICODE_FORMS = ("NFC", "NFKC")


def normalize(text: str, form: str = "NFC") -> str:
    """Unicode normalization, then each whitespace run becomes one space; ends are stripped."""
    return " ".join(unicodedata.normalize(form, text).split())


def words(normalized: str) -> list[str]:
    return normalized.split(" ") if normalized else []


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class EditCounts:
    """Levenshtein operations turning the reference into the hypothesis (unit costs)."""

    reference: int  # reference length: the denominator of CER/WER
    hypothesis: int
    substitutions: int
    deletions: int  # reference units missing from the hypothesis
    insertions: int  # extra hypothesis units, including hallucinations and repetitions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """Errors per reference unit; None for an empty reference. It can exceed 1."""
        return _ratio(self.errors, self.reference)

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def edit_counts(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> EditCounts:
    """Counts of one minimal alignment of two strings (code points) or two token lists.

    The total is the edit distance. Equally minimal alignments can trade one substitution for a
    deletion plus an insertion; the pinned rapidfuzz version chooses deterministically.
    """
    ops = Counter(tag for tag, _, _ in Levenshtein.editops(reference, hypothesis))
    return EditCounts(len(reference), len(hypothesis), ops["replace"], ops["delete"], ops["insert"])


@dataclass(frozen=True)
class Overlap:
    """Multiset overlap of reference and hypothesis items (words or word bigrams)."""

    reference: int
    hypothesis: int
    matched: int

    @property
    def precision(self) -> float | None:
        """Matched share of the hypothesis; penalizes extra items. None without hypothesis."""
        return _ratio(self.matched, self.hypothesis)

    @property
    def recall(self) -> float | None:
        """Matched share of the reference; penalizes missing items. None without reference."""
        return _ratio(self.matched, self.reference)

    @property
    def f1(self) -> float | None:
        """2·matched / (reference + hypothesis): 0 when only one side is empty, None for both."""
        return _ratio(2 * self.matched, self.reference + self.hypothesis)

    def __add__(self, other: Overlap) -> Overlap:
        return Overlap(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def overlap(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> Overlap:
    matched = sum((Counter(reference) & Counter(hypothesis)).values())
    return Overlap(len(reference), len(hypothesis), matched)


def bigrams(tokens: Sequence[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:], strict=False))


@dataclass(frozen=True)
class TextScores:
    chars: EditCounts  # CER
    words: EditCounts  # WER
    bag_of_words: Overlap
    rouge2: Overlap

    def __add__(self, other: TextScores) -> TextScores:
        return TextScores(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


def score_text(reference: str, hypothesis: str, form: str = "NFC") -> TextScores:
    """All metrics of one sample from its raw reference and predicted page text."""
    ref, hyp = normalize(reference, form), normalize(hypothesis, form)
    ref_words, hyp_words = words(ref), words(hyp)
    return TextScores(
        chars=edit_counts(ref, hyp),
        words=edit_counts(ref_words, hyp_words),
        bag_of_words=overlap(ref_words, hyp_words),
        rouge2=overlap(bigrams(ref_words), bigrams(hyp_words)),
    )
