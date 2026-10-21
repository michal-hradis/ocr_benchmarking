"""Flexible (order-independent) CER/WER: GT lines matched with prediction lines in groups.

The rules are documented in docs/EVALUATION.md; the decisions are in ADR 0009. Reference units are
the PAGE XML reference lines; prediction units are the lines of the prediction's structure text.
A *group* pairs a run of consecutive reference units with a run of consecutive prediction units
and costs the Levenshtein distance of their joined texts. Groups can appear in any order, so
column order and segmentation do not count, while order inside a group does. A unit belongs to one
group at most: content is never credited twice. An unmatched unit costs its length plus one
separator (characters only) as deletions or insertions; separators between groups are matched.

The best grouping is found exhaustively for small samples. Larger ones use character q-gram
anchors, which avoid a dense comparison of every reference line with every prediction line:
consistently anchored prediction units form blocks, the blocks are put in reference order and
aligned with the reference, and the reference is cut into groups at line boundaries. The ordered
alignment (one group) is always a candidate, so the result is never worse than ordered CER/WER.
Every result carries a lower bound; ``exact`` is true when the result is proven optimal.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import chain
from statistics import median

from rapidfuzz.distance import Levenshtein

from ocrbench.metrics import EditCounts, edit_counts, normalize, words
from ocrbench.prediction import Prediction, structure_text

Q = 3  # character q-gram length of the retrieval index
MAX_POSTINGS = 64  # q-grams in more reference lines than this are too common to anchor
MIN_SCORE = 0.5  # share of a probe's or a line's q-grams that must be shared to anchor
MAX_JUMP = 2  # reference lines a block may skip between consecutive prediction units
EXHAUSTIVE_UNITS = 8  # samples with at most this many units on both sides are solved exhaustively
CUT_SHIFTS = (-2, -1, 1, 2)  # line shifts tried for every cut between groups


def reference_units(lines: Sequence[str], form: str = "NFC") -> list[str]:
    """Normalized, non-empty reference lines in reading order."""
    return [unit for unit in (normalize(line, form) for line in lines) if unit]


def _unit_lines(lines: Sequence[str], form: str) -> tuple[int, ...]:
    """The index of the input line behind each unit of ``reference_units(lines, form)``."""
    return tuple(n for n, line in enumerate(lines) if normalize(line, form))


def prediction_lines(prediction: Prediction) -> list[str]:
    """The lines of the structure text: prediction lines, the lines of a region's own text, or
    the lines of the page text when no region carries text. Paragraphs are never split further."""
    text = structure_text(prediction.regions)
    return (prediction.text if text is None else text).split("\n")


@dataclass(frozen=True)
class Group:
    """Reference units [ref_start, ref_stop) matched with prediction units [hyp_start, hyp_stop)."""

    ref_start: int
    ref_stop: int
    hyp_start: int
    hyp_stop: int


@dataclass(frozen=True)
class Alignment:
    counts: EditCounts
    lower_bound: int  # no grouping has fewer errors
    exact: bool  # counts.errors is proven minimal
    method: str  # "exhaustive", "ordered" (one group) or "reordered"
    groups: tuple[Group, ...]
    order_breaks: int  # consecutive groups (in prediction order) not consecutive in the reference
    unmatched_reference: int  # reference units in no group
    unmatched_hypothesis: int


class _Side:
    """The units of one side: strings joined by a space (characters) or token tuples (words)."""

    def __init__(self, units: Sequence[str] | Sequence[tuple[str, ...]], separator: bool) -> None:
        self.units = list(units)
        self.separator = separator

    def __len__(self) -> int:
        return len(self.units)

    def join(self, start: int, stop: int) -> str | tuple[str, ...]:
        if self.separator:
            return " ".join(self.units[start:stop])
        return tuple(chain.from_iterable(self.units[start:stop]))

    def cost(self, index: int) -> int:
        """Cost of leaving a unit unmatched: its length plus its separator."""
        return len(self.units[index]) + self.separator


def solution_counts(ref: _Side, hyp: _Side, groups: Sequence[Group]) -> EditCounts:
    """Edit counts of a grouping; units outside every group are deleted or inserted."""
    groups = [g for g in groups if g.ref_start < g.ref_stop and g.hyp_start < g.hyp_stop]
    covered_ref = [False] * len(ref)
    covered_hyp = [False] * len(hyp)
    total = EditCounts(0, 0, 0, 0, 0)
    for g in groups:
        for flags, start, stop in (
            (covered_ref, g.ref_start, g.ref_stop),
            (covered_hyp, g.hyp_start, g.hyp_stop),
        ):
            if any(flags[start:stop]):
                raise ValueError("a unit belongs to more than one group")
            flags[start:stop] = [True] * (stop - start)
        total += edit_counts(ref.join(g.ref_start, g.ref_stop), hyp.join(g.hyp_start, g.hyp_stop))
    deletions = sum(ref.cost(i) for i, done in enumerate(covered_ref) if not done)
    insertions = sum(hyp.cost(k) for k, done in enumerate(covered_hyp) if not done)
    if not groups:  # no separator between groups: one fewer separator on each non-empty side
        deletions -= ref.separator and len(ref) > 0
        insertions -= hyp.separator and len(hyp) > 0
    return EditCounts(
        len(ref.join(0, len(ref))),
        len(hyp.join(0, len(hyp))),
        total.substitutions,
        total.deletions + deletions,
        total.insertions + insertions,
    )


def lower_bound(ref: _Side, hyp: _Side) -> int:
    """max(|ref|, |hyp|) minus their multiset overlap: no alignment, in any order, has fewer
    errors, since each match pairs equal items."""
    r, h = ref.join(0, len(ref)), hyp.join(0, len(hyp))
    return max(len(r), len(h)) - sum((Counter(r) & Counter(h)).values())


def exhaustive_groups(ref: _Side, hyp: _Side) -> list[Group]:
    """The optimal grouping by exhaustive search: exponential, for small samples and tests."""
    runs = [(k, stop) for k in range(len(hyp)) for stop in range(k + 1, len(hyp) + 1)]

    @cache
    def distance(i: int, j: int, k: int, stop: int) -> int:
        return Levenshtein.distance(ref.join(i, j), hyp.join(k, stop))

    @cache
    def best(i: int, used: int) -> tuple[int, tuple[Group, ...]]:
        # Per-unit separator costs; the grouping without any group is compared separately.
        if i == len(ref):
            return sum(hyp.cost(k) for k in range(len(hyp)) if not used >> k & 1), ()
        cost, groups = best(i + 1, used)
        result = (cost + ref.cost(i), groups)
        for j in range(i + 1, len(ref) + 1):
            for k, stop in runs:
                bits = (1 << stop) - (1 << k)
                if used & bits:
                    continue
                cost, groups = best(j, used | bits)
                cost += distance(i, j, k, stop)
                if cost < result[0]:
                    result = (cost, (Group(i, j, k, stop), *groups))
        return result

    groups = list(best(0, 0)[1])
    if groups and solution_counts(ref, hyp, []).errors <= solution_counts(ref, hyp, groups).errors:
        return []
    return groups


# Scalable search -----------------------------------------------------------------------------


def _grams(text: str) -> set[str]:
    return {text[i : i + Q] for i in range(len(text) - Q + 1)} or {text}


class _Anchors:
    """Finds the reference line a piece of prediction text comes from, by shared q-grams."""

    def __init__(self, lines: Sequence[str]) -> None:
        self.size = [len(_grams(line)) for line in lines]
        self.index: dict[str, list[int]] = {}
        for i, line in enumerate(lines):
            for gram in _grams(line):
                self.index.setdefault(gram, []).append(i)

    def candidates(self, probe: str) -> list[tuple[int, int]]:
        """(votes, line) of the lines sharing enough q-grams with the probe, best first."""
        grams = _grams(probe)
        votes: Counter[int] = Counter()
        for gram in grams:
            postings = self.index.get(gram, ())
            if len(postings) <= MAX_POSTINGS:
                votes.update(postings)
        if not votes:
            return []
        top = max(votes.values())
        found = [
            (n, i)
            for i, n in votes.items()
            if n >= MIN_SCORE * min(len(grams), self.size[i]) and n >= MIN_SCORE * top
        ]
        return sorted(found, key=lambda item: (-item[0], item[1]))

    def find(self, probe: str, preferred: range) -> int | None:
        """The best line; a line in the preferred range (continuing the text) wins ties."""
        found = self.candidates(probe)
        if not found:
            return None
        for _, i in found:
            if i in preferred:
                return i
        return found[0][1]


def anchor_units(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[int, int] | None]:
    """The first and last reference line of every prediction unit, or None without evidence."""
    anchors = _Anchors(ref)
    length = max(8, min(80, round(median(len(line) for line in ref)))) if ref else 8
    result: list[tuple[int, int] | None] = []
    previous: int | None = None
    for text in hyp:
        follows = range(previous, previous + MAX_JUMP + 1) if previous is not None else range(0)
        if len(text) <= 1.5 * length:
            start = end = anchors.find(text, follows)
        else:
            start = anchors.find(text[:length], follows)
            # A unit spanning several lines ends after where it starts.
            steps = 2 * round(len(text) / length) + 2
            span = range(start, start + steps) if start is not None else range(0)
            end = anchors.find(text[-length:], span)
            if end is not None and (start is None or not 0 <= end - start < steps):
                # A start without context (such as a repeated line) is resolved from the end.
                start = anchors.find(text[:length], range(max(0, end - steps), end + 1))
                start = end if start is None or start > end else start
            end = start if end is None or end < start else end
        result.append(None if start is None else (start, end))
        previous = end if start is not None else previous
    return result


def block_order(anchors: Sequence[tuple[int, int] | None]) -> list[tuple[int, int]]:
    """Runs [start, stop) of consecutive prediction units with consistent anchors, put in
    reference order. Units without anchors stay with the preceding unit."""
    blocks: list[list] = []  # [start, stop, key, last anchored line]
    for k, anchor in enumerate(anchors):
        if blocks and (
            anchor is None
            or blocks[-1][3] is None
            or blocks[-1][3] <= anchor[0] <= blocks[-1][3] + MAX_JUMP
        ):
            block = blocks[-1]
            block[1] = k + 1
            if anchor is not None:
                block[2] = anchor[0] if block[2] is None else block[2]
                block[3] = anchor[1]
        else:
            blocks.append([k, k + 1, *(anchor or (None, None))])
    blocks.sort(key=lambda b: (b[2] is None, b[2] or 0, b[0]))
    merged: list[tuple[int, int]] = []
    for start, stop, *_ in blocks:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def _cuts(ref: _Side, hyp: _Side, blocks: Sequence[tuple[int, int]]) -> list[int]:
    """Reference line boundaries where the blocks, aligned in their new order, start."""
    sep = hyp.separator
    reordered = [hyp.join(start, stop) for start, stop in blocks]
    text = " ".join(reordered) if sep else tuple(chain.from_iterable(reordered))
    starts, position = [], 0
    for part in reordered:
        starts.append(position)
        position += len(part) + sep
    ref_text = ref.join(0, len(ref))
    line_starts, position = [], 0
    for i in range(len(ref)):
        line_starts.append(position)
        position += len(ref.units[i]) + ref.separator
    opcodes = Levenshtein.opcodes(ref_text, text)
    cuts, index = [0], 0
    for h in starts[1:]:
        while opcodes[index].dest_end <= h and index + 1 < len(opcodes):
            index += 1
        op = opcodes[index]
        r = op.src_start
        if op.tag in ("equal", "replace"):
            r += min(h - op.dest_start, op.src_end - op.src_start)
        i = bisect_right(line_starts, r) - 1
        if i < 0:
            cut = 0
        else:
            end = line_starts[i] + len(ref.units[i])
            cut = i if r - line_starts[i] < end - r else i + 1
        cuts.append(max(cut, cuts[-1]))
    cuts.append(len(ref))
    return cuts


def _groups(blocks: Sequence[tuple[int, int]], cuts: Sequence[int]) -> list[Group]:
    return [Group(cuts[s], cuts[s + 1], *block) for s, block in enumerate(blocks)]


def reordered_groups(ref: _Side, hyp: _Side, blocks: Sequence[tuple[int, int]]) -> list[Group]:
    """Groups of the reordered blocks, with every cut shifted by a few lines where that helps."""
    cuts = _cuts(ref, hyp, blocks)

    def cost(s: int, start: int, stop: int) -> int:
        if start >= stop:
            return sum(hyp.cost(k) for k in range(*blocks[s]))
        return Levenshtein.distance(ref.join(start, stop), hyp.join(*blocks[s]))

    for s in range(1, len(blocks)):
        low, high = cuts[s - 1], cuts[s + 1]
        best = cost(s - 1, low, cuts[s]) + cost(s, cuts[s], high), cuts[s]
        for shift in CUT_SHIFTS:
            cut = cuts[s] + shift
            if low <= cut <= high:
                best = min(best, (cost(s - 1, low, cut) + cost(s, cut, high), cut))
        cuts[s] = best[1]
    return _groups(blocks, cuts)


def _order_breaks(groups: Sequence[Group]) -> int:
    ranked = sorted(groups, key=lambda g: g.ref_start)
    rank = {g: n for n, g in enumerate(ranked)}
    in_prediction = [rank[g] for g in sorted(groups, key=lambda g: g.hyp_start)]
    return sum(b != a + 1 for a, b in zip(in_prediction, in_prediction[1:], strict=False))


def _result(ref: _Side, hyp: _Side, groups: list[Group], method: str, bound: int) -> Alignment:
    groups = [g for g in groups if g.ref_start < g.ref_stop and g.hyp_start < g.hyp_stop]
    counts = solution_counts(ref, hyp, groups)
    return Alignment(
        counts=counts,
        lower_bound=bound,
        exact=method == "exhaustive" or counts.errors == bound,
        method=method,
        groups=tuple(groups),
        order_breaks=_order_breaks(groups),
        unmatched_reference=len(ref) - sum(g.ref_stop - g.ref_start for g in groups),
        unmatched_hypothesis=len(hyp) - sum(g.hyp_stop - g.hyp_start for g in groups),
    )


def align(ref: _Side, hyp: _Side, blocks: Sequence[tuple[int, int]] | None) -> Alignment:
    """The best grouping found: exhaustive for small samples, otherwise the cheaper of the
    ordered alignment and the reordered blocks (ties keep the ordered alignment)."""
    bound = lower_bound(ref, hyp)
    if len(ref) <= EXHAUSTIVE_UNITS and len(hyp) <= EXHAUSTIVE_UNITS:
        return _result(ref, hyp, exhaustive_groups(ref, hyp), "exhaustive", bound)
    one = [Group(0, len(ref), 0, len(hyp))]
    ordered = _result(ref, hyp, one, "ordered", bound)
    if not blocks or len(blocks) == 1:
        return ordered
    reordered = _result(ref, hyp, reordered_groups(ref, hyp, blocks), "reordered", bound)
    return reordered if reordered.counts.errors < ordered.counts.errors else ordered


@dataclass(frozen=True)
class FlexibleScores:
    chars: Alignment  # flexible CER
    words: Alignment  # flexible WER
    reference_units: int
    hypothesis_units: int
    anchored_units: int  # prediction units with retrieval evidence
    reference_lines: tuple[int, ...]  # input line index of every reference unit
    hypothesis_lines: tuple[int, ...]  # input line index of every prediction unit

    def line_ranges(self, alignment: Alignment) -> list[tuple[int, int, int, int]]:
        """The groups as half-open ranges of input lines, (reference start, stop, prediction
        start, stop), in prediction order. Empty lines inside a range belong to no unit."""
        ref, hyp = self.reference_lines, self.hypothesis_lines
        return [
            (ref[g.ref_start], ref[g.ref_stop - 1] + 1, hyp[g.hyp_start], hyp[g.hyp_stop - 1] + 1)
            for g in sorted(alignment.groups, key=lambda g: g.hyp_start)
        ]


def score_flexible(
    reference_lines: Sequence[str], hypothesis_lines: Sequence[str], form: str = "NFC"
) -> FlexibleScores:
    """Flexible CER and WER of one sample, each optimized separately on the same units."""
    ref = reference_units(reference_lines, form)
    hyp = reference_units(hypothesis_lines, form)
    anchors = anchor_units(ref, hyp) if ref and hyp else [None] * len(hyp)
    blocks = block_order(anchors) if hyp else []
    return FlexibleScores(
        chars=align(_Side(ref, True), _Side(hyp, True), blocks),
        words=align(
            _Side([tuple(words(u)) for u in ref], False),
            _Side([tuple(words(u)) for u in hyp], False),
            blocks,
        ),
        reference_units=len(ref),
        hypothesis_units=len(hyp),
        anchored_units=sum(anchor is not None for anchor in anchors),
        reference_lines=_unit_lines(reference_lines, form),
        hypothesis_lines=_unit_lines(hypothesis_lines, form),
    )
