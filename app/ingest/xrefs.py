"""Parser for OpenBible.info's `cross_references.txt` (CC-BY).

Tab-separated, one header line, OSIS references with dots:

    From Verse  To Verse                Votes
    Gen.1.1     Rom.1.19-Rom.1.20       60
    Ps.23.1     John.10.11              212

The target may be a single verse or a range; ranges may cross chapters
(`Rom.1.32-Rom.2.1`). A handful of ranges cross books; those are skipped.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

_REF_RE: Final = re.compile(r"^(?P<book>[1-3]?[A-Za-z]+)\.(?P<chapter>\d+)\.(?P<verse>\d+)$")


class XrefParseError(ValueError):
    """A line does not have the expected `from<TAB>to<TAB>votes` shape."""


@dataclass(frozen=True, slots=True)
class ParsedXref:
    from_book: str
    from_chapter: int
    from_verse: int
    to_book: str
    to_chapter: int
    to_start_verse: int
    to_end_chapter: int
    to_end_verse: int
    votes: int


@dataclass(slots=True)
class XrefParseResult:
    rows: list[ParsedXref] = field(default_factory=list[ParsedXref])
    total: int = 0
    skipped_low_votes: int = 0
    skipped_cross_book: int = 0
    duplicates: int = 0


def _ref(text: str, line_number: int) -> tuple[str, int, int]:
    match = _REF_RE.match(text)
    if match is None:
        raise XrefParseError(f"line {line_number}: bad reference {text!r}")
    return match["book"], int(match["chapter"]), int(match["verse"])


def parse_xrefs(lines: Iterable[str], *, min_votes: int) -> XrefParseResult:
    """Parse every line, keeping rows with `votes >= min_votes`.

    Duplicate source/target pairs keep the highest vote count.
    """
    result = XrefParseResult()
    kept: dict[tuple[object, ...], ParsedXref] = {}
    for line_number, raw in enumerate(lines, start=1):
        line = raw.rstrip("\r\n")
        if not line.strip() or line.startswith("From Verse"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise XrefParseError(f"line {line_number}: expected 3 tab-separated fields")
        source, target, votes_text = fields
        try:
            votes = int(votes_text)
        except ValueError as exc:
            raise XrefParseError(f"line {line_number}: bad votes {votes_text!r}") from exc
        result.total += 1

        from_book, from_chapter, from_verse = _ref(source, line_number)
        start_text, _, end_text = target.partition("-")
        to_book, to_chapter, to_start = _ref(start_text, line_number)
        end_book, end_chapter, end_verse = (
            _ref(end_text, line_number) if end_text else (to_book, to_chapter, to_start)
        )
        if votes < min_votes:
            result.skipped_low_votes += 1
            continue
        if end_book != to_book:
            result.skipped_cross_book += 1
            continue

        row = ParsedXref(
            from_book,
            from_chapter,
            from_verse,
            to_book,
            to_chapter,
            to_start,
            end_chapter,
            end_verse,
            votes,
        )
        key = (from_book, from_chapter, from_verse, to_book, to_chapter, to_start)
        key += (end_chapter, end_verse)
        previous = kept.get(key)
        if previous is not None:
            result.duplicates += 1
            if previous.votes >= votes:
                continue
        kept[key] = row
    result.rows = list(kept.values())
    return result
