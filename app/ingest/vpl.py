"""Parser for ebible.org verse-per-line files (`*_vpl.txt`).

Each line is `<CODE> <chapter>:<verse> <text>`, e.g. `PSA 23:1 A Psalm by David. ...`.
A few verses are present with no text (verses the WEB omits on text-critical
grounds, e.g. `LUK 17:36`); those are skipped, not loaded as empty strings.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

_LINE = re.compile(
    r"^(?P<code>[1-3A-Z][A-Z0-9]{2}) (?P<chapter>\d+):(?P<verse>\d+)(?: (?P<text>.*))?$"
)


class VplParseError(ValueError):
    """A line does not follow the VPL format."""


@dataclass(frozen=True, slots=True)
class ParsedVerse:
    book_code: str
    chapter: int
    verse: int
    text: str


def parse_vpl(lines: Iterable[str]) -> Iterator[ParsedVerse]:
    """Yield one `ParsedVerse` per non-empty verse line, in file order."""
    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip("\r\n").lstrip("﻿")
        if not line.strip():
            continue
        match = _LINE.match(line)
        if match is None:
            raise VplParseError(f"line {number}: not a VPL verse line: {line[:60]!r}")
        text = " ".join((match["text"] or "").split())
        if not text:
            continue
        yield ParsedVerse(
            book_code=match["code"],
            chapter=int(match["chapter"]),
            verse=int(match["verse"]),
            text=text,
        )
