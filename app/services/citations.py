"""Post-generation citation check. Pure code: the model is never asked to verify itself.

Every `[...]` in the answer that parses as a reference (with the B3 parser) is a
citation. A citation is valid only if its book is in the corpus and every verse it
names lies inside the passages that were retrieved for this answer. Everything else
is reported as discarded, with the reason, so the UI can strike it out.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.core.books import BookInfo
from app.core.corpus import CORPUS_BOOKS
from app.core.references import InvalidReferenceError, parse_reference
from app.services.search import SearchHit

# `[Proverbs 3:5]`, `[Prov 3:5-6; 22:9]`. Square brackets only, on one line.
_BRACKET_RE: Final = re.compile(r"\[([^\[\]\n]{2,120})\]")

# The chapter/verse tail of `Prov 3:5-6`, leaving the book name.
_TRAILING_NUMBERS_RE: Final = re.compile(r"\s*\d+(?:[:.]\d+)?(?:\s*[-\N{EN DASH}]\s*[\d:.]+)?\s*$")
# Models group citations with `;` or `,`: `[Proverbs 14:31, Proverbs 14:21]`.
_SEPARATOR_RE: Final = re.compile(r"[;,]")
_CHAPTER_VERSE_RE: Final = re.compile(r"^\d+[:.]\d+(?:\s*[-\N{EN DASH}]\s*[\d:.]+)?$")
_VERSE_ONLY_RE: Final = re.compile(r"^\d+(?:\s*[-\N{EN DASH}]\s*\d+)?$")
_CHAPTER_RE: Final = re.compile(r"(\d+)[:.]\d+")

REASON_OUTSIDE_CORPUS: Final = "outside the corpus"
REASON_NOT_RETRIEVED: Final = "not in the retrieved passages"
REASON_INVALID: Final = "not a valid reference"


@dataclass(frozen=True, slots=True)
class Citation:
    cited: str  # as written by the model, e.g. "Prov 19:17"
    reference: str  # normalized, e.g. "Proverbs 19:17"
    book: str | None  # OSIS code when the book is known
    url: str | None  # chapter page on ebible.org when the book is known
    valid: bool
    reason: str | None = None  # why it was discarded
    outside_corpus: bool = False  # agent only: a valid citation of a non-wisdom book


@dataclass(frozen=True, slots=True)
class CitationReport:
    valid: list[Citation]
    discarded: list[Citation]


def _parts(bracket: str) -> list[str]:
    """Split a bracket into single references, carrying the book and chapter over.

    `Prov 3:5; 22:9` -> `Prov 3:5`, `Prov 22:9`
    `Proverbs 14:31, Proverbs 14:21` -> `Proverbs 14:31`, `Proverbs 14:21`
    `Prov 3:5, 7` -> `Prov 3:5`, `Prov 3:7`
    """
    parts: list[str] = []
    book_prefix = ""
    chapter: str | None = None
    for raw in _SEPARATOR_RE.split(bracket):
        part = raw.strip()
        if not part:
            continue
        if book_prefix and _CHAPTER_VERSE_RE.match(part):
            part = f"{book_prefix} {part}"
        elif book_prefix and chapter and _VERSE_ONLY_RE.match(part):
            part = f"{book_prefix} {chapter}:{part}"
        elif not part[0].isdigit():
            book_prefix = _TRAILING_NUMBERS_RE.sub("", part)
        chapter_match = _CHAPTER_RE.search(part)
        chapter = chapter_match.group(1) if chapter_match else None
        parts.append(part)
    return parts


def _covered_verses(passages: Sequence[SearchHit]) -> set[tuple[str, int, int]]:
    return {
        (p.book.osis_code, p.chapter, verse)
        for p in passages
        for verse in range(p.start_verse, p.end_verse + 1)
    }


def _discard(cited: str, reference: str, book: BookInfo | None, reason: str) -> Citation:
    """Discarded citations carry no URL: the UI shows them struck out, without a link."""
    return Citation(
        cited=cited,
        reference=reference,
        book=book.osis_code if book else None,
        url=None,
        valid=False,
        reason=reason,
    )


def verify_citations(
    answer: str, passages: Sequence[SearchHit], *, allow_outside_corpus: bool = False
) -> CitationReport:
    """Classify every citation in `answer` against the retrieved `passages`.

    Chat (default): a citation to a book outside the corpus is always discarded.
    Agent (`allow_outside_corpus=True`): it is valid when a tool returned that text,
    and is flagged `outside_corpus` so the UI can present it as a cross reference.
    """
    covered = _covered_verses(passages)
    valid: list[Citation] = []
    discarded: list[Citation] = []
    seen: set[str] = set()

    for match in _BRACKET_RE.finditer(answer):
        for cited in _parts(match.group(1)):
            try:
                reference = parse_reference(cited)
            except InvalidReferenceError:
                if cited not in seen:
                    seen.add(cited)
                    discarded.append(_discard(cited, cited, None, REASON_INVALID))
                continue
            if reference is None:
                continue  # e.g. a verse marker like `[17]`: not a citation
            normalized = reference.label
            if normalized in seen:
                continue
            seen.add(normalized)
            book = reference.book

            outside_corpus = book.osis_code not in CORPUS_BOOKS
            if outside_corpus and not allow_outside_corpus:
                discarded.append(_discard(cited, normalized, book, REASON_OUTSIDE_CORPUS))
                continue
            single_span = (
                reference.end_verse is not None and reference.start_chapter == reference.end_chapter
            )
            if not single_span or not all(
                (book.osis_code, reference.start_chapter, verse) in covered
                for verse in range(reference.start_verse, (reference.end_verse or 0) + 1)
            ):
                discarded.append(_discard(cited, normalized, book, REASON_NOT_RETRIEVED))
                continue
            valid.append(
                Citation(
                    cited=cited,
                    reference=normalized,
                    book=book.osis_code,
                    url=book.chapter_url(reference.start_chapter),
                    valid=True,
                    outside_corpus=outside_corpus,
                )
            )
    return CitationReport(valid=valid, discarded=discarded)
