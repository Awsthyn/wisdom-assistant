"""Parser for exact Bible references such as `John 3:16`, `Jn 3:16-18` or `Psalm 23`.

`parse_reference` returns:

- `None` when the text is not a reference (it should go to semantic search),
- a `Reference` when it is a valid reference to any of the 66 books,
- or raises `InvalidReferenceError` when it is clearly a reference that cannot
  exist, e.g. `John 99:1` (John has 21 chapters).

Chapters are checked against the catalog. Verse numbers are only checked for
shape here; whether a verse exists is answered by the database.
"""

import re
from dataclasses import dataclass
from typing import Final

from app.core.books import BIBLE_BOOKS, BookInfo

# A reference may span at most this many chapters (e.g. `Psalm 120-134` is too long).
MAX_REFERENCE_CHAPTERS: Final = 5

# Common English abbreviations, without the leading number of numbered books
# (`1 John` → `1` + `john`/`jn`/...). Book names and OSIS codes are added automatically.
_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "Gen": ("ge", "gn"),
    "Exod": ("exo", "ex"),
    "Lev": ("le", "lv"),
    "Num": ("nu", "nm", "nb"),
    "Deut": ("deu", "dt"),
    "Josh": ("jos", "jsh"),
    "Judg": ("jdg", "jdgs", "jg"),
    "Ruth": ("rth", "ru"),
    "1Sam": ("sa", "sm"),
    "2Sam": ("sa", "sm"),
    "1Kgs": ("ki", "kin", "kgs"),
    "2Kgs": ("ki", "kin", "kgs"),
    "1Chr": ("ch", "chron"),
    "2Chr": ("ch", "chron"),
    "Ezra": ("ezr",),
    "Neh": ("ne",),
    "Esth": ("est", "es"),
    "Job": ("jb",),
    "Ps": ("psa", "psalm", "pss", "psm"),
    "Prov": ("pro", "prv", "pr"),
    "Eccl": ("ecc", "ec", "eccles", "qoh", "qoheleth"),
    "Song": ("sng", "sos", "songofsolomon", "canticles", "cant"),
    "Isa": ("is",),
    "Jer": ("je", "jr"),
    "Lam": ("la",),
    "Ezek": ("eze", "ezk"),
    "Dan": ("da", "dn"),
    "Hos": ("ho",),
    "Joel": ("jl",),
    "Amos": ("am",),
    "Obad": ("ob", "oba"),
    "Jonah": ("jon", "jnh"),
    "Mic": ("mc",),
    "Nah": ("na",),
    "Hab": ("hb",),
    "Zeph": ("zep", "zp"),
    "Hag": ("hg",),
    "Zech": ("zec", "zc"),
    "Mal": ("ml",),
    "Matt": ("mat", "mt"),
    "Mark": ("mrk", "mk", "mr"),
    "Luke": ("luk", "lk"),
    "John": ("joh", "jhn", "jn"),
    "Acts": ("act", "ac"),
    "Rom": ("ro", "rm"),
    "1Cor": ("co",),
    "2Cor": ("co",),
    "Gal": ("ga",),
    "Eph": ("ephes",),
    "Phil": ("php", "pp"),
    "Col": (),
    "1Thess": ("th", "thes"),
    "2Thess": ("th", "thes"),
    "1Tim": ("ti", "tm"),
    "2Tim": ("ti", "tm"),
    "Titus": ("tit",),
    "Phlm": ("philem", "phm"),
    "Heb": (),
    "Jas": ("jam", "jm"),
    "1Pet": ("pe", "pt"),
    "2Pet": ("pe", "pt"),
    "1John": ("jn", "jo", "jhn"),
    "2John": ("jn", "jo", "jhn"),
    "3John": ("jn", "jo", "jhn"),
    "Jude": ("jud", "jd"),
    "Rev": ("re", "rv", "revelations"),
}

# Leading ordinals accepted for numbered books: `First John`, `II Kings`, `1st Peter`.
_ORDINALS: Final = {
    "1": "1",
    "2": "2",
    "3": "3",
    "i": "1",
    "ii": "2",
    "iii": "3",
    "1st": "1",
    "2nd": "2",
    "3rd": "3",
    "first": "1",
    "second": "2",
    "third": "3",
}

_REFERENCE_RE: Final = re.compile(
    r"""
    ^(?P<book>(?:[1-3]|i{1,3}|1st|2nd|3rd|first|second|third)?\s*[a-z][a-z. ]*?)\.?\s*
    (?P<c1>\d{1,3})
    (?:[:.](?P<v1>\d{1,3}))?
    (?:\s*[-\N{EN DASH}\N{EM DASH}]\s*(?:(?P<c2>\d{1,3})[:.])?(?P<n2>\d{1,3}))?
    $
    """,
    re.VERBOSE,
)


class InvalidReferenceError(ValueError):
    """The text is a reference to a known book, but that passage cannot exist."""


@dataclass(frozen=True, slots=True)
class Reference:
    """An inclusive span from `start_chapter:start_verse` to `end_chapter:end_verse`.

    `end_verse is None` means "to the end of `end_chapter`".
    """

    book: BookInfo
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int | None


def _normalize(name: str) -> str:
    return re.sub(r"[\s.]", "", name.lower())


def _build_alias_index() -> dict[str, BookInfo]:
    index: dict[str, BookInfo] = {}

    def add(alias: str, book: BookInfo) -> None:
        key = _normalize(alias)
        existing = index.get(key)
        if existing is not None and existing is not book:
            raise RuntimeError(f"alias {key!r} maps to {existing.osis_code} and {book.osis_code}")
        index[key] = book

    for book in BIBLE_BOOKS:
        add(book.osis_code, book)
        add(book.name, book)
        add(book.reference_name, book)
        number = book.osis_code[0] if book.osis_code[0].isdigit() else ""
        for alias in _ALIASES.get(book.osis_code, ()):
            add(number + alias, book)
    return index


_ALIAS_INDEX: Final = _build_alias_index()


def find_book(name: str) -> BookInfo | None:
    """Resolve a book name or abbreviation (`Psalm`, `Ps`, `1 Jn`, `First John`)."""
    words = name.lower().replace(".", " ").split()
    if len(words) > 1 and words[0] in _ORDINALS:
        words[0] = _ORDINALS[words[0]]
    return _ALIAS_INDEX.get("".join(words))


def parse_reference(text: str) -> Reference | None:
    """Parse `text` as an exact reference. See the module docstring for the contract."""
    match = _REFERENCE_RE.match(" ".join(text.lower().split()))
    if match is None:
        return None
    book = find_book(match["book"])
    if book is None:
        return None

    c1 = int(match["c1"])
    v1 = int(match["v1"]) if match["v1"] else None
    c2 = int(match["c2"]) if match["c2"] else None
    n2 = int(match["n2"]) if match["n2"] else None
    label = text.strip()

    if book.chapters == 1 and v1 is None and c2 is None:
        # Single-chapter books are cited by verse: `Jude 3` is Jude 1:3.
        start = (1, c1)
        end: tuple[int, int | None] = (1, n2 if n2 is not None else c1)
    elif v1 is None:
        if c2 is not None:
            raise InvalidReferenceError(f"'{label}' mixes a chapter range with a verse")
        # `Psalm 23` (whole chapter) or `Psalm 23-24` (chapter range).
        start = (c1, 1)
        end = (n2 if n2 is not None else c1, None)
    elif n2 is None:
        start = end = (c1, v1)  # `John 3:16`
    elif c2 is None:
        start, end = (c1, v1), (c1, n2)  # `John 3:16-18`
    else:
        start, end = (c1, v1), (c2, n2)  # `John 3:16-4:2`

    _validate(book, label, start, end)
    return Reference(book, start[0], start[1], end[0], end[1])


def _validate(
    book: BookInfo, label: str, start: tuple[int, int], end: tuple[int, int | None]
) -> None:
    for chapter in (start[0], end[0]):
        if not 1 <= chapter <= book.chapters:
            raise InvalidReferenceError(
                f"'{label}': {book.name} has {book.chapters} chapter"
                f"{'' if book.chapters == 1 else 's'}"
            )
    if start[1] < 1 or (end[1] is not None and end[1] < 1):
        raise InvalidReferenceError(f"'{label}': verse numbers start at 1")
    end_key = (end[0], end[1] if end[1] is not None else start[1])
    if end_key < start:
        raise InvalidReferenceError(f"'{label}': the range ends before it starts")
    if end[0] - start[0] + 1 > MAX_REFERENCE_CHAPTERS:
        raise InvalidReferenceError(
            f"'{label}': a reference may span at most {MAX_REFERENCE_CHAPTERS} chapters"
        )
