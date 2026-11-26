import pytest

from app.core.books import BIBLE_BOOKS
from app.core.references import (
    MAX_REFERENCE_CHAPTERS,
    InvalidReferenceError,
    find_book,
    parse_reference,
)

Span = tuple[str, int, int, int, int | None]


def _span(text: str) -> Span:
    reference = parse_reference(text)
    assert reference is not None, text
    return (
        reference.book.osis_code,
        reference.start_chapter,
        reference.start_verse,
        reference.end_chapter,
        reference.end_verse,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("John 3:16", ("John", 3, 16, 3, 16)),
        ("Jn 3:16-18", ("John", 3, 16, 3, 18)),
        ("Psalm 23", ("Ps", 23, 1, 23, None)),
        ("Psalms 23:1", ("Ps", 23, 1, 23, 1)),
        ("Ps. 119:105", ("Ps", 119, 105, 119, 105)),
        ("prov 3:5-6", ("Prov", 3, 5, 3, 6)),
        ("Proverbs 18.21", ("Prov", 18, 21, 18, 21)),
        ("Eccl 3:1-8", ("Eccl", 3, 1, 3, 8)),
        ("Ecclesiastes 12", ("Eccl", 12, 1, 12, None)),
        ("Song of Songs 2:4", ("Song", 2, 4, 2, 4)),
        ("Song of Solomon 8:6", ("Song", 8, 6, 8, 6)),
        ("Job 38:4", ("Job", 38, 4, 38, 4)),
        ("1 John 4:8", ("1John", 4, 8, 4, 8)),
        ("1Jn 4:7-8", ("1John", 4, 7, 4, 8)),
        ("First Corinthians 13:4", ("1Cor", 13, 4, 13, 4)),
        ("II Kings 2:11", ("2Kgs", 2, 11, 2, 11)),
        ("Gen 1:1-2:3", ("Gen", 1, 1, 2, 3)),
        ("Psalm 23-24", ("Ps", 23, 1, 24, None)),
        ("John 3:16\N{EN DASH}18", ("John", 3, 16, 3, 18)),
        ("  JOHN   3:16  ", ("John", 3, 16, 3, 16)),
        ("Jude 3", ("Jude", 1, 3, 1, 3)),
        ("Obadiah 1:4", ("Obad", 1, 4, 1, 4)),
        ("Mt 5:3", ("Matt", 5, 3, 5, 3)),
        ("Rev 22:21", ("Rev", 22, 21, 22, 21)),
    ],
)
def test_valid_references_parse_to_the_right_span(text: str, expected: Span) -> None:
    """Protects reference parsing and abbreviations; rejects a wrong book, chapter or verse span."""
    assert _span(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "the tongue has power of life and death",
        "don't boast about tomorrow",
        "job loss",
        "love 3",
        "3:16",
        "psalm",
        "Psalm twenty-three",
        "Hezekiah 1:1",
    ],
)
def test_free_text_is_not_a_reference(text: str) -> None:
    """Protects semantic search routing; rejects treating ordinary text as a reference."""
    assert parse_reference(text) is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("John 99:1", "John has 21 chapters"),
        ("Psalm 151", "Psalms has 150 chapters"),
        ("Jude 2:1", "Jude has 1 chapter"),
        ("Prov 0:1", "Proverbs has 31 chapters"),
        ("John 3:0", "verse numbers start at 1"),
        ("John 3:18-16", "ends before it starts"),
        ("Psalm 24-23", "ends before it starts"),
        ("Psalm 23-24:3", "mixes a chapter range"),
    ],
)
def test_impossible_references_raise(text: str, message: str) -> None:
    """Protects against silently searching nonsense; rejects `John 99:1` and similar as valid."""
    with pytest.raises(InvalidReferenceError, match=message):
        parse_reference(text)


def test_reference_span_is_capped() -> None:
    """Protects response size; rejects references spanning more than the chapter cap."""
    parse_reference(f"Psalm 1-{MAX_REFERENCE_CHAPTERS}")
    with pytest.raises(InvalidReferenceError, match="at most"):
        parse_reference(f"Psalm 1-{MAX_REFERENCE_CHAPTERS + 1}")


def test_every_book_resolves_by_name_and_osis_code() -> None:
    """Protects the alias index; rejects a book that cannot be found by its own name or code."""
    for book in BIBLE_BOOKS:
        assert find_book(book.name) is book
        assert find_book(book.osis_code) is book


def test_catalog_chapter_counts_match_the_web() -> None:
    """Protects chapter validation data; rejects a catalog whose chapter total differs (1189)."""
    by_osis = {b.osis_code: b for b in BIBLE_BOOKS}

    assert sum(b.chapters for b in BIBLE_BOOKS) == 1189
    assert [by_osis[c].chapters for c in ("Job", "Ps", "Prov", "Eccl", "Song")] == [
        42,
        150,
        31,
        12,
        8,
    ]


def test_chapter_urls_follow_ebible_padding() -> None:
    """Protects result links; rejects `PSA23.htm` (404 on ebible.org) or a wrong book code."""
    by_osis = {b.osis_code: b for b in BIBLE_BOOKS}

    assert by_osis["Ps"].chapter_url(23) == "https://ebible.org/engwebp/PSA023.htm"
    assert by_osis["John"].chapter_url(3) == "https://ebible.org/engwebp/JHN03.htm"
    assert by_osis["Song"].chapter_url(1) == "https://ebible.org/engwebp/SNG01.htm"
