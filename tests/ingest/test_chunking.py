import itertools

import pytest

from app.ingest.chunking import (
    VerseText,
    chunk_chapter,
    format_passage_text,
    format_reference,
    window_bounds,
)


def _verses(count: int) -> list[VerseText]:
    return [VerseText(n, f"v{n}") for n in range(1, count + 1)]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, [(1, 1)]),
        (5, [(1, 5)]),
        (6, [(1, 5), (4, 6)]),
        (8, [(1, 5), (4, 8)]),
        (9, [(1, 5), (4, 8), (7, 9)]),
        (11, [(1, 5), (4, 8), (7, 11)]),
    ],
)
def test_windows_of_five_overlap_by_two(count: int, expected: list[tuple[int, int]]) -> None:
    """Protects the 5/2 window rule; rejects wrong stride, missing tail or redundant tail."""
    spans = [(c.start_verse, c.end_verse) for c in chunk_chapter(1, _verses(count))]

    assert spans == expected


def test_consecutive_windows_share_exactly_two_verses() -> None:
    """Protects the overlap; rejects windows that overlap by more or fewer than 2 verses."""
    chunks = chunk_chapter(1, _verses(20))

    for previous, current in itertools.pairwise(chunks):
        assert previous.end_verse - current.start_verse + 1 == 2


def test_every_verse_is_covered() -> None:
    """Protects coverage; rejects chunking that drops verses at the end of a chapter."""
    covered = {
        n for c in chunk_chapter(3, _verses(17)) for n in range(c.start_verse, c.end_verse + 1)
    }

    assert covered == set(range(1, 18))


def test_chunk_text_carries_verse_numbers_and_chapter() -> None:
    """Protects citation data in passages; rejects text without verse markers or wrong chapter."""
    [chunk] = chunk_chapter(23, [VerseText(1, "The LORD is my shepherd;"), VerseText(2, "He...")])

    assert chunk.chapter == 23
    assert chunk.text == "[1] The LORD is my shepherd; [2] He..."


def test_empty_chapter_produces_no_chunks() -> None:
    """Protects edge cases; rejects emitting an empty passage for a chapter with no verses."""
    assert chunk_chapter(1, []) == []


def test_invalid_window_parameters_are_rejected() -> None:
    """Protects against infinite loops; rejects overlap >= size."""
    with pytest.raises(ValueError, match="overlap"):
        window_bounds(10, size=3, overlap=3)


def test_reference_formatting() -> None:
    """Protects the citation format; rejects `Psalm 23:1-1` or a missing range."""
    assert format_reference("Psalm", 23, 1, 5) == "Psalm 23:1-5"
    assert format_reference("Proverbs", 3, 5, 5) == "Proverbs 3:5"


def test_passage_text_format() -> None:
    """Protects the stored passage text shape; rejects separators that change between runs."""
    assert format_passage_text([VerseText(4, "a"), VerseText(5, "b")]) == "[4] a [5] b"
