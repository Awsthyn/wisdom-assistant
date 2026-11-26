"""Sliding-window chunking of a chapter into passages.

Windows of `size` verses overlapping by `overlap` verses, never crossing a
chapter boundary. The last window of a chapter may be shorter, and a window is
not emitted when it would add no verse that the previous one lacked.
"""

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_SIZE = 5
DEFAULT_OVERLAP = 2


@dataclass(frozen=True, slots=True)
class VerseText:
    verse: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chapter: int
    start_verse: int
    end_verse: int
    text: str


def window_bounds(
    count: int, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[tuple[int, int]]:
    """Return `(start, end)` index pairs (end exclusive) covering `count` items."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    step = size - overlap
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + size, count)
        bounds.append((start, end))
        if end == count:
            break
        start += step
    return bounds


def format_passage_text(verses: Sequence[VerseText]) -> str:
    """Concatenate verses with their numbers: `[1] ... [2] ...`."""
    return " ".join(f"[{v.verse}] {v.text}" for v in verses)


def format_reference(reference_name: str, chapter: int, start_verse: int, end_verse: int) -> str:
    """`Psalm 23:1-5`, or `Psalm 23:1` for a single verse."""
    span = str(start_verse) if start_verse == end_verse else f"{start_verse}-{end_verse}"
    return f"{reference_name} {chapter}:{span}"


def chunk_chapter(
    chapter: int,
    verses: Sequence[VerseText],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk one chapter's verses (already sorted by verse number)."""
    chunks: list[Chunk] = []
    for start, end in window_bounds(len(verses), size, overlap):
        window = verses[start:end]
        chunks.append(
            Chunk(
                chapter=chapter,
                start_verse=window[0].verse,
                end_verse=window[-1].verse,
                text=format_passage_text(window),
            )
        )
    return chunks
