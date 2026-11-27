from app.core.books import BIBLE_BOOKS
from app.services.citations import (
    REASON_INVALID,
    REASON_NOT_RETRIEVED,
    REASON_OUTSIDE_CORPUS,
    CitationReport,
    verify_citations,
)
from app.services.search import SearchHit

_BOOKS = {b.osis_code: b for b in BIBLE_BOOKS}


def _hit(code: str, chapter: int, start: int, end: int) -> SearchHit:
    name = _BOOKS[code].reference_name
    return SearchHit(
        book=_BOOKS[code],
        chapter=chapter,
        start_verse=start,
        end_verse=end,
        reference=f"{name} {chapter}:{start}-{end}",
        text="...",
        in_corpus=True,
        score=0.8,
    )


PASSAGES = [_hit("Prov", 19, 15, 19), _hit("Prov", 19, 18, 22), _hit("Ps", 41, 1, 5)]


def _check(answer: str) -> CitationReport:
    return verify_citations(answer, PASSAGES)


def _valid(report: CitationReport) -> list[str]:
    return [c.reference for c in report.valid]


def _discarded(report: CitationReport) -> list[tuple[str, str | None]]:
    return [(c.reference, c.reason) for c in report.discarded]


def test_citation_inside_a_retrieved_passage_is_valid() -> None:
    """Protects grounded citations; rejects discarding a verse that was retrieved."""
    report = _check("Kindness to the poor is a loan to the LORD [Proverbs 19:17].")

    assert _valid(report) == ["Proverbs 19:17"]
    assert report.valid[0].url == "https://ebible.org/engwebp/PRO19.htm"
    assert report.discarded == []


def test_abbreviated_citation_is_normalized() -> None:
    """Protects parsing model citations with the B3 parser; rejects missing `Prov`/`Ps` forms."""
    report = _check("See [Prov 19:17] and [Ps 41:1].")

    assert _valid(report) == ["Proverbs 19:17", "Psalm 41:1"]


def test_invented_citation_is_discarded() -> None:
    """Protects principle 4; rejects keeping a real corpus verse that was not retrieved."""
    report = _check("Also [Proverbs 22:9] and [Proverbs 19:17].")

    assert _valid(report) == ["Proverbs 19:17"]
    assert _discarded(report) == [("Proverbs 22:9", REASON_NOT_RETRIEVED)]
    assert report.discarded[0].url is None


def test_out_of_corpus_citation_is_always_discarded() -> None:
    """Protects the corpus boundary in chat; rejects [John 10:11] even though the verse exists."""
    report = _check("The good shepherd [John 10:11].")

    assert report.valid == []
    assert _discarded(report) == [("John 10:11", REASON_OUTSIDE_CORPUS)]


def test_range_must_be_fully_covered() -> None:
    """Protects partial matches; rejects a range that runs past the retrieved verses."""
    report = _check("[Psalm 41:1-3] but not [Psalm 41:4-8].")

    assert _valid(report) == ["Psalm 41:1-3"]
    assert _discarded(report) == [("Psalm 41:4-8", REASON_NOT_RETRIEVED)]


def test_range_covered_by_overlapping_passages_is_valid() -> None:
    """Protects citations spanning two retrieved windows; rejects requiring a single passage."""
    report = _check("[Proverbs 19:16-21]")

    assert _valid(report) == ["Proverbs 19:16-21"]


def test_whole_chapter_citation_is_discarded() -> None:
    """Protects verse-level grounding; rejects [Psalm 41] when only some verses were retrieved."""
    report = _check("[Psalm 41]")

    assert _discarded(report) == [("Psalm 41", REASON_NOT_RETRIEVED)]


def test_impossible_reference_is_discarded_as_invalid() -> None:
    """Protects against nonsense citations; rejects silently ignoring [Psalm 151:1]."""
    report = _check("[Psalm 151:1]")

    assert _discarded(report) == [("Psalm 151:1", REASON_INVALID)]


def test_semicolon_list_carries_the_book_over() -> None:
    """Protects grouped citations; rejects reading `19:17` in `[Prov 19:15; 19:17]` as no book."""
    report = _check("[Prov 19:15; 19:17]")

    assert _valid(report) == ["Proverbs 19:15", "Proverbs 19:17"]


def test_verse_markers_and_other_brackets_are_not_citations() -> None:
    """Protects against false positives; rejects treating `[17]` or `[sic]` as citations."""
    report = _check("As verse [17] says [sic], see [1].")

    assert report.valid == []
    assert report.discarded == []


def test_repeated_citations_are_reported_once() -> None:
    """Protects a clean citations list; rejects duplicates for the same normalized verse."""
    report = _check("[Proverbs 19:17] ... [Prov 19:17] ... [John 10:11] [Jn 10:11]")

    assert _valid(report) == ["Proverbs 19:17"]
    assert _discarded(report) == [("John 10:11", REASON_OUTSIDE_CORPUS)]


def test_comma_separated_citations_are_each_checked() -> None:
    """Protects grouped citations as Gemini writes them; rejects leaving them unverified."""
    report = _check("Kindness honors God [Proverbs 19:17, Psalm 41:1, Proverbs 22:9].")

    assert _valid(report) == ["Proverbs 19:17", "Psalm 41:1"]
    assert _discarded(report) == [("Proverbs 22:9", REASON_NOT_RETRIEVED)]


def test_comma_verse_list_carries_book_and_chapter() -> None:
    """Protects `[Prov 19:15, 17]` and `[Prov 19:15, 20:1]`; rejects reading `17` as no citation."""
    report = _check("[Prov 19:15, 17] and [Prov 19:16, 20:1]")

    assert _valid(report) == ["Proverbs 19:15", "Proverbs 19:17", "Proverbs 19:16"]
    assert _discarded(report) == [("Proverbs 20:1", REASON_NOT_RETRIEVED)]
