import pytest

from app.ingest.xrefs import ParsedXref, XrefParseError, parse_xrefs

HEADER = "From Verse\tTo Verse\tVotes\t#www.openbible.info CC-BY 2026-11-23\n"


def test_single_verse_and_range_targets() -> None:
    """Protects the OSIS dotted format; rejects mis-split books, chapters, verses or votes."""
    result = parse_xrefs(
        [HEADER, "Ps.23.1\tJohn.10.11\t212\n", "Gen.1.1\tRom.1.19-Rom.1.20\t60\n"], min_votes=5
    )

    assert result.rows == [
        ParsedXref("Ps", 23, 1, "John", 10, 11, 10, 11, 212),
        ParsedXref("Gen", 1, 1, "Rom", 1, 19, 1, 20, 60),
    ]
    assert result.total == 2


def test_target_range_may_cross_chapters() -> None:
    """Protects ranges like Rom.1.32-Rom.2.1; rejects losing the end chapter."""
    [row] = parse_xrefs(["Prov.1.1\tRom.1.32-Rom.2.1\t9"], min_votes=5).rows

    assert (row.to_chapter, row.to_start_verse, row.to_end_chapter, row.to_end_verse) == (
        1,
        32,
        2,
        1,
    )


def test_low_votes_and_cross_book_targets_are_skipped_and_counted() -> None:
    """Protects the votes >= 5 filter; rejects loading weak or book-spanning links silently."""
    result = parse_xrefs(
        [
            "Ps.23.1\tJohn.10.11\t5",
            "Ps.23.1\tIsa.40.11\t4",
            "Ps.23.1\tIsa.40.11\t-3",
            "Ps.150.6\tPs.150.6-Prov.1.1\t12",
        ],
        min_votes=5,
    )

    assert [r.to_book for r in result.rows] == ["John"]
    assert (result.skipped_low_votes, result.skipped_cross_book, result.total) == (2, 1, 4)


def test_duplicate_pairs_keep_the_highest_votes() -> None:
    """Protects the unique source/target key; rejects a duplicate row or the lower vote count."""
    result = parse_xrefs(["Ps.23.1\tJohn.10.11\t7", "Ps.23.1\tJohn.10.11\t30"], min_votes=5)

    assert [r.votes for r in result.rows] == [30]
    assert result.duplicates == 1


@pytest.mark.parametrize(
    "line", ["Ps.23.1\tJohn.10.11", "Ps 23:1\tJohn.10.11\t7", "Ps.23.1\tJohn.10.11\tmany"]
)
def test_malformed_lines_raise_with_line_number(line: str) -> None:
    """Protects against silently loading garbage; rejects a parse that skips a bad line."""
    with pytest.raises(XrefParseError, match="line 2"):
        parse_xrefs([HEADER, line], min_votes=5)
