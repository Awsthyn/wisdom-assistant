import pytest

from app.ingest.vpl import ParsedVerse, VplParseError, parse_vpl


def test_parses_code_chapter_verse_and_text() -> None:
    """Protects the VPL line format; rejects mis-split book, chapter, verse or text."""
    [verse] = parse_vpl(["PSA 23:1 A Psalm by David. The LORD is my shepherd;\n"])

    assert verse == ParsedVerse("PSA", 23, 1, "A Psalm by David. The LORD is my shepherd;")


def test_numeric_book_codes_and_bom_are_handled() -> None:
    """Protects codes like 1JO and a UTF-8 BOM; rejects treating them as malformed lines."""
    bom = "\N{ZERO WIDTH NO-BREAK SPACE}"
    [verse] = parse_vpl([f"{bom}1JO 4:8 He who does not love does not know God."])

    assert (verse.book_code, verse.chapter, verse.verse) == ("1JO", 4, 8)


def test_verses_without_text_are_skipped() -> None:
    """Protects against empty rows (e.g. LUK 17:36); rejects loading verses with no text."""
    verses = list(parse_vpl(["LUK 17:35 Two women...", "LUK 17:36 ", "LUK 17:37 They..."]))

    assert [v.verse for v in verses] == [35, 37]


def test_whitespace_is_collapsed_and_blank_lines_ignored() -> None:
    """Protects clean passage text; rejects double spaces, CRLF residue or blank-line errors."""
    verses = list(parse_vpl(["", "GEN 1:1  In the   beginning,\r\n", "   "]))

    assert verses == [ParsedVerse("GEN", 1, 1, "In the beginning,")]


def test_malformed_line_raises_with_line_number() -> None:
    """Protects against silently dropping garbage; rejects a parse that skips a bad line."""
    with pytest.raises(VplParseError, match="line 2"):
        list(parse_vpl(["GEN 1:1 In the beginning", "Genesis chapter one"]))
