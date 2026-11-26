from app.core.books import BIBLE_BOOKS
from app.core.corpus import CORPUS_BOOKS


def test_catalog_has_66_unique_books_in_order() -> None:
    """Protects the canon catalog; rejects missing, duplicated or misordered books."""
    assert len(BIBLE_BOOKS) == 66
    assert [b.position for b in BIBLE_BOOKS] == list(range(1, 67))
    assert len({b.osis_code for b in BIBLE_BOOKS}) == 66
    assert len({b.vpl_code for b in BIBLE_BOOKS}) == 66


def test_testaments_split_at_matthew() -> None:
    """Protects OT/NT labels; rejects Malachi as NT or Matthew as OT."""
    testaments = {b.osis_code: b.testament for b in BIBLE_BOOKS}

    assert (testaments["Mal"], testaments["Matt"]) == ("OT", "NT")
    assert sum(1 for b in BIBLE_BOOKS if b.testament == "OT") == 39


def test_corpus_books_are_the_five_wisdom_books_in_the_catalog() -> None:
    """Protects CORPUS_BOOKS; rejects a code that the catalog (and the database) cannot match."""
    by_osis = {b.osis_code: b for b in BIBLE_BOOKS}

    assert CORPUS_BOOKS == ("Job", "Ps", "Prov", "Eccl", "Song")
    assert [by_osis[code].vpl_code for code in CORPUS_BOOKS] == ["JOB", "PSA", "PRO", "ECC", "SOL"]


def test_psalms_are_cited_in_singular() -> None:
    """Protects citation style; rejects `Psalms 23:1` instead of `Psalm 23:1`."""
    by_osis = {b.osis_code: b for b in BIBLE_BOOKS}

    assert by_osis["Ps"].reference_name == "Psalm"
    assert by_osis["Prov"].reference_name == "Proverbs"
