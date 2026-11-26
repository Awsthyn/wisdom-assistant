"""Catalog of the 66 books of the Protestant canon, in canonical order."""

from dataclasses import dataclass
from typing import Final, Literal

Testament = Literal["OT", "NT"]


@dataclass(frozen=True, slots=True)
class BookInfo:
    position: int
    osis_code: str
    name: str
    testament: Testament
    vpl_code: str  # BibleWorks-style code used by ebible.org's *_vpl.txt files
    usfm_code: str = ""  # Paratext/USFM code used in ebible.org page URLs (e.g. JHN, SNG)
    chapters: int = 0  # number of chapters in the WEB

    @property
    def reference_name(self) -> str:
        """Name used in citations: `Psalm 23:1`, not `Psalms 23:1`."""
        return "Psalm" if self.osis_code == "Ps" else self.name

    def chapter_url(self, chapter: int) -> str:
        """Stable public page for one chapter of the WEB on ebible.org.

        ebible.org pads chapter numbers to 3 digits for Psalms and to 2 elsewhere
        (`PSA023.htm`, `JHN03.htm`).
        """
        width = 3 if self.osis_code == "Ps" else 2
        return f"{EBIBLE_CHAPTER_BASE_URL}{self.usfm_code}{chapter:0{width}d}.htm"


EBIBLE_CHAPTER_BASE_URL: Final = "https://ebible.org/engwebp/"


_RAW: Final = [
    # (osis_code, name, vpl_code, usfm_code, chapters)
    ("Gen", "Genesis", "GEN", "GEN", 50),
    ("Exod", "Exodus", "EXO", "EXO", 40),
    ("Lev", "Leviticus", "LEV", "LEV", 27),
    ("Num", "Numbers", "NUM", "NUM", 36),
    ("Deut", "Deuteronomy", "DEU", "DEU", 34),
    ("Josh", "Joshua", "JOS", "JOS", 24),
    ("Judg", "Judges", "JDG", "JDG", 21),
    ("Ruth", "Ruth", "RUT", "RUT", 4),
    ("1Sam", "1 Samuel", "1SA", "1SA", 31),
    ("2Sam", "2 Samuel", "2SA", "2SA", 24),
    ("1Kgs", "1 Kings", "1KI", "1KI", 22),
    ("2Kgs", "2 Kings", "2KI", "2KI", 25),
    ("1Chr", "1 Chronicles", "1CH", "1CH", 29),
    ("2Chr", "2 Chronicles", "2CH", "2CH", 36),
    ("Ezra", "Ezra", "EZR", "EZR", 10),
    ("Neh", "Nehemiah", "NEH", "NEH", 13),
    ("Esth", "Esther", "EST", "EST", 10),
    ("Job", "Job", "JOB", "JOB", 42),
    ("Ps", "Psalms", "PSA", "PSA", 150),
    ("Prov", "Proverbs", "PRO", "PRO", 31),
    ("Eccl", "Ecclesiastes", "ECC", "ECC", 12),
    ("Song", "Song of Songs", "SOL", "SNG", 8),
    ("Isa", "Isaiah", "ISA", "ISA", 66),
    ("Jer", "Jeremiah", "JER", "JER", 52),
    ("Lam", "Lamentations", "LAM", "LAM", 5),
    ("Ezek", "Ezekiel", "EZE", "EZK", 48),
    ("Dan", "Daniel", "DAN", "DAN", 12),
    ("Hos", "Hosea", "HOS", "HOS", 14),
    ("Joel", "Joel", "JOE", "JOL", 3),
    ("Amos", "Amos", "AMO", "AMO", 9),
    ("Obad", "Obadiah", "OBA", "OBA", 1),
    ("Jonah", "Jonah", "JON", "JON", 4),
    ("Mic", "Micah", "MIC", "MIC", 7),
    ("Nah", "Nahum", "NAH", "NAM", 3),
    ("Hab", "Habakkuk", "HAB", "HAB", 3),
    ("Zeph", "Zephaniah", "ZEP", "ZEP", 3),
    ("Hag", "Haggai", "HAG", "HAG", 2),
    ("Zech", "Zechariah", "ZEC", "ZEC", 14),
    ("Mal", "Malachi", "MAL", "MAL", 4),
    ("Matt", "Matthew", "MAT", "MAT", 28),
    ("Mark", "Mark", "MAR", "MRK", 16),
    ("Luke", "Luke", "LUK", "LUK", 24),
    ("John", "John", "JOH", "JHN", 21),
    ("Acts", "Acts", "ACT", "ACT", 28),
    ("Rom", "Romans", "ROM", "ROM", 16),
    ("1Cor", "1 Corinthians", "1CO", "1CO", 16),
    ("2Cor", "2 Corinthians", "2CO", "2CO", 13),
    ("Gal", "Galatians", "GAL", "GAL", 6),
    ("Eph", "Ephesians", "EPH", "EPH", 6),
    ("Phil", "Philippians", "PHI", "PHP", 4),
    ("Col", "Colossians", "COL", "COL", 4),
    ("1Thess", "1 Thessalonians", "1TH", "1TH", 5),
    ("2Thess", "2 Thessalonians", "2TH", "2TH", 3),
    ("1Tim", "1 Timothy", "1TI", "1TI", 6),
    ("2Tim", "2 Timothy", "2TI", "2TI", 4),
    ("Titus", "Titus", "TIT", "TIT", 3),
    ("Phlm", "Philemon", "PHM", "PHM", 1),
    ("Heb", "Hebrews", "HEB", "HEB", 13),
    ("Jas", "James", "JAM", "JAS", 5),
    ("1Pet", "1 Peter", "1PE", "1PE", 5),
    ("2Pet", "2 Peter", "2PE", "2PE", 3),
    ("1John", "1 John", "1JO", "1JN", 5),
    ("2John", "2 John", "2JO", "2JN", 1),
    ("3John", "3 John", "3JO", "3JN", 1),
    ("Jude", "Jude", "JUD", "JUD", 1),
    ("Rev", "Revelation", "REV", "REV", 22),
]

FIRST_NT_POSITION: Final = 40  # Matthew

BIBLE_BOOKS: Final[tuple[BookInfo, ...]] = tuple(
    BookInfo(
        position=position,
        osis_code=osis,
        name=name,
        testament="OT" if position < FIRST_NT_POSITION else "NT",
        vpl_code=vpl,
        usfm_code=usfm,
        chapters=chapters,
    )
    for position, (osis, name, vpl, usfm, chapters) in enumerate(_RAW, start=1)
)
