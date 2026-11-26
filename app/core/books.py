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

    @property
    def reference_name(self) -> str:
        """Name used in citations: `Psalm 23:1`, not `Psalms 23:1`."""
        return "Psalm" if self.osis_code == "Ps" else self.name


_RAW: Final = [
    # (osis_code, name, vpl_code)
    ("Gen", "Genesis", "GEN"),
    ("Exod", "Exodus", "EXO"),
    ("Lev", "Leviticus", "LEV"),
    ("Num", "Numbers", "NUM"),
    ("Deut", "Deuteronomy", "DEU"),
    ("Josh", "Joshua", "JOS"),
    ("Judg", "Judges", "JDG"),
    ("Ruth", "Ruth", "RUT"),
    ("1Sam", "1 Samuel", "1SA"),
    ("2Sam", "2 Samuel", "2SA"),
    ("1Kgs", "1 Kings", "1KI"),
    ("2Kgs", "2 Kings", "2KI"),
    ("1Chr", "1 Chronicles", "1CH"),
    ("2Chr", "2 Chronicles", "2CH"),
    ("Ezra", "Ezra", "EZR"),
    ("Neh", "Nehemiah", "NEH"),
    ("Esth", "Esther", "EST"),
    ("Job", "Job", "JOB"),
    ("Ps", "Psalms", "PSA"),
    ("Prov", "Proverbs", "PRO"),
    ("Eccl", "Ecclesiastes", "ECC"),
    ("Song", "Song of Songs", "SOL"),
    ("Isa", "Isaiah", "ISA"),
    ("Jer", "Jeremiah", "JER"),
    ("Lam", "Lamentations", "LAM"),
    ("Ezek", "Ezekiel", "EZE"),
    ("Dan", "Daniel", "DAN"),
    ("Hos", "Hosea", "HOS"),
    ("Joel", "Joel", "JOE"),
    ("Amos", "Amos", "AMO"),
    ("Obad", "Obadiah", "OBA"),
    ("Jonah", "Jonah", "JON"),
    ("Mic", "Micah", "MIC"),
    ("Nah", "Nahum", "NAH"),
    ("Hab", "Habakkuk", "HAB"),
    ("Zeph", "Zephaniah", "ZEP"),
    ("Hag", "Haggai", "HAG"),
    ("Zech", "Zechariah", "ZEC"),
    ("Mal", "Malachi", "MAL"),
    ("Matt", "Matthew", "MAT"),
    ("Mark", "Mark", "MAR"),
    ("Luke", "Luke", "LUK"),
    ("John", "John", "JOH"),
    ("Acts", "Acts", "ACT"),
    ("Rom", "Romans", "ROM"),
    ("1Cor", "1 Corinthians", "1CO"),
    ("2Cor", "2 Corinthians", "2CO"),
    ("Gal", "Galatians", "GAL"),
    ("Eph", "Ephesians", "EPH"),
    ("Phil", "Philippians", "PHI"),
    ("Col", "Colossians", "COL"),
    ("1Thess", "1 Thessalonians", "1TH"),
    ("2Thess", "2 Thessalonians", "2TH"),
    ("1Tim", "1 Timothy", "1TI"),
    ("2Tim", "2 Timothy", "2TI"),
    ("Titus", "Titus", "TIT"),
    ("Phlm", "Philemon", "PHM"),
    ("Heb", "Hebrews", "HEB"),
    ("Jas", "James", "JAM"),
    ("1Pet", "1 Peter", "1PE"),
    ("2Pet", "2 Peter", "2PE"),
    ("1John", "1 John", "1JO"),
    ("2John", "2 John", "2JO"),
    ("3John", "3 John", "3JO"),
    ("Jude", "Jude", "JUD"),
    ("Rev", "Revelation", "REV"),
]

FIRST_NT_POSITION: Final = 40  # Matthew

BIBLE_BOOKS: Final[tuple[BookInfo, ...]] = tuple(
    BookInfo(
        position=position,
        osis_code=osis,
        name=name,
        testament="OT" if position < FIRST_NT_POSITION else "NT",
        vpl_code=vpl,
    )
    for position, (osis, name, vpl) in enumerate(_RAW, start=1)
)
