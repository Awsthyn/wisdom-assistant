"""The semantic corpus: the five wisdom books.

This is the single definition of the corpus. It is a constant, not an
environment variable, because changing it means re-chunking and re-embedding.
"""

from typing import Final

# OSIS codes: Job, Psalms, Proverbs, Ecclesiastes, Song of Songs.
CORPUS_BOOKS: Final[tuple[str, ...]] = ("Job", "Ps", "Prov", "Eccl", "Song")
