"""Curated example questions for the public demo.

`uv run python -m app.scripts.warm_cache` answers them with the real LLM ahead of
time, so they keep working (from the answer cache) after the daily quota runs out.
Between them they cover all five corpus books.
"""

from typing import Final

EXAMPLE_QUESTIONS: Final[tuple[str, ...]] = (
    "What do the wisdom books say about caring for the poor?",
    "How does Proverbs describe the power of words?",
    "How do Job's friends explain his suffering?",
    "What does Ecclesiastes say about the value of work?",
    "How do the Psalms picture God as a shepherd?",
    "How does the Song of Songs describe love?",
)
