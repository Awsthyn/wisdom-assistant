"""The three Bible tools: lookup_reference, search_passages and cross_references."""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.core.corpus import CORPUS_BOOKS
from app.core.references import InvalidReferenceError, parse_reference
from app.services import search as search_service
from app.services import xrefs as xrefs_service
from app.services.search import (
    NotAReferenceError,
    ReferenceNotFoundError,
    SearchHit,
    UnknownCorpusBookError,
)
from app.tools.base import Tool, ToolContext, ToolError, ToolOutput

# Keeps a single tool result (and the prompt) small: about one long Psalm.
MAX_LOOKUP_VERSES: Final = 60

_REFERENCE_ERRORS = (
    NotAReferenceError,
    InvalidReferenceError,
    ReferenceNotFoundError,
    UnknownCorpusBookError,
)


def _verse_count(hits: list[SearchHit]) -> int:
    return sum(h.end_verse - h.start_verse + 1 for h in hits)


def _passage(hit: SearchHit) -> dict[str, object]:
    return {"reference": hit.reference, "text": hit.text}


def _short_list(references: list[str], limit: int = 3) -> str:
    shown = ", ".join(references[:limit])
    return shown + (f" (+{len(references) - limit} more)" if len(references) > limit else "")


# --- lookup_reference ---------------------------------------------------------


class LookupReferenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str = Field(
        min_length=2,
        max_length=60,
        description="A Bible reference in English, e.g. 'Psalm 23:1-3', 'John 10:11', 'Prov 3'.",
    )


async def _lookup_reference(context: ToolContext, args: LookupReferenceArgs) -> ToolOutput:
    try:
        hits = await search_service.lookup_reference(context.session, args.reference)
    except _REFERENCE_ERRORS as exc:
        raise ToolError(str(exc)) from exc
    if _verse_count(hits) > MAX_LOOKUP_VERSES:
        raise ToolError(
            f"'{args.reference}' has {_verse_count(hits)} verses; ask for at most "
            f"{MAX_LOOKUP_VERSES} at a time."
        )
    reference = parse_reference(args.reference)
    label = reference.label if reference else hits[0].reference
    in_corpus = hits[0].in_corpus
    return ToolOutput(
        data={"reference": label, "in_corpus": in_corpus, "passages": [_passage(h) for h in hits]},
        summary=(
            f"{label}: {_verse_count(hits)} verse(s), "
            f"{'in the corpus' if in_corpus else 'outside the corpus'}"
        ),
        passages=tuple(hits),
    )


LOOKUP_REFERENCE: Final = Tool(
    name="lookup_reference",
    description=(
        "Return the exact World English Bible text of a reference from any of the 66 books "
        "(e.g. 'Romans 8:28-30'), with in_corpus telling whether it belongs to the five "
        "wisdom books."
    ),
    args_model=LookupReferenceArgs,
    handler=_lookup_reference,
)


# --- search_passages ------------------------------------------------------------


class SearchPassagesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3, max_length=300, description="What to look for, in natural language."
    )
    k: int = Field(default=5, ge=1, le=10, description="Number of passages to return.")
    book: str | None = Field(
        default=None,
        description=f"Optional: only search one wisdom book ({', '.join(CORPUS_BOOKS)}).",
    )


async def _search_passages(context: ToolContext, args: SearchPassagesArgs) -> ToolOutput:
    try:
        corpus_book = (
            search_service.resolve_corpus_book(args.book) if args.book is not None else None
        )
    except UnknownCorpusBookError as exc:
        raise ToolError(str(exc)) from exc
    hits = await search_service.semantic_hits(
        context.session, context.embedder, args.query, args.k, corpus_book
    )
    return ToolOutput(
        data={
            "query": args.query,
            "results": [{**_passage(h), "score": h.score} for h in hits],
        },
        summary=f"{len(hits)} passage(s): {_short_list([h.reference for h in hits])}",
        passages=tuple(hits),
    )


SEARCH_PASSAGES: Final = Tool(
    name="search_passages",
    description=(
        "Semantic search over the five wisdom books (Job, Psalms, Proverbs, Ecclesiastes, "
        "Song of Songs). Returns the most relevant 5-verse passages with their text."
    ),
    args_model=SearchPassagesArgs,
    handler=_search_passages,
)


# --- cross_references -----------------------------------------------------------


class CrossReferencesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str = Field(
        min_length=2,
        max_length=60,
        description="A verse or verse range within one chapter, e.g. 'Psalm 23:1'.",
    )
    limit: int = Field(default=10, ge=1, le=20, description="Maximum cross references.")


async def _cross_references(context: ToolContext, args: CrossReferencesArgs) -> ToolOutput:
    try:
        result = await xrefs_service.cross_references(context.session, args.reference, args.limit)
    except _REFERENCE_ERRORS as exc:
        raise ToolError(str(exc)) from exc
    outside = sum(1 for t in result.targets if not t.in_corpus)
    return ToolOutput(
        data={
            "source": result.source,
            "cross_references": [
                {
                    "reference": t.reference,
                    "votes": t.votes,
                    "in_corpus": t.in_corpus,
                    "text": " ".join(p.text for p in t.passages),
                }
                for t in result.targets
            ],
        },
        summary=(
            f"{len(result.targets)} cross reference(s) for {result.source}"
            + (f" ({outside} outside the corpus)" if outside else "")
            + (f": {_short_list([t.reference for t in result.targets])}" if result.targets else "")
        ),
        passages=tuple(p for t in result.targets for p in t.passages),
    )


CROSS_REFERENCES: Final = Tool(
    name="cross_references",
    description=(
        "Cross references (OpenBible.info) for a verse: other passages anywhere in the Bible "
        "that share its words or themes, ordered by votes, each with its text and in_corpus."
    ),
    args_model=CrossReferencesArgs,
    handler=_cross_references,
)
