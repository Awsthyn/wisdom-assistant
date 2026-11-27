"""`GET /api/v1/examples`: curated questions that work even when the quota is used up."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.examples import EXAMPLE_QUESTIONS

router = APIRouter(prefix="/api/v1", tags=["examples"])


class ExamplesResponse(BaseModel):
    questions: list[str]


@router.get("/examples", summary="Curated example questions")
async def examples() -> ExamplesResponse:
    """Six questions covering the five wisdom books. Their chat answers are cached ahead
    of time (`app.scripts.warm_cache`), so asking one costs no LLM call."""
    return ExamplesResponse(questions=list(EXAMPLE_QUESTIONS))
