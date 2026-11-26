"""Liveness endpoint."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health", summary="Liveness check")
async def health() -> HealthResponse:
    """Return `ok` when the process is up. Does not touch the database or the LLM."""
    return HealthResponse()
