"""Liveness endpoint, plus the LLM quota status the UI shows before anyone asks."""

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_guard
from app.services.guard import DemoGuard

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_quota: Literal["available", "exhausted"] | None = None
    llm_quota_resets_at: datetime | None = None


@router.get(
    "/health", summary="Liveness check and LLM quota status", response_model_exclude_none=True
)
async def health(guard: Annotated[DemoGuard, Depends(get_guard)]) -> HealthResponse:
    """`status` is `ok` whenever the process is up; the LLM is never called.

    `llm_quota` is `exhausted` from the moment the provider reports its daily quota
    used up until the next daily reset (`llm_quota_resets_at`), and `available`
    otherwise. It is omitted when the database is not configured or cannot be read,
    so a database problem never fails the liveness check. The status is read at
    most once a minute."""
    quota = await guard.quota_status()
    if quota is None:
        return HealthResponse()
    return HealthResponse(llm_quota=quota.label, llm_quota_resets_at=quota.resets_at)
