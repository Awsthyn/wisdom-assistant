"""HTTP side of the demo guard: who the client is, and the 429 response."""

from fastapi import HTTPException, Request, status
from fastapi.sse import ServerSentEvent

from app.core.config import Settings
from app.services.guard import DemoGuard, StreamEvent, client_key

RATE_LIMIT_RESPONSES: dict[int | str, dict[str, str]] = {
    429: {
        "description": "Too many questions from this address in the current hour. "
        "`Retry-After` gives the seconds until the window resets. Cached answers "
        "do not count."
    }
}


async def enforce_rate_limit(request: Request, guard: DemoGuard, settings: Settings) -> None:
    """Count this request against its client's hourly budget, or raise 429."""
    client = client_key(
        request.headers.get("x-forwarded-for"),
        request.client.host if request.client else None,
        settings.trusted_proxy_hops,
    )
    decision = await guard.consume_rate_limit(client)
    if not decision.allowed:
        now = guard.clock()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            decision.message(now),
            headers={"Retry-After": str(decision.retry_after_seconds(now))},
        )


def to_sse(event: StreamEvent) -> ServerSentEvent:
    return ServerSentEvent(event=event.name, data=event.data)
