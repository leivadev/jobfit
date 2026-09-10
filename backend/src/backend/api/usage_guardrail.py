import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.config import UsageGuardrailSettings


def _current_period() -> tuple[int, int]:
    now = datetime.now(UTC)
    return (now.year, now.month)


@dataclass
class _UsageCounters:
    period: tuple[int, int]
    request_count: int = 0
    vcpu_seconds: float = 0.0
    gib_seconds: float = 0.0


class UsageGuardrailMiddleware(BaseHTTPMiddleware):
    """Best-effort in-memory guardrail against Azure Container Apps' free
    grant. Opt-in (see `UsageGuardrailSettings.usage_guardrail_enabled`);
    the authoritative backstop is the Azure Monitor budget alert provisioned
    in Phase 6, not this middleware — counters are per-instance and reset on
    cold start, so they undercount if the app ever scales beyond 1 replica.
    """

    # Exempt from both the check and the count: Azure's startup/liveness
    # probes hit this path (see docs/design/phase-7a-deploy-backend.md), and
    # a tripped guardrail returning 503 here would fail the liveness probe,
    # causing Azure to restart the container — which resets the very
    # counters the guardrail relies on.
    _EXEMPT_PATHS = frozenset({"/health"})

    def __init__(self, app, settings: UsageGuardrailSettings) -> None:
        super().__init__(app)
        self._settings = settings
        self._counters = _UsageCounters(period=_current_period())

    def _reset_if_new_period(self) -> None:
        current = _current_period()
        if current != self._counters.period:
            self._counters = _UsageCounters(period=current)

    def _over_threshold(self) -> bool:
        settings = self._settings
        counters = self._counters
        ratio = settings.usage_guardrail_threshold_ratio

        def _nearing(used: float, limit: float) -> bool:
            return used >= limit * ratio

        return (
            _nearing(counters.request_count, settings.usage_guardrail_request_limit)
            or _nearing(counters.vcpu_seconds, settings.usage_guardrail_vcpu_seconds_limit)
            or _nearing(counters.gib_seconds, settings.usage_guardrail_gib_seconds_limit)
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        self._reset_if_new_period()

        if self._over_threshold():
            return JSONResponse(
                {"detail": "Service temporarily unavailable: nearing monthly usage grant."},
                status_code=503,
            )

        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        self._counters.request_count += 1
        self._counters.vcpu_seconds += duration * self._settings.usage_guardrail_vcpu_allocation
        self._counters.gib_seconds += duration * self._settings.usage_guardrail_gib_allocation

        return response
