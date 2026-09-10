from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from backend.api.usage_guardrail import UsageGuardrailMiddleware
from backend.config import UsageGuardrailSettings


def _build_middleware(settings: UsageGuardrailSettings) -> UsageGuardrailMiddleware:
    async def endpoint(request):
        return PlainTextResponse("ok")

    app = Starlette(
        routes=[Route("/ping", endpoint), Route("/health", endpoint)],
    )
    return UsageGuardrailMiddleware(app, settings=settings)


def test_disabled_by_default():
    assert UsageGuardrailSettings().usage_guardrail_enabled is False


def test_new_instance_starts_with_zero_counters():
    middleware = _build_middleware(UsageGuardrailSettings(usage_guardrail_enabled=True))

    assert middleware._counters.request_count == 0
    assert middleware._counters.vcpu_seconds == 0.0
    assert middleware._counters.gib_seconds == 0.0


def test_allows_requests_under_threshold():
    middleware = _build_middleware(UsageGuardrailSettings(usage_guardrail_enabled=True))

    with TestClient(middleware) as client:
        response = client.get("/ping")

    assert response.status_code == 200


def test_health_is_exempt_even_when_over_threshold():
    settings = UsageGuardrailSettings(
        usage_guardrail_enabled=True,
        usage_guardrail_request_limit=1,
        usage_guardrail_threshold_ratio=1.0,
    )
    middleware = _build_middleware(settings)
    middleware._counters.request_count = 1

    with TestClient(middleware) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_trips_at_request_count_threshold():
    settings = UsageGuardrailSettings(
        usage_guardrail_enabled=True,
        usage_guardrail_request_limit=2,
        usage_guardrail_threshold_ratio=1.0,
    )
    middleware = _build_middleware(settings)

    with TestClient(middleware) as client:
        client.get("/ping")
        client.get("/ping")
        response = client.get("/ping")

    assert response.status_code == 503
    assert "detail" in response.json()


def test_trips_at_vcpu_seconds_threshold():
    settings = UsageGuardrailSettings(
        usage_guardrail_enabled=True,
        usage_guardrail_vcpu_seconds_limit=1.0,
        usage_guardrail_threshold_ratio=1.0,
    )
    middleware = _build_middleware(settings)
    middleware._counters.vcpu_seconds = 1.0

    with TestClient(middleware) as client:
        response = client.get("/ping")

    assert response.status_code == 503


def test_trips_at_gib_seconds_threshold():
    settings = UsageGuardrailSettings(
        usage_guardrail_enabled=True,
        usage_guardrail_gib_seconds_limit=1.0,
        usage_guardrail_threshold_ratio=1.0,
    )
    middleware = _build_middleware(settings)
    middleware._counters.gib_seconds = 1.0

    with TestClient(middleware) as client:
        response = client.get("/ping")

    assert response.status_code == 503


def test_resets_on_calendar_month_rollover():
    settings = UsageGuardrailSettings(
        usage_guardrail_enabled=True,
        usage_guardrail_request_limit=1,
        usage_guardrail_threshold_ratio=1.0,
    )
    middleware = _build_middleware(settings)
    middleware._counters.request_count = 1
    middleware._counters.period = (2000, 1)

    with TestClient(middleware) as client:
        response = client.get("/ping")

    assert response.status_code == 200
