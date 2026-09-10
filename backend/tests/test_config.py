import pytest
from pydantic import ValidationError
from slowapi.util import get_remote_address

from backend.config import (
    Settings,
    UsageGuardrailSettings,
    resolve_cors_allowed_origins,
    resolve_rate_limit_key_func,
)


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("R2_BUCKET_NAME", "jobfit-artifacts")
    monkeypatch.setenv("R2_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret-key")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://account-id.r2.cloudflarestorage.com")

    settings = Settings(_env_file=None)

    assert settings.r2_bucket_name == "jobfit-artifacts"
    assert settings.r2_account_id == "account-id"
    assert settings.r2_access_key_id == "access-key"
    assert settings.r2_secret_access_key == "secret-key"
    assert settings.r2_endpoint_url == "https://account-id.r2.cloudflarestorage.com"


def test_settings_requires_all_fields(monkeypatch):
    for var in (
        "R2_BUCKET_NAME",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_ENDPOINT_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_cors_allowed_origins_defaults_to_local_dev_frontend(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    assert resolve_cors_allowed_origins() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_cors_allowed_origins_reads_comma_separated_env_var(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://jobfit-app.leivadev.com, http://localhost:5173",
    )

    assert resolve_cors_allowed_origins() == [
        "https://jobfit-app.leivadev.com",
        "http://localhost:5173",
    ]


def test_rate_limit_key_func_defaults_to_remote_address(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_KEY_STRATEGY", raising=False)

    assert resolve_rate_limit_key_func() is get_remote_address


def test_rate_limit_key_func_rejects_unknown_strategy(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_KEY_STRATEGY", "forwarded_header")

    with pytest.raises(ValueError, match="forwarded_header"):
        resolve_rate_limit_key_func()


def test_usage_guardrail_defaults_to_disabled_with_azure_free_grant_thresholds(monkeypatch):
    monkeypatch.delenv("USAGE_GUARDRAIL_ENABLED", raising=False)

    settings = UsageGuardrailSettings(_env_file=None)

    assert settings.usage_guardrail_enabled is False
    assert settings.usage_guardrail_request_limit == 2_000_000
    assert settings.usage_guardrail_vcpu_seconds_limit == 180_000
    assert settings.usage_guardrail_gib_seconds_limit == 360_000


def test_usage_guardrail_enabled_via_env(monkeypatch):
    monkeypatch.setenv("USAGE_GUARDRAIL_ENABLED", "true")

    assert UsageGuardrailSettings().usage_guardrail_enabled is True
