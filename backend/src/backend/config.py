from collections.abc import Callable

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict
from slowapi.util import get_remote_address


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    r2_bucket_name: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_endpoint_url: str


class CorsSettings(BaseSettings):
    """Split from `Settings`: it has no required fields, so building the CORS
    whitelist never requires R2 config to be present (e.g. in tests that
    inject a fixture `AppState` and never touch R2)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


def resolve_cors_allowed_origins() -> list[str]:
    """Comma-separated allowlist from `CORS_ALLOWED_ORIGINS`."""
    raw = CorsSettings().cors_allowed_origins
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class RateLimitSettings(BaseSettings):
    """Split from `Settings`: it has no required fields, so building the
    rate-limit key function never requires R2 config (mirrors `CorsSettings`)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rate_limit_key_strategy: str = "remote_address"


RATE_LIMIT_KEY_FUNCS: dict[str, Callable[[Request], str]] = {
    "remote_address": get_remote_address,
}


def resolve_rate_limit_key_func() -> Callable[[Request], str]:
    """Client-IP key function for the `/recommend` rate limiter, swappable via
    `RATE_LIMIT_KEY_STRATEGY`. Only `remote_address` (the raw peer address) is
    implemented; a trusted forwarded-header strategy is reserved for
    Phase 6/7, once the deployment platform's proxy behavior is known."""
    strategy = RateLimitSettings().rate_limit_key_strategy
    try:
        return RATE_LIMIT_KEY_FUNCS[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown RATE_LIMIT_KEY_STRATEGY {strategy!r}; valid options: "
            f"{sorted(RATE_LIMIT_KEY_FUNCS)}"
        ) from None
