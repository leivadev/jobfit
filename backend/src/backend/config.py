from pydantic_settings import BaseSettings, SettingsConfigDict


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

    cors_allowed_origins: str = "http://localhost:5173"


def resolve_cors_allowed_origins() -> list[str]:
    """Comma-separated allowlist from `CORS_ALLOWED_ORIGINS`."""
    raw = CorsSettings().cors_allowed_origins
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
