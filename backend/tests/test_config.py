import pytest
from pydantic import ValidationError

from backend.config import Settings


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
