from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import web_app
from rag.web_auth import (
    auth_is_required,
    is_user_allowed,
    parse_allowed_user_emails,
    write_oidc_secrets_file,
)


def test_auth_is_required_parses_boolean_values():
    assert auth_is_required("true")
    assert auth_is_required(" YES ")
    assert not auth_is_required("false")
    assert not auth_is_required("")


def test_parse_allowed_user_emails_normalizes_addresses():
    assert parse_allowed_user_emails(" Alice@example.com, bob@example.com ") == {
        "alice@example.com",
        "bob@example.com",
    }
    assert parse_allowed_user_emails("") == set()


def test_is_user_allowed_ignores_email_case_and_whitespace():
    assert is_user_allowed(" Alice@Example.com ", {"alice@example.com"})
    assert not is_user_allowed(None, {"alice@example.com"})
    assert not is_user_allowed("bob@example.com", {"alice@example.com"})


def test_require_user_access_accepts_authorized_user(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("ALLOWED_USER_EMAILS", "alice@example.com")
    monkeypatch.setattr(
        web_app.st,
        "experimental_user",
        SimpleNamespace(is_logged_in=True, email="Alice@Example.com"),
    )

    web_app.require_user_access()


def test_main_checks_access_before_initializing_rag(monkeypatch):
    def stop_after_access_check():
        raise RuntimeError("access gate reached")

    monkeypatch.setattr(web_app, "require_user_access", stop_after_access_check)
    monkeypatch.setattr(web_app, "initialize_app", lambda: pytest.fail("RAG initialized before access check"))

    with pytest.raises(RuntimeError, match="access gate reached"):
        web_app.main()


def test_write_oidc_secrets_file_creates_auth_configuration(tmp_path: Path):
    environment = {
        "OIDC_REDIRECT_URI": "https://rag.example.com/oauth2callback",
        "OIDC_COOKIE_SECRET": "random-cookie-secret",
        "OIDC_CLIENT_ID": "client-id",
        "OIDC_CLIENT_SECRET": "client-secret",
        "OIDC_METADATA_URL": "https://identity.example.com/.well-known/openid-configuration",
    }
    destination = tmp_path / ".streamlit" / "secrets.toml"

    write_oidc_secrets_file(environment, destination)

    contents = destination.read_text(encoding="utf-8")
    assert contents.startswith("[auth]\n")
    assert 'redirect_uri = "https://rag.example.com/oauth2callback"' in contents
    assert 'client_secret = "client-secret"' in contents


def test_write_oidc_secrets_file_rejects_missing_settings(tmp_path: Path):
    with pytest.raises(ValueError, match="OIDC_CLIENT_SECRET"):
        write_oidc_secrets_file({}, tmp_path / ".streamlit" / "secrets.toml")
