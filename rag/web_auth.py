import json
import os
from pathlib import Path
from typing import Mapping, Optional, Set


OIDC_SETTINGS = (
    ("redirect_uri", "OIDC_REDIRECT_URI"),
    ("cookie_secret", "OIDC_COOKIE_SECRET"),
    ("client_id", "OIDC_CLIENT_ID"),
    ("client_secret", "OIDC_CLIENT_SECRET"),
    ("server_metadata_url", "OIDC_METADATA_URL"),
)


def auth_is_required(value: Optional[str] = None) -> bool:
    if value is None:
        value = os.getenv("AUTH_REQUIRED", "false")
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def parse_allowed_user_emails(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {email.strip().casefold() for email in value.split(",") if email.strip()}


def is_user_allowed(email: Optional[str], allowed_emails: Set[str]) -> bool:
    return bool(email and email.strip().casefold() in allowed_emails)


def write_oidc_secrets_file(environment: Mapping[str, str], destination: Path) -> None:
    missing = [env_name for _, env_name in OIDC_SETTINGS if not environment.get(env_name, "").strip()]
    if missing:
        raise ValueError(f"Missing OIDC configuration: {', '.join(missing)}")

    settings = "\n".join(
        f"{setting_name} = {json.dumps(environment[env_name])}"
        for setting_name, env_name in OIDC_SETTINGS
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"[auth]\n{settings}\n", encoding="utf-8")
