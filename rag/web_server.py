import os
from pathlib import Path

from rag.web_auth import auth_is_required, write_oidc_secrets_file


def main() -> None:
    if auth_is_required(os.getenv("AUTH_REQUIRED", "true")):
        write_oidc_secrets_file(os.environ, Path(".streamlit/secrets.toml"))

    command = [
        "streamlit",
        "run",
        "rag/web_app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
