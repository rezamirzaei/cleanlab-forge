from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup basic logging for the demo."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("cleanlab_demo")


def _find_project_root() -> Path:
    """
    Best-effort project root detection.

    Prefers the nearest parent of the current working directory that contains a
    `pyproject.toml` (works well for notebooks run from `notebooks/`). Falls
    back to the directory that contains `src/` when executed from the repo.
    """
    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if (p / "pyproject.toml").exists():
            return p

    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "pyproject.toml").exists():
            return p

    return cwd


PROJECT_ROOT = _find_project_root()


def _load_dotenv_keys(dotenv_path: Path) -> None:
    """Load common AI provider keys from a .env file into process env vars."""
    if not dotenv_path.exists():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    parsed: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[key] = value

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if parsed.get(key) and not os.getenv(key):
            os.environ[key] = parsed[key]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CLEANLAB_DEMO_",
        extra="ignore",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    log_level: str = "INFO"

    # Optional AI provider keys (used by pydantic-ai depending on configured model)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> Settings:
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = (PROJECT_ROOT / self.artifacts_dir).resolve()
        return self

    def ensure_dirs(self) -> None:
        """Ensure data and artifacts directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
_load_dotenv_keys(PROJECT_ROOT / ".env")

# If keys are provided via CLEANLAB_DEMO_* settings (or .env), propagate to the
# standard env vars expected by pydantic-ai providers.
if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
if settings.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

logger = _setup_logging(settings.log_level)
