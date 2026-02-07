from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup basic logging for the demo."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("cleanlab_demo")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="CLEANLAB_DEMO_", extra="ignore")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    log_level: str = "INFO"

    # Optional AI provider keys (used by pydantic-ai depending on configured model)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    def ensure_dirs(self) -> None:
        """Ensure data and artifacts directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
logger = _setup_logging(settings.log_level)

