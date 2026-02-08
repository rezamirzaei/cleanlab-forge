from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DemoConfig(BaseModel):
    """Base configuration shared by all tasks."""

    seed: int = Field(default=42, ge=0, le=1_000_000)


class DemoResult(BaseModel):
    """Base result shared by all tasks."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

