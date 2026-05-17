from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    gemma_model: str = os.getenv("GEMMA_MODEL", "gemma-4-e4b-it")
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,http://0.0.0.0:5173,null,*",
        ).split(",")
        if origin.strip()
    )
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    gemma_request_timeout_seconds: int = int(os.getenv("GEMMA_REQUEST_TIMEOUT_SECONDS", "240"))


settings = Settings()
