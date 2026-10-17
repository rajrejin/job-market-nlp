"""API settings from environment variables (prefix JOBNLP_)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JOBNLP_", env_file=".env", extra="ignore")

    api_key: str = ""  # empty: no key required (local use)
    cors_origins: str = ""  # comma separated; empty: CORS off
    max_text_chars: int = 30_000
    max_upload_bytes: int = 2_000_000
    parse_timeout_s: float = 15.0
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct"
    enable_llm: bool = True


settings = Settings()
