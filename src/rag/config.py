"""Configuration helpers for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openrouter_api_key: str | None = None
    serper_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"
    llm_model: str = "openai/gpt-4.1-mini"
    pdf_path: Path = PROJECT_ROOT / "documents" / "Fractal-Financial-Results-FY-2025-26.pdf"
    data_dir: Path = PROJECT_ROOT / "data"
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens_rewrite: int = 100
    max_tokens_response: int = 1000
    insecure_skip_tls_verify: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from .env and process environment."""
        load_dotenv(override=True)

        pdf_path = Path(
            os.getenv(
                "RAG_PDF_PATH",
                str(PROJECT_ROOT / "documents" / "Fractal-Financial-Results-FY-2025-26.pdf"),
            )
        )
        data_dir = Path(os.getenv("RAG_DATA_DIR", str(PROJECT_ROOT / "data")))

        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            serper_api_key=os.getenv("SERPER_API_KEY"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            llm_model=os.getenv("RAG_LLM_MODEL", "openai/gpt-4.1-mini"),
            pdf_path=pdf_path if pdf_path.is_absolute() else PROJECT_ROOT / pdf_path,
            data_dir=data_dir if data_dir.is_absolute() else PROJECT_ROOT / data_dir,
            tesseract_cmd=os.getenv(
                "TESSERACT_CMD",
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            ),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            max_tokens_rewrite=int(os.getenv("RAG_MAX_TOKENS_REWRITE", "100")),
            max_tokens_response=int(os.getenv("RAG_MAX_TOKENS_RESPONSE", "1000")),
            insecure_skip_tls_verify=_env_bool("RAG_INSECURE_SKIP_TLS_VERIFY", False),
        )
