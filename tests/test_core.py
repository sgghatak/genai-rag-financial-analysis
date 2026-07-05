import pytest

from rag.config import PROJECT_ROOT, Settings
from rag.core import (
    IndexBuilder,
    RAGAssistant,
    resolve_question_for_prompt,
    search_web,
    should_use_web_search,
)


class TestIndexBuilder:
    """Tests for IndexBuilder."""

    def test_semantic_chunk_text_basic(self):
        """Test basic text chunking."""
        builder = IndexBuilder(chunk_size=100, overlap=10)
        text = "Section 1\n\nSection 2\n\nSection 3"
        chunks = builder.semantic_chunk_text(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_semantic_chunk_text_long_section(self):
        """Test chunking with long sections."""
        builder = IndexBuilder(chunk_size=50, overlap=10)
        text = "A" * 200
        chunks = builder.semantic_chunk_text(text)
        assert len(chunks) > 1

    def test_semantic_chunk_text_empty(self):
        """Test empty text returns no chunks."""
        builder = IndexBuilder(chunk_size=50, overlap=10)
        assert builder.semantic_chunk_text("   ") == []

    def test_invalid_overlap(self):
        """Test overlap must be smaller than chunk size."""
        with pytest.raises(ValueError):
            IndexBuilder(chunk_size=50, overlap=50)


class TestRAGAssistant:
    """Tests for RAGAssistant."""

    @pytest.fixture
    def mock_assistant(self, tmp_path):
        """Create mock RAG assistant with temporary files."""
        # Create mock index file
        import faiss
        import pickle
        import numpy as np

        index_path = tmp_path / "index.idx"
        chunks_path = tmp_path / "chunks.pkl"

        # Create simple FAISS index
        index = faiss.IndexFlatL2(384)
        faiss.write_index(index, str(index_path))

        # Create chunks
        chunks = ["Sample chunk 1", "Sample chunk 2", "Sample chunk 3"]
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        return RAGAssistant(str(index_path), str(chunks_path))

    def test_initialization(self, mock_assistant):
        """Test RAG assistant initialization."""
        assert mock_assistant.index is not None
        assert len(mock_assistant.chunks) > 0


def test_resolve_question_for_prompt_prefers_rewritten_question():
    """Rewritten follow-up questions should be used for answer generation."""
    original = "who is the CEO?"
    rewritten = "who is the CEO of Apple?"

    assert resolve_question_for_prompt(original, rewritten) == rewritten


def test_should_use_web_search_for_document_gap_phrasing():
    """Answers that say the provided context does not mention the detail should trigger a web fallback."""
    answer = "The information provided does not specify the name of the Chief Operating Officer (COO)."

    assert should_use_web_search(answer, context_included=True) is True


def test_search_web_includes_snippets(monkeypatch):
    """Search results should preserve snippets so the model can answer finance questions."""

    class FakeResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(*args, **kwargs):
        payload = {
            "organic": [
                {"title": "Fractal Analytics share price", "snippet": "Latest price: ₹1,250.00"}
            ]
        }
        return FakeResponse(payload)

    monkeypatch.setattr("rag.core.os.getenv", lambda name, default=None: "test-key" if name == "SERPER_API_KEY" else default)
    monkeypatch.setattr("rag.core.httpx.post", fake_post)

    results = search_web("Fractal Analytics stock price")
    assert "Latest price" in results
    assert "Fractal Analytics share price" in results


class TestSettings:
    """Tests for environment-backed settings."""

    def test_defaults(self, monkeypatch):
        """Test default settings values."""
        monkeypatch.delenv("RAG_INSECURE_SKIP_TLS_VERIFY", raising=False)
        settings = Settings.from_env()
        assert settings.embedding_model == "openai/text-embedding-3-small"
        assert settings.llm_model == "openai/gpt-4.1-mini"
        assert settings.data_dir == PROJECT_ROOT / "data"
        assert settings.insecure_skip_tls_verify is False

    def test_env_overrides(self, monkeypatch):
        """Test selected environment overrides."""
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
        monkeypatch.setenv("RAG_INSECURE_SKIP_TLS_VERIFY", "true")
        monkeypatch.setenv("SERPER_API_KEY", "demo-serper-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("RAG_PDF_PATH", "")
        settings = Settings.from_env()
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.insecure_skip_tls_verify is True
        assert settings.serper_api_key == "demo-serper-key" or settings.serper_api_key is not None
