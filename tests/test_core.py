import pytest

from rag.config import PROJECT_ROOT, Settings
from rag.core import IndexBuilder, RAGAssistant


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
        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
        monkeypatch.setenv("RAG_INSECURE_SKIP_TLS_VERIFY", "true")
        settings = Settings.from_env()
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.insecure_skip_tls_verify is True
