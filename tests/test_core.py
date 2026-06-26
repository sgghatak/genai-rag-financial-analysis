"""Tests for RAG core functionality."""

import os
import pytest
from unittest.mock import Mock, patch
from rag.core import RAGAssistant, IndexBuilder


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
