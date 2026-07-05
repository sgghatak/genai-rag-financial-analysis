"""Core RAG utilities for indexing text and querying a FAISS index."""

from __future__ import annotations

import os
import pickle
import re
import json
from html import unescape
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np
from openai import OpenAI

from rag.config import Settings


def resolve_question_for_prompt(question: str, rewritten_question: str | None = None) -> str:
    """Return the rewritten question when it adds useful context; otherwise keep the original."""
    if rewritten_question and rewritten_question.strip():
        rewritten = rewritten_question.strip()
        if rewritten.lower() != question.strip().lower():
            return rewritten
    return question.strip()


def should_use_web_search(answer: str, *, context_included: bool) -> bool:
    """Return True when the model reports that the local context did not contain the answer."""
    if not context_included:
        return False

    answer_lower = answer.lower().replace("**", "").replace("__", "").replace("_", "")
    explicit_gap_indicators = [
        "no information",
        "not available",
        "not found in the provided context",
        "not provided in the context",
        "the provided information does not mention",
        "the context does not mention",
        "does not specify",
        "does not mention",
        "does not contain",
        "does not include information",
        "does not include the latest",
        "latest available stock price",
        "current stock price",
        "chief executive officer",
        "ceo",
    ]
    return any(indicator in answer_lower for indicator in explicit_gap_indicators)


def search_web(query: str, *, max_results: int = 3) -> str:
    """Fetch a few web search results for a query using Serper when available, else return an empty string."""
    if not query.strip():
        return ""

    settings = Settings.from_env()
    api_key = settings.serper_api_key
    if api_key:
        try:
            response = httpx.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=15.0,
            )
            response.raise_for_status()
            payload = response.json()
            results: list[str] = []
            for item in payload.get("organic", [])[:max_results]:
                title = item.get("title", "").strip()
                snippet = item.get("snippet", "").strip()
                if title or snippet:
                    results.append(f"{title} - {snippet}".strip(" -"))
            if results:
                return "\n".join(f"- {result}" for result in results)
        except Exception:
            return ""

    try:
        response = httpx.get(
            "https://www.google.com/search",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        response.raise_for_status()
        return response.text[:200]
    except Exception:
        return ""


def create_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    insecure_skip_tls_verify: bool | None = None,
) -> OpenAI:
    """Create an OpenRouter-compatible client."""
    settings = Settings.from_env()
    skip_tls_verify = (
        settings.insecure_skip_tls_verify
        if insecure_skip_tls_verify is None
        else insecure_skip_tls_verify
    )
    client_kwargs: dict[str, Any] = {
        "base_url": base_url or settings.openrouter_base_url,
        "api_key": api_key or settings.openrouter_api_key,
    }

    if skip_tls_verify:
        client_kwargs["http_client"] = httpx.Client(verify=False)

    return OpenAI(**client_kwargs)


class IndexBuilder:
    """Build searchable text chunks from source document text."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def semantic_chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks while preserving paragraph boundaries."""
        normalized_text = text.strip()
        if not normalized_text:
            return []

        paragraphs = [paragraph.strip() for paragraph in normalized_text.split("\n\n")]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]

        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(self._split_long_text(paragraph))
                continue

            candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                chunks.append(current_chunk.strip())
                overlap_text = self._tail_overlap(current_chunk)
                current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_long_text(self, text: str) -> list[str]:
        """Split text that cannot fit in one chunk."""
        chunks = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break

        return chunks

    def _tail_overlap(self, text: str) -> str:
        """Return the trailing overlap window from an existing chunk."""
        if self.overlap == 0:
            return ""
        return text[-self.overlap :].strip()


class RAGAssistant:
    """Load a FAISS index and answer questions using retrieved document chunks."""

    def __init__(
        self,
        index_path: str | os.PathLike[str],
        chunks_path: str | os.PathLike[str],
        *,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        insecure_skip_tls_verify: bool | None = None,
    ) -> None:
        settings = Settings.from_env()

        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.embedding_model = embedding_model or settings.embedding_model
        self.llm_model = llm_model or settings.llm_model
        self.base_url = base_url or settings.openrouter_base_url
        self.api_key = api_key or settings.openrouter_api_key
        self.insecure_skip_tls_verify = (
            settings.insecure_skip_tls_verify
            if insecure_skip_tls_verify is None
            else insecure_skip_tls_verify
        )
        self.client: OpenAI | None = None

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        self.index = faiss.read_index(str(self.index_path))
        with self.chunks_path.open("rb") as file:
            self.chunks = pickle.load(file)

    def query(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Retrieve relevant chunks and generate an answer for a question."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        client = self._get_client()
        query_embedding = client.embeddings.create(
            model=self.embedding_model,
            input=question,
        ).data[0].embedding

        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        retrieved_indices = [
            int(index)
            for index in indices[0].tolist()
            if 0 <= int(index) < len(self.chunks)
        ]
        retrieved_chunks = [self.chunks[index] for index in retrieved_indices]
        context = "\n\n".join(retrieved_chunks)

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer using the provided context only.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=1000,
        )

        return {
            "answer": response.choices[0].message.content,
            "retrieved_indices": retrieved_indices,
            "retrieved_chunks": retrieved_chunks,
            "distances": distances[0][: len(retrieved_indices)].tolist(),
        }

    def _get_client(self) -> OpenAI:
        """Create the OpenAI-compatible client on first use."""
        if self.client is None:
            self.client = create_openai_client(
                base_url=self.base_url,
                api_key=self.api_key,
                insecure_skip_tls_verify=self.insecure_skip_tls_verify,
            )
        return self.client
