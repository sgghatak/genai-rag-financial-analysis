"""Core RAG functionality for retrieval and generation."""

import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


class RAGAssistant:
    """RAG Assistant for querying indexed documents."""

    def __init__(self, index_path: str, chunks_path: str, metadata_path: str = None):
        """
        Initialize RAG Assistant.

        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to pickled chunks file
            metadata_path: Optional path to chunk metadata
        """
        load_dotenv(override=True)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Query the index and generate an answer.

        Args:
            question: User question
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with retrieved chunks and answer
        """
        # Generate embedding for query
        query_embedding = self.client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=question
        ).data[0].embedding

        query_vector = np.array([query_embedding], dtype=np.float32)

        # Retrieve top-k chunks
        distances, indices = self.index.search(query_vector, top_k)

        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        # Generate answer
        response = self.client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer ONLY from the provided context. "
                        "If the answer is not present in the context, "
                        "say 'I could not find that information in the document.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=1000
        )

        return {
            "question": question,
            "retrieved_indices": indices[0].tolist(),
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "answer": response.choices[0].message.content,
        }


class IndexBuilder:
    """Builder for creating FAISS indexes from documents."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize IndexBuilder.

        Args:
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def semantic_chunk_text(self, text: str) -> list:
        """Semantically chunk text with overlap."""
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        chunks = []
        current_chunk = ""

        for section in sections:
            if len(section) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                start = 0
                while start < len(section):
                    end = start + self.chunk_size
                    chunk = section[start:end]
                    chunks.append(chunk)
                    start += (self.chunk_size - self.overlap)
                continue

            if len(current_chunk) + len(section) + 2 <= self.chunk_size:
                current_chunk += section + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                overlap_text = (
                    current_chunk[-self.overlap:]
                    if len(current_chunk) > self.overlap
                    else current_chunk
                )
                current_chunk = overlap_text + "\n\n" + section + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
