#!/usr/bin/env python3
"""Advanced RAG with hybrid search and memory."""

import sys
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

load_dotenv(override=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def load_resources(data_dir: str) -> tuple:
    """Load index, chunks, and metadata."""
    index = faiss.read_index(os.path.join(data_dir, "faiss_index.idx"))

    with open(os.path.join(data_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    with open(os.path.join(data_dir, "chunk_metadata.pkl"), "rb") as f:
        chunk_metadata = pickle.load(f)

    # Build BM25 index
    print("Building BM25 index...")
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return index, chunks, chunk_metadata, bm25


def rewrite_query(question: str, history_text: str) -> str:
    """Rewrite query based on conversation history."""
    if not history_text.strip():
        return question

    response = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Rewrite the user's question to be more specific and clear given the conversation context.",
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nNew question: {question}\n\nRewritten question:",
            },
        ],
        max_tokens=100,
    )

    return response.choices[0].message.content.strip()


def main():
    """Run advanced RAG with memory and hybrid search."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    memory_file = os.path.join(data_dir, "memory.pkl")

    print("Loading resources...")
    index, chunks, chunk_metadata, bm25 = load_resources(data_dir)

    # Load or create conversation history
    if os.path.exists(memory_file):
        with open(memory_file, "rb") as f:
            conversation_history = pickle.load(f)
        print(f"Loaded {len(conversation_history)} previous messages.")
    else:
        conversation_history = []

    print("\n" + "=" * 60)
    print("RAG Assistant with Memory (Level 2)")
    print("Type 'exit', 'quit', or 'stop' to end.")
    print("=" * 60)

    try:
        while True:
            question = input("\nEnter your question: ").strip()

            if question.lower() in ["exit", "quit", "stop"]:
                print("\nSaving conversation history...")
                with open(memory_file, "wb") as f:
                    pickle.dump(conversation_history, f)
                print("Goodbye!")
                break

            # Rewrite query if there's history
            history_text = "\n".join(
                [f"{msg['role']}: {msg['content'][:100]}..." for msg in conversation_history[-4:]]
            )
            rewritten_q = rewrite_query(question, history_text)

            # Hybrid search: Vector + BM25
            query_embedding = client.embeddings.create(
                model="openai/text-embedding-3-small",
                input=rewritten_q,
            ).data[0].embedding

            query_vector = np.array([query_embedding], dtype=np.float32)
            _, vector_indices = index.search(query_vector, k=5)

            bm25_scores = bm25.get_scores(rewritten_q.lower().split())
            bm25_indices = np.argsort(bm25_scores)[::-1][:5]

            # Combine results
            combined_indices = list(set(vector_indices[0].tolist() + bm25_indices.tolist()))[:5]
            retrieved_chunks = [chunks[i] for i in combined_indices]
            context = "\n\n".join(retrieved_chunks)

            # Generate response
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer using the provided context only.",
                },
            ] + conversation_history + [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ]

            response = client.chat.completions.create(
                model="openai/gpt-4.1-mini",
                messages=messages,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content

            # Update history
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

            print("\nAnswer:")
            print(answer)
            print("\n" + "-" * 80)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving conversation history...")
        with open(memory_file, "wb") as f:
            pickle.dump(conversation_history, f)
        print("Goodbye!")


if __name__ == "__main__":
    main()
