#!/usr/bin/env python3
"""Interactive query script for RAG Assistant."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag.core import RAGAssistant


def main():
    """Run interactive RAG assistant."""
    # Paths relative to data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    index_path = os.path.join(data_dir, "faiss_index.idx")
    chunks_path = os.path.join(data_dir, "chunks.pkl")

    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        print("Please run scripts/build_index.py first")
        sys.exit(1)

    assistant = RAGAssistant(index_path, chunks_path)

    print("=" * 60)
    print("RAG Assistant Ready")
    print("Type 'exit', 'quit', or 'stop' to end the session.")
    print("=" * 60)

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() in ["exit", "quit", "stop"]:
            print("\nGoodbye!")
            break

        try:
            result = assistant.query(question, top_k=3)
            print("\nRetrieved Chunk IDs:", result["retrieved_indices"])
            print("\nAnswer:")
            print(result["answer"])
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"\nError: {e}")
            print("-" * 80)


if __name__ == "__main__":
    main()
