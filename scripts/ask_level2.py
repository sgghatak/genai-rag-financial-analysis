#!/usr/bin/env python3
"""Advanced RAG with hybrid search and memory."""

import sys
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import Settings
from rag.core import (
    create_openai_client,
    resolve_question_for_prompt,
    search_web,
    should_use_web_search,
)

settings = Settings.from_env()
client = create_openai_client()


def load_resources(data_dir: Path) -> tuple:
    """Load index, chunks, and metadata."""
    index = faiss.read_index(str(data_dir / "faiss_index.idx"))

    with open(data_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(data_dir / "chunk_metadata.pkl", "rb") as f:
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

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
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
            max_tokens=settings.max_tokens_rewrite,
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return question
    except Exception as e:
        print(f"[WARNING] Query rewrite failed: {e}. Using original question.")
        return question


def main():
    """Run advanced RAG with memory and hybrid search."""
    data_dir = settings.data_dir
    memory_file = data_dir / "memory.pkl"
    
    print(f"Using model: {settings.llm_model}")

    print("Loading resources...")
    index, chunks, chunk_metadata, bm25 = load_resources(data_dir)

    # Load or create conversation history
    if memory_file.exists():
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

            final_question = resolve_question_for_prompt(question, rewritten_q)

            # Hybrid search: Vector + BM25
            query_embedding = client.embeddings.create(
                model=settings.embedding_model,
                input=rewritten_q,
            ).data[0].embedding

            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, vector_indices = index.search(query_vector, k=5)

            bm25_scores = bm25.get_scores(rewritten_q.lower().split())
            bm25_indices = np.argsort(bm25_scores)[::-1][:5]

            # Combine results
            combined_indices = list(set(vector_indices[0].tolist() + bm25_indices.tolist()))[:5]
            
            # Calculate relevance: use max score from top results, not average
            if len(combined_indices) > 0:
                top_vector_index = combined_indices[0]
                top_vector_sim = 1 / (1 + distances[0][0])  # Best vector match
                top_bm25_score = bm25_scores[bm25_indices[0]] / max(bm25_scores) if max(bm25_scores) > 0 else 0
                # Use whichever is higher
                max_relevance = max(top_vector_sim, top_bm25_score)
            else:
                max_relevance = 0
            
            retrieved_chunks = [chunks[i] for i in combined_indices]
            
            # Check if context is relevant
            relevance_threshold = 0.4  # Only use context if best match is relevant
            if max_relevance < relevance_threshold:
                context = ""
                context_included = False
                system_instruction = "You are a helpful assistant. Use your general knowledge to answer this question."
                print(f"[DEBUG] Max relevance {max_relevance:.3f} below threshold {relevance_threshold}, using general knowledge only")
            else:
                context = "\n\n".join(retrieved_chunks)
                context_included = True
                system_instruction = (
                    "You are a helpful assistant. Answer using the provided context. "
                    "Look carefully through the context for any numerical data, percentages, tables, "
                    "or comparisons that might answer the question, even if they're not explicitly stated. "
                    "If the context truly doesn't have the answer, state that clearly and do not speculate."
                )
                print(f"[DEBUG] Max relevance {max_relevance:.3f} above threshold {relevance_threshold}, using context")

            # Generate response
            if context_included:
                user_content = f"Context:\n{context}\n\nQuestion: {final_question}"
            else:
                user_content = f"Question: {final_question}"
            
            messages = [
                {
                    "role": "system",
                    "content": system_instruction,
                },
            ] + conversation_history + [
                {
                    "role": "user",
                    "content": user_content,
                }
            ]

            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                max_tokens=settings.max_tokens_response,
            )

            answer = response.choices[0].message.content
            
            should_retry_without_context = should_use_web_search(
                answer,
                context_included=context_included,
            )

            if should_retry_without_context:
                print(f"[DEBUG] LLM said answer not available, asking for permission to search the web...")
                try:
                    allow_web_search = input(
                        "I can search the web for a more up-to-date answer. Do you want me to proceed? (y/n): "
                    ).strip().lower()
                except KeyboardInterrupt:
                    allow_web_search = "n"

                if allow_web_search in {"y", "yes"}:
                    search_results = search_web(final_question)
                    if search_results:
                        answer = (
                            "I found the following web-search evidence for your question:\n\n"
                            f"{search_results}\n\n"
                            "Please verify the latest value from the cited source if you need a live market quote."
                        )
                    else:
                        answer = (
                            "I couldn't find that detail in the provided context or in the web search results."
                        )
                else:
                    answer = (
                        "I will not search the web. I can answer from the provided context only."
                    )

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
