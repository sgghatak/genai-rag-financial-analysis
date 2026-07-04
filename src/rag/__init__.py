"""
RAG (Retrieval-Augmented Generation) for Financial Analysis

A package for building and querying FAISS indexes with LLM-powered responses.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import Settings
from .core import IndexBuilder, RAGAssistant, create_openai_client

__all__ = ["RAGAssistant", "IndexBuilder", "Settings", "create_openai_client"]
