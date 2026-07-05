# RAG for Financial Analysis

A Retrieval-Augmented Generation (RAG) system for analyzing financial documents using FAISS vector indexing and OpenAI embeddings.

## Project Structure

```
genai-rag-financial-analysis/
├── src/
│   └── rag/
│       ├── __init__.py          # Package initialization
│       └── core.py              # Core RAG classes (RAGAssistant, IndexBuilder)
├── scripts/
│   ├── ask.py                   # Interactive query script
│   ├── ask_level2.py            # Advanced query with memory and hybrid search
│   └── build_index.py           # Build FAISS index from PDF documents
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   └── test_core.py             # Unit tests for core module
├── data/                        # Data directory for indexes and chunks
│   ├── documents/               # PDF documents to process
│   ├── faiss_index.idx          # FAISS vector index
│   ├── chunks.pkl               # Extracted text chunks
│   ├── chunk_metadata.pkl       # Chunk metadata
│   └── memory.pkl               # Conversation memory
├── docs/                        # Documentation
├── pyproject.toml              # Project configuration
├── README.md                   # This file
└── .gitignore                  # Git ignore file
```

## Features

- **Vector Search**: Uses FAISS for fast similarity search on embeddings
- **Hybrid Search**: Combines vector search with BM25 keyword search (Level 2)
- **OCR Support**: Extracts text from PDF images using Tesseract
- **Conversation Memory**: Stores conversation history for context-aware responses
- **Query Rewriting**: Rewrites queries based on conversation history for better context

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR (for Windows): [Download](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sgghatak/genai-rag-financial-analysis
   cd genai-rag-financial-analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Create a `.env` file with your API key:
   ```env
   OPENROUTER_API_KEY=your_key_here
   SERPER_API_KEY=your_serper_key_here
   ```

   You can copy `.env.example` for the full list of optional settings.

## Usage

### 1. Build Index from PDF

Place your PDF documents in `data/documents/` and run:

```bash
python scripts/build_index.py
```

This will:
- Extract text and images from PDFs
- Run OCR on images
- Create semantic chunks with overlap
- Generate embeddings using OpenAI
- Save FAISS index and chunks

### 2. Query the Index (Basic)

```bash
python scripts/ask.py
```

Interactive query interface:
- Enter questions about your documents
- Retrieves top 3 relevant chunks
- Generates answers using GPT-4

### 3. Query with Memory (Advanced)

```bash
python scripts/ask_level2.py
```

Enhanced interface with:
- Hybrid search (vector + BM25)
- Conversation memory
- Query rewriting for context awareness
- Persistent memory across sessions

## API Usage

Use the RAG classes directly in your code:

```python
from rag.core import RAGAssistant, IndexBuilder

# Initialize assistant
assistant = RAGAssistant(
    index_path="data/faiss_index.idx",
    chunks_path="data/chunks.pkl"
)

# Query
result = assistant.query("What are the financial results?")
print(result["answer"])

# Or use IndexBuilder to create new indexes
builder = IndexBuilder(chunk_size=1000, overlap=200)
chunks = builder.semantic_chunk_text(text)
```

## Development

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src/rag
```

### Code Quality

Format code:
```bash
black src/ scripts/ tests/
```

Lint:
```bash
ruff check src/ scripts/ tests/
```

Type checking:
```bash
mypy src/rag/
```

## Configuration

Runtime settings are loaded from environment variables by `rag.config.Settings`.

Common settings:
- `OPENROUTER_API_KEY`: OpenRouter API key
- `SERPER_API_KEY`: Serper API key for web-search fallback
- `OPENROUTER_BASE_URL`: OpenAI-compatible API base URL
- `RAG_EMBEDDING_MODEL`: Embedding model, default `openai/text-embedding-3-small`
- `RAG_LLM_MODEL`: Chat model, default `openai/gpt-4.1-mini`
- `RAG_PDF_PATH`: PDF to index
- `RAG_DATA_DIR`: Directory for FAISS index, chunks, metadata, and memory
- `RAG_CHUNK_SIZE`: Text chunk size, default `1000`
- `RAG_CHUNK_OVERLAP`: Overlap between chunks, default `200`
- `TESSERACT_CMD`: Tesseract executable path
- `RAG_INSECURE_SKIP_TLS_VERIFY`: Local OpenSSL workaround; keep `false` in production

## Environment Variables

Create `.env` file in project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
SERPER_API_KEY=your_serper_api_key
```

For this local Windows environment, if OpenSSL fails with `OPENSSL_Applink`, add:

```env
RAG_INSECURE_SKIP_TLS_VERIFY=true
```

This disables TLS certificate verification and should not be used in production.

## Troubleshooting

### Tesseract not found (Windows)
Install from: https://github.com/UB-Mannheim/tesseract/wiki
Update the path in `.env`:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### FAISS index not found
Run `scripts/build_index.py` first to create the index from PDF documents.

### Memory errors with large PDFs
Reduce `chunk_size` or `batch_size` in the indexing script.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Authors

- [Your Name](https://github.com/sgghatak)

## Support

For issues and questions, please open an issue on GitHub.
