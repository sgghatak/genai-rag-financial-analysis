#!/usr/bin/env python3
"""Build FAISS index from PDF documents."""

import sys
import io
import pickle
from pathlib import Path

import faiss
import fitz
import pytesseract
import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import Settings
from rag.core import IndexBuilder, create_openai_client


def main():
    """Build index from PDF documents."""
    settings = Settings.from_env()
    client = create_openai_client()
    data_dir = settings.data_dir
    pdf_path = settings.pdf_path
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    print("Opening PDF...")
    pdf_document = fitz.open(pdf_path)

    print("Extracting text and images...")
    full_text = ""

    for page_num, page in enumerate(pdf_document):
        print(f"Processing page {page_num + 1}/{len(pdf_document)}")

        # Extract text
        text = page.get_text()
        full_text += text + "\n"

        # Extract images and run OCR
        image_list = page.get_images(full=True)
        for img_index, image_info in enumerate(image_list):
            xref = image_info[0]
            pix = fitz.Pixmap(pdf_document, xref)

            try:
                # Export a PNG to preserve alpha if present, then convert to RGB for OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception:
                # Fallback: convert to RGB via Pixmap if PNG export fails
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

            try:
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    full_text += "\n[OCR from image]\n" + ocr_text + "\n"
            except Exception as e:
                print(f"OCR failed for image {img_index} on page {page_num + 1}: {e}")
            finally:
                pix = None

    pdf_document.close()

    # Chunk the text
    print("\nChunking text...")
    builder = IndexBuilder(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    chunks = builder.semantic_chunk_text(full_text)
    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")

        embedding = client.embeddings.create(
            model=settings.embedding_model,
            input=chunk
        ).data[0].embedding

        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Generated {len(embeddings)} embeddings")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    print("\nSaving index and chunks...")
    data_dir.mkdir(parents=True, exist_ok=True)

    index_path = data_dir / "faiss_index.idx"
    chunks_path = data_dir / "chunks.pkl"
    metadata_path = data_dir / "chunk_metadata.pkl"

    faiss.write_index(index, str(index_path))
    print(f"Index saved to {index_path}")

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {chunks_path}")

    # Save metadata
    metadata = {
        "chunk_count": len(chunks),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "source_pdf": str(pdf_path),
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")

    print("\nIndex building complete!")


if __name__ == "__main__":
    main()
