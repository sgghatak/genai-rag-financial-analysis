#!/usr/bin/env python3
"""Build FAISS index from PDF documents."""

import sys
import os
import io
import pickle
import faiss
import fitz
import pytesseract
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag.core import IndexBuilder

load_dotenv(override=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Configure Tesseract for OCR (Windows)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

PDF_PATH = "data/documents/Fractal-Financial-Results-FY-2025-26.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    """Build index from PDF documents."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_path = os.path.join(os.path.dirname(__file__), "..", PDF_PATH)

    if not os.path.exists(pdf_path):
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
        image_list = page.get_images()
        for img_index, img_id in enumerate(image_list):
            xref = img_id
            pix = fitz.Pixmap(pdf_document, xref)

            # Check if image is CMYK or RGB
            if pix.n - pix.alpha < 4:
                img_data = pix.tobytes("ppm")
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix.tobytes("ppm")

            img = Image.open(io.BytesIO(img_data))

            try:
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    full_text += "\n[OCR from image]\n" + ocr_text + "\n"
            except Exception as e:
                print(f"OCR failed for image {img_index} on page {page_num + 1}: {e}")

    pdf_document.close()

    # Chunk the text
    print("\nChunking text...")
    builder = IndexBuilder(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks = builder.semantic_chunk_text(full_text)
    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")

        embedding = client.embeddings.create(
            model="openai/text-embedding-3-small",
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
    os.makedirs(data_dir, exist_ok=True)

    index_path = os.path.join(data_dir, "faiss_index.idx")
    chunks_path = os.path.join(data_dir, "chunks.pkl")
    metadata_path = os.path.join(data_dir, "chunk_metadata.pkl")

    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {chunks_path}")

    # Save metadata
    metadata = {
        "chunk_count": len(chunks),
        "embedding_model": "openai/text-embedding-3-small",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")

    print("\nIndex building complete!")


if __name__ == "__main__":
    main()
