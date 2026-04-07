import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import wikipediaapi
import faiss
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL_ID

SAVE_DIR      = "saved_models"
CHUNKS_PATH   = os.path.join(SAVE_DIR, "rag_chunks.pkl")
FAISS_PATH    = os.path.join(SAVE_DIR, "rag_faiss.index")
CHUNK_SIZE    = 200
CHUNK_OVERLAP = 50

TOPICS = [
    "Pneumonia",
    "Glioma",
    "Meningioma",
    "Pituitary tumor",
    "Brain tumor",
    "Common cold",
    "Dengue fever",
    "Malaria",
    "Typhoid fever",
    "Allergy",
    "Hypertension",
    "Diabetes mellitus",
    "Chest X-ray",
    "Magnetic resonance imaging",
]


def fetch_wikipedia(topic: str) -> str:
    wiki   = wikipediaapi.Wikipedia(
        language="en",
        user_agent="MedAI-Diagnosis/1.0 (educational project)",
    )
    page = wiki.page(topic)
    if not page.exists():
        print(f"  NOT FOUND: {topic}")
        return ""
    print(f"  OK: {topic} ({len(page.text)} chars)")
    return page.text


def chunk_text(text: str, source: str) -> list:
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk_text  = " ".join(chunk_words)
        chunks.append({
            "text":   chunk_text,
            "source": source,
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build():
    print("=" * 50)
    print("Building RAG knowledge base...")
    print(f"Topics: {len(TOPICS)}")
    print("=" * 50)

    all_chunks = []
    for topic in TOPICS:
        print(f"Fetching: {topic}")
        text = fetch_wikipedia(topic)
        if text:
            chunks = chunk_text(text, source=topic)
            all_chunks.extend(chunks)
            print(f"  Chunks: {len(chunks)}")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("\nLoading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID)

    print("Embedding chunks...")
    texts      = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(SAVE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("=" * 50)
    print(f"Done. {len(all_chunks)} chunks indexed.")
    print(f"FAISS index: {FAISS_PATH}")
    print(f"Chunks:      {CHUNKS_PATH}")


if __name__ == "__main__":
    build()