import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL_ID

CHUNKS_PATH = "saved_models/rag_chunks.pkl"
FAISS_PATH  = "saved_models/rag_faiss.index"
TOP_K       = 3

_embedder = None
_index    = None
_chunks   = None


def load_rag():
    global _embedder, _index, _chunks

    if _embedder is None:
        print("Loading RAG embedder...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_ID)

    if _index is None:
        print("Loading FAISS index...")
        _index = faiss.read_index(FAISS_PATH)

    if _chunks is None:
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)

    print(f"RAG ready. {len(_chunks)} chunks loaded.")


def retrieve(query: str, top_k: int = TOP_K) -> list:
    global _embedder, _index, _chunks

    if _embedder is None or _index is None or _chunks is None:
        load_rag()

    query_vec = _embedder.encode(
        [query],
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(_chunks):
            results.append({
                "text":   _chunks[idx]["text"],
                "source": _chunks[idx]["source"],
                "score":  round(float(score), 3),
            })

    return results


def format_context(results: list) -> str:
    if not results:
        return ""

    context = "MEDICAL KNOWLEDGE BASE:\n"
    context += "=" * 40 + "\n"
    for i, r in enumerate(results):
        context += f"[Source: {r['source']}]\n"
        context += f"{r['text'][:600]}\n"
        context += "-" * 40 + "\n"

    return context