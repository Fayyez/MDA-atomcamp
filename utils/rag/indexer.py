"""
Builds and persists a FAISS vector index from patient documents.

Embeds the clinical_text of each document using OpenAI's
text-embedding-3-small model, then stores a cosine-similarity index
together with the full documents and metadata on disk.
"""

import json
import pickle
import time
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from typing import Sequence


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100


def _get_embeddings(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embeds texts in batches and returns an (N, D) float32 array."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)
    return np.array(all_embeddings, dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalizes rows so inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def build_index(
    documents: list[dict],
    api_key: str,
    index_dir: str | Path,
) -> None:
    """
    Embeds all clinical texts, builds a FAISS IndexFlatIP (cosine similarity),
    and saves the index + document store to index_dir.

    Files written:
      index_dir/faiss.index    – the FAISS index
      index_dir/documents.pkl  – list of full_document strings
      index_dir/metadata.json  – list of metadata dicts (mrno, diagnosis, icd_codes)
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)

    print(f"Embedding {len(documents)} patient records …")
    clinical_texts = [doc["clinical_text"] for doc in documents]
    embeddings = _get_embeddings(client, clinical_texts)
    embeddings = _normalize(embeddings)

    print("Building FAISS index …")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "faiss.index"))

    full_documents = [doc["full_document"] for doc in documents]
    with open(index_dir / "documents.pkl", "wb") as f:
        pickle.dump(full_documents, f)

    metadata = [doc["metadata"] for doc in documents]
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Index saved to {index_dir}  ({len(documents)} vectors, dim={EMBEDDING_DIM})")


def load_index(index_dir: str | Path) -> tuple[faiss.Index, list[str], list[dict]]:
    """
    Loads the FAISS index, full documents, and metadata from disk.

    Returns:
        (faiss_index, full_documents, metadata_list)
    """
    index_dir = Path(index_dir)

    index = faiss.read_index(str(index_dir / "faiss.index"))

    with open(index_dir / "documents.pkl", "rb") as f:
        full_documents: list[str] = pickle.load(f)

    with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata: list[dict] = json.load(f)

    return index, full_documents, metadata
