"""
General-purpose RAG (Retrieval-Augmented Generation) retriever.

Indexes any collection of text documents with OpenAI embeddings and
performs cosine-similarity search via FAISS. Supports incremental
document additions, persistent storage, and arbitrary metadata.

Typical usage
-------------
    from utils.rag_retriever import GeneralRAGRetriever

    # --- Build once ---
    retriever = GeneralRAGRetriever()
    retriever.add_documents(
        documents=["Patient A: 60F, cough, tree-in-bud on HRCT …", ...],
        metadata=[{"id": 1, "label": "TB"}, ...],
    )
    retriever.save("data/my_index")

    # --- Use later ---
    retriever = GeneralRAGRetriever.load("data/my_index")
    results = retriever.retrieve("55M, haemoptysis, cavitary lesion", k=5)
    for r in results:
        print(r["rank"], r["score"], r["metadata"])
        print(r["document"])
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
_EMBED_BATCH_SIZE = 100
_EMBED_BATCH_SLEEP = 0.3   # seconds between batches to respect rate limits

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_client(api_key: str | None) -> OpenAI:
    key = api_key or os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not provided. Pass api_key= or set OPEN_AI_API_KEY "
            "in your .env file."
        )
    return OpenAI(api_key=key)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so that inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_texts(
    texts: list[str],
    client: OpenAI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of strings in batches.

    Returns an (N, EMBEDDING_DIM) float32 array of L2-normalised vectors.
    """
    all_vecs: list[list[float]] = []
    total_batches = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(texts), _EMBED_BATCH_SIZE)):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        if show_progress:
            print(
                f"  Embedding batch {batch_idx + 1}/{total_batches} "
                f"({start}–{start + len(batch) - 1}) …"
            )
        response = client.embeddings.create(model=model, input=batch)
        all_vecs.extend(item.embedding for item in response.data)
        if start + _EMBED_BATCH_SIZE < len(texts):
            time.sleep(_EMBED_BATCH_SLEEP)

    matrix = np.array(all_vecs, dtype=np.float32)
    return _normalize(matrix)


# ── main class ─────────────────────────────────────────────────────────────────

class GeneralRAGRetriever:
    """
    A reusable retriever that can index any list of text documents and
    answer similarity queries against them.

    Parameters
    ----------
    api_key         : OpenAI API key. Falls back to OPEN_AI_API_KEY /
                      OPENAI_API_KEY environment variables.
    embedding_model : OpenAI embedding model name.
    """

    def __init__(
        self,
        api_key: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._client = _make_client(api_key)
        self._model = embedding_model
        self._index: faiss.Index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    # ── building the index ────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        show_progress: bool = False,
    ) -> "GeneralRAGRetriever":
        """
        Embed and index a list of documents.

        Can be called multiple times to add documents incrementally.

        Parameters
        ----------
        documents     : Plain-text strings to index.
        metadata      : Optional per-document dicts (e.g. ids, labels).
                        Defaults to {"index": i} if omitted.
        show_progress : Print embedding progress to stdout.

        Returns self so calls can be chained.
        """
        if not documents:
            return self

        if metadata is None:
            offset = len(self._documents)
            metadata = [{"index": offset + i} for i in range(len(documents))]

        if len(metadata) != len(documents):
            raise ValueError(
                f"documents ({len(documents)}) and metadata ({len(metadata)}) "
                "must have the same length."
            )

        if show_progress:
            print(f"Embedding {len(documents)} document(s) …")

        vectors = embed_texts(documents, self._client, self._model, show_progress)
        self._index.add(vectors)
        self._documents.extend(documents)
        self._metadata.extend(metadata)

        return self

    # ── querying ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar documents to the query.

        Parameters
        ----------
        query            : Query text (will be embedded on the fly).
        k                : Maximum number of results to return.
        score_threshold  : If set, only return results with cosine
                           similarity ≥ this value (range 0–1).

        Returns
        -------
        List of dicts ordered by descending similarity:
          {
            "rank":     int,
            "score":    float,    # cosine similarity (0–1)
            "document": str,
            "metadata": dict,
          }
        """
        if self._index.ntotal == 0:
            return []

        query_vec = embed_texts([query], self._client, self._model)
        fetch_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, fetch_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            if score_threshold is not None and float(score) < score_threshold:
                continue
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "document": self._documents[idx],
                    "metadata": self._metadata[idx],
                }
            )

        return results

    def retrieve_batch(
        self,
        queries: list[str],
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve top-k results for multiple queries in a single embedding call.

        Returns a list of result lists, one per query.
        """
        if self._index.ntotal == 0:
            return [[] for _ in queries]

        query_vecs = embed_texts(queries, self._client, self._model)
        fetch_k = min(k, self._index.ntotal)
        all_scores, all_indices = self._index.search(query_vecs, fetch_k)

        batch_results = []
        for scores, indices in zip(all_scores, all_indices):
            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx == -1:
                    continue
                if score_threshold is not None and float(score) < score_threshold:
                    continue
                results.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "document": self._documents[idx],
                        "metadata": self._metadata[idx],
                    }
                )
            batch_results.append(results)

        return batch_results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """
        Persist the index, documents, and metadata to disk.

        Creates the directory if it does not exist. Three files are written:
          <directory>/faiss.index
          <directory>/documents.pkl
          <directory>/metadata.json
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "faiss.index"))

        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        api_key: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "GeneralRAGRetriever":
        """
        Load a previously saved retriever from disk.

        Parameters
        ----------
        directory       : Directory containing faiss.index, documents.pkl,
                          and metadata.json (written by save()).
        api_key         : OpenAI API key for embedding future queries.
        embedding_model : Must match the model used during indexing.
        """
        directory = Path(directory)

        instance = cls.__new__(cls)
        instance._client = _make_client(api_key)
        instance._model = embedding_model
        instance._index = faiss.read_index(str(directory / "faiss.index"))

        with open(directory / "documents.pkl", "rb") as f:
            instance._documents = pickle.load(f)

        with open(directory / "metadata.json", "r", encoding="utf-8") as f:
            instance._metadata = json.load(f)

        return instance

    # ── introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"GeneralRAGRetriever(docs={len(self)}, "
            f"model={self._model!r})"
        )
