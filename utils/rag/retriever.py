"""
RAG Retriever – loads the pre-built FAISS index and retrieves the
top-k most clinically similar cases for k-shot prompting.

Usage
-----
    from utils.rag.retriever import RAGRetriever

    retriever = RAGRetriever()          # loads index from default location
    results = retriever.retrieve(query_clinical_text, k=5)

    for r in results:
        print(r["score"], r["metadata"]["diagnosis"])
        print(r["document"])            # full k-shot example text
"""

import numpy as np
from pathlib import Path
from openai import OpenAI

from utils.rag.indexer import load_index, _normalize, EMBEDDING_MODEL

DEFAULT_INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "rag_index"


class RAGRetriever:
    """
    Wraps the FAISS index and provides a single `retrieve` method.

    Parameters
    ----------
    index_dir : path to the directory containing faiss.index, documents.pkl,
                and metadata.json (produced by build_index.py).
    api_key   : OpenAI API key used to embed query texts.
    """

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPEN_AI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPEN_AI_API_KEY in .env "
                    "or pass api_key= explicitly."
                )

        self._client = OpenAI(api_key=api_key)
        self._index, self._documents, self._metadata = load_index(index_dir)

    def _embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL, input=[text]
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        return _normalize(vec)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        exclude_mrnos: list[int] | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most similar patient cases.

        Parameters
        ----------
        query        : Clinical text describing the query patient (vitals,
                       lab results, history, etc.). Use
                       document_builder.build_clinical_text() to format it.
        k            : Number of examples to return.
        exclude_mrnos: List of MRNos to exclude from results (e.g. the
                       query patient itself if it exists in the index).

        Returns
        -------
        List of dicts, each containing:
          {
            "rank":     int,          # 1-indexed
            "score":    float,        # cosine similarity (higher = more similar)
            "document": str,          # full k-shot example text
            "metadata": {
                "mrno":      int,
                "diagnosis": str,
                "reasoning": str,
                "icd_codes": list[str],
            }
          }
        """
        exclude_set = set(exclude_mrnos or [])

        query_vec = self._embed(query)

        # Retrieve extra candidates in case we need to filter some out
        fetch_k = k + len(exclude_set) + 5
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            if meta["mrno"] in exclude_set:
                continue
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "document": self._documents[idx],
                "metadata": meta,
            })
            if len(results) == k:
                break

        return results

    def retrieve_for_patient(
        self,
        patient_dict: dict,
        k: int = 5,
    ) -> list[dict]:
        """
        Convenience wrapper: builds the clinical text from a raw patient
        dict (matching the JSON schema in sample-patient-info.json) and
        retrieves similar cases, automatically excluding this patient if
        they are already in the index.

        Parameters
        ----------
        patient_dict : Parsed patient JSON (as returned by json.loads on
                       the patient_details column).
        k            : Number of examples to return.
        """
        from utils.rag.document_builder import build_clinical_text

        clinical_text = build_clinical_text(patient_dict)
        mrno = int(patient_dict.get("patient_id", -1))
        return self.retrieve(clinical_text, k=k, exclude_mrnos=[mrno])

    def __len__(self) -> int:
        return self._index.ntotal
