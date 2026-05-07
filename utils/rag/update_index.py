"""
Incremental updates to the FAISS RAG index.

Used after a clinician approves an MDT consensus to append the new case
into the existing index without re-embedding the full corpus.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss

from utils.rag.retriever import DEFAULT_INDEX_DIR, RAGRetriever


def next_mrno(metadata: List[Dict[str, Any]]) -> int:
    """Return ``max(mrno) + 1`` over the existing metadata, or 1 if empty."""
    if not metadata:
        return 1
    existing = [int(m.get("mrno", 0) or 0) for m in metadata]
    return (max(existing) if existing else 0) + 1


def _extract_icd_codes(report: Dict[str, Any]) -> List[str]:
    """Pull ICD codes if the validated report happens to include them.

    The current moderator schema does not formally include ICD codes, so this
    is best-effort: we accept either ``diagnosis.icd_codes`` or a top-level
    ``icd_codes`` list.
    """
    diag = report.get("diagnosis") or {}
    if isinstance(diag, dict):
        codes = diag.get("icd_codes")
        if isinstance(codes, list):
            return [str(c) for c in codes if c]
    top = report.get("icd_codes")
    if isinstance(top, list):
        return [str(c) for c in top if c]
    return []


def report_to_full_document(
    clinical_text: str, validated_report: Dict[str, Any]
) -> str:
    """Format an approved consensus into the same k-shot template used by
    :func:`utils.rag.document_builder.build_full_document`.
    """
    diag = validated_report.get("diagnosis") or {}
    primary = diag.get("primary_diagnosis", "")
    reasoning = diag.get("evidence_summary", "")
    icd_codes = _extract_icd_codes(validated_report)

    doc = f"=== CLINICAL PRESENTATION ===\n{clinical_text}\n\n=== DIAGNOSIS ===\n"
    doc += f"Diagnosis: {primary}\n"
    if reasoning:
        doc += f"Reasoning: {reasoning}\n"
    if icd_codes:
        doc += f"ICD-10 Codes: {', '.join(icd_codes)}\n"
    return doc


def append_document(
    retriever: RAGRetriever,
    clinical_text: str,
    full_document: str,
    metadata: Dict[str, Any],
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> int:
    """Embed ``clinical_text`` and append it (with its document + metadata)
    to the live retriever and to the on-disk index.

    Returns the assigned MRNo.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    vec = retriever._embed(clinical_text)  # noqa: SLF001 - intentional reuse
    retriever._index.add(vec)               # noqa: SLF001
    retriever._documents.append(full_document)  # noqa: SLF001
    retriever._metadata.append(metadata)        # noqa: SLF001

    faiss.write_index(retriever._index, str(index_dir / "faiss.index"))  # noqa: SLF001
    with open(index_dir / "documents.pkl", "wb") as f:
        pickle.dump(retriever._documents, f)  # noqa: SLF001
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(retriever._metadata, f, ensure_ascii=False, indent=2)  # noqa: SLF001

    return int(metadata.get("mrno", -1))
