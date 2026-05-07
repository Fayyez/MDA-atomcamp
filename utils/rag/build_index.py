"""
One-time script: builds the FAISS RAG index from the patient and diagnosis CSVs.

Run this once (or whenever the data changes):
    python -m utils.rag.build_index

The index is saved to data/rag_index/ and is loaded automatically by RAGRetriever.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from utils.rag.document_builder import load_documents
from utils.rag.indexer import build_index


PATIENTS_CSV = ROOT / "data" / "all_patients.csv"
DIAGNOSIS_CSV = ROOT / "data" / "diagnosis.csv"
INDEX_DIR = ROOT / "data" / "rag_index"


def main() -> None:
    api_key = os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPEN_AI_API_KEY not set. Add it to your .env file."
        )

    print("Loading documents …")
    documents = load_documents(PATIENTS_CSV, DIAGNOSIS_CSV)
    print(f"Loaded {len(documents)} patient records.")

    build_index(documents, api_key=api_key, index_dir=INDEX_DIR)
    print("Done. Index is ready for use with RAGRetriever.")


if __name__ == "__main__":
    main()
