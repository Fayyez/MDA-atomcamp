"""
Builds text documents from patient CSV and diagnosis CSV for RAG indexing.

Each document combines a patient's clinical data with their known diagnosis,
so that retrieved examples serve directly as k-shot prompting context.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional


def _format_vitals(vitals: dict) -> str:
    parts = []
    if vitals.get("blood_pressure"):
        parts.append(f"BP: {vitals['blood_pressure']}")
    if vitals.get("temperature"):
        parts.append(f"Temp: {vitals['temperature']}{vitals.get('temperature_unit', '')}")
    if vitals.get("pulse"):
        parts.append(f"Pulse: {vitals['pulse']}{vitals.get('pulse_unit', '')}")
    if vitals.get("respiratory_rate"):
        parts.append(f"RR: {vitals['respiratory_rate']}{vitals.get('respiratory_rate_unit', '')}")
    if vitals.get("weight"):
        parts.append(f"Weight: {vitals['weight']}{vitals.get('weight_unit', '')}")
    if vitals.get("height"):
        parts.append(f"Height: {vitals['height']}{vitals.get('height_unit', '')}")
    return ", ".join(parts) if parts else "N/A"


def _format_lab_results(lab_results: list) -> str:
    if not lab_results:
        return "None"
    lines = []
    for lab in lab_results:
        name = lab.get("cpt_name", "Unknown test")
        date = lab.get("date", "")
        results = lab.get("results", {})
        result_strs = []
        for marker, data in results.items():
            val = data.get("result", "")
            unit = data.get("unit", "")
            normal = data.get("normal_range", ["", ""])
            normal_str = f" (normal: {normal[0]}-{normal[1]})" if normal[0] or normal[1] else ""
            result_strs.append(f"{marker}: {val}{' ' + unit if unit else ''}{normal_str}")
        lines.append(f"  [{date}] {name}: {'; '.join(result_strs)}")
    return "\n".join(lines)


def _format_radiology(reports: list) -> str:
    if not reports:
        return "None"
    lines = []
    for rep in reports:
        name = rep.get("cpt_name", "Unknown scan")
        date = rep.get("date", "")[:10] if rep.get("date") else ""
        conclusion = rep.get("conclusion") or rep.get("result", "")
        if conclusion:
            conclusion = conclusion.strip().replace("\n", " ")[:500]
        lines.append(f"  [{date}] {name}: {conclusion}")
    return "\n".join(lines)


def _format_encounters(encounters: list) -> str:
    if not encounters:
        return "None"
    lines = []
    for enc in encounters[:3]:
        date = enc.get("date", "")[:10] if enc.get("date") else ""
        clinician = enc.get("clinician", "")
        notes = enc.get("notes", "").strip().replace("\n", " ")[:300]
        lines.append(f"  [{date}] {clinician}: {notes}")
    return "\n".join(lines)


def build_clinical_text(patient: dict) -> str:
    """
    Converts a patient record dict into a plain-text clinical summary
    suitable for embedding. This is the text used for similarity search.
    """
    info = patient.get("personal_information", {})
    lines = [
        f"Patient: {info.get('gender', 'Unknown')} | Age: {info.get('age', 'Unknown')} | DOB: {info.get('dob', '')}",
    ]

    history = patient.get("known_medical_history", [])
    if history:
        lines.append(f"Medical History: {' | '.join(history)}")

    symptoms = patient.get("current_symptoms", [])
    if symptoms:
        lines.append(f"Current Symptoms: {', '.join(str(s) for s in symptoms)}")

    lines.append(f"Vitals: {_format_vitals(patient.get('vitals', {}))}")

    lab_text = _format_lab_results(patient.get("lab_results", []))
    lines.append(f"Lab Results:\n{lab_text}")

    radiology_text = _format_radiology(patient.get("radiology_reports", []))
    lines.append(f"Radiology:\n{radiology_text}")

    encounters_text = _format_encounters(patient.get("recent_encounters", []))
    lines.append(f"Recent Encounters:\n{encounters_text}")

    meds = patient.get("medications", [])
    if meds:
        lines.append(f"Medications: {', '.join(str(m) for m in meds)}")

    return "\n".join(lines)


def build_full_document(clinical_text: str, diagnosis_row: Optional[pd.Series]) -> str:
    """
    Combines clinical text with the known diagnosis to form the complete
    k-shot example document that will be returned during retrieval.
    """
    doc = f"=== CLINICAL PRESENTATION ===\n{clinical_text}"
    if diagnosis_row is not None:
        diagnosis = diagnosis_row.get("Diagnosis", "")
        reasoning = diagnosis_row.get("Reasoning", "")
        icd_raw = diagnosis_row.get("ICD-10 Codes", "")

        icd_codes = []
        if icd_raw and str(icd_raw).strip() not in ("", "nan"):
            try:
                icd_data = json.loads(str(icd_raw))
                icd_codes = icd_data.get("ICD_codes", [])
            except (json.JSONDecodeError, TypeError):
                icd_codes = []

        doc += f"\n\n=== DIAGNOSIS ===\n"
        doc += f"Diagnosis: {diagnosis}\n"
        if reasoning:
            doc += f"Reasoning: {reasoning}\n"
        if icd_codes:
            doc += f"ICD-10 Codes: {', '.join(icd_codes)}\n"

    return doc


def load_documents(
    patients_csv: str | Path,
    diagnosis_csv: str | Path,
) -> list[dict]:
    """
    Loads and joins both CSVs, returning a list of document dicts:
      {
        "mrno": int,
        "clinical_text": str,   # used for embedding
        "full_document": str,   # returned to caller (k-shot example)
        "metadata": {
            "mrno": int,
            "diagnosis": str,
            "reasoning": str,
            "icd_codes": list[str],
        }
      }
    """
    patients_df = pd.read_csv(patients_csv)
    diagnosis_df = pd.read_csv(diagnosis_csv)

    diagnosis_map = {int(row["MRNo"]): row for _, row in diagnosis_df.iterrows()}

    documents = []
    for _, row in patients_df.iterrows():
        mrno = int(row["mrno"])
        try:
            patient = json.loads(row["patient_details"])
        except (json.JSONDecodeError, KeyError):
            continue

        clinical_text = build_clinical_text(patient)
        diag_row = diagnosis_map.get(mrno)
        full_document = build_full_document(clinical_text, diag_row)

        icd_codes = []
        diagnosis = ""
        reasoning = ""
        if diag_row is not None:
            diagnosis = str(diag_row.get("Diagnosis", ""))
            reasoning = str(diag_row.get("Reasoning", ""))
            icd_raw = diag_row.get("ICD-10 Codes", "")
            if icd_raw and str(icd_raw).strip() not in ("", "nan"):
                try:
                    icd_data = json.loads(str(icd_raw))
                    icd_codes = icd_data.get("ICD_codes", [])
                except (json.JSONDecodeError, TypeError):
                    icd_codes = []

        documents.append({
            "mrno": mrno,
            "clinical_text": clinical_text,
            "full_document": full_document,
            "metadata": {
                "mrno": mrno,
                "diagnosis": diagnosis,
                "reasoning": reasoning,
                "icd_codes": icd_codes,
            },
        })

    return documents
