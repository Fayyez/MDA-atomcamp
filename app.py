"""AI Clinical MDT (Multi-Disciplinary Team) Simulator.

Streamlit app that orchestrates a multi-round debate between six clinical
AI agents (Radiologist, Pathologist, Oncologist, Surgeon, Pharmacist,
Insurance/Cost Agent) using the OpenAI ``gpt-4o-mini`` model and produces
a structured consensus report.

The OpenAI API key is loaded from environment variables (``.env`` file or
the system environment) and is never requested in the UI.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from utils.rag import (
    RAGRetriever,
    append_document,
    build_clinical_text,
    next_mrno,
    report_to_full_document,
)


# ---------------------------------------------------------------------------
# Environment & client setup
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Clinical MDT Simulator",
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
if not API_KEY:
    st.error(
        "OpenAI API key not found in environment. "
        "Create a `.env` file with `OPENAI_API_KEY=your_key_here` "
        "(or legacy `OPEN_AI_API_KEY=your_key_here`) "
        "(see `.env.example`) and restart the app."
    )
    st.stop()

client = OpenAI(api_key=API_KEY)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
TEMP_R1 = 0.2
TEMP_R2 = 0.25
TEMP_MOD = 0.2
TEMP_VALIDATOR = 0.2
MAX_TOKENS_AGENT = 800
MAX_TOKENS_MODERATOR = 1600
MAX_TOKENS_VALIDATOR = 1800
RETRY_ATTEMPTS = 2
RAG_K = 5
SAMPLE_PATIENT_PATH = Path("data") / "sample-patient-info.json"

AGENT_ORDER: List[str] = [
    "Radiologist",
    "Pathologist",
    "Oncologist",
    "Surgeon",
    "Pharmacist",
    "Insurance/Cost Agent",
]


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

# Reusable suffix that enforces detailed, evidence-grounded clinical reasoning
# for every agent. Each prompt below appends this block.
DETAIL_REQUIREMENTS = """
RESPONSE REQUIREMENTS (mandatory for every reply):
- Write at least 3-5 substantive sentences (no telegraphic bullet lists only).
- Reference at least three specific patient data points by name AND value
  (e.g. "ESR 104", "pulse 127/min", "tree-in-bud pattern on HRCT").
- State an explicit numeric confidence percentage for each clinical claim
  you make (e.g. "75% confident of MAC pulmonary disease").
- If a needed data element is missing, write
  "insufficient data to conclude X, recommend Y test/imaging".
- Do NOT invent values that are not present in the provided patient summary.
- Stay strictly within your role; defer outside-scope questions to the
  appropriate teammate by name.
"""

AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "Radiologist": (
        "You are a senior board-certified radiologist participating in a "
        "hospital MDT. Interpret every entry in `radiology_reports` in "
        "detail. Describe key findings (e.g. tree-in-bud pattern, "
        "lymphadenopathy, bronchiectasis, consolidation, effusion) and the "
        "anatomic distribution. Provide an imaging-based differential "
        "diagnosis with a numeric confidence for each entry, and suggest "
        "follow-up imaging or alternative modalities if useful." + DETAIL_REQUIREMENTS
    ),
    "Pathologist": (
        "You are a clinical pathologist with expertise in lab medicine and "
        "histology. Analyse `lab_results` (flagging abnormal values, "
        "trends, and how they sit relative to typical reference ranges "
        "even when the JSON does not include them). Correlate findings "
        "with `known_medical_history`. Provide a cellular, inflammatory, "
        "and (where relevant) molecular interpretation, and explicitly "
        "list missing tests (e.g. AFB culture, GeneXpert, ANA, RF, "
        "procalcitonin) that would resolve diagnostic uncertainty."
        + DETAIL_REQUIREMENTS
    ),
    "Oncologist": (
        "You are a consultant medical oncologist. Evaluate malignancy risk "
        "from all available imaging, lab, vitals, and history data. If "
        "cancer is unlikely, state so and explain why with explicit "
        "numeric confidence. If suspicious, propose a staging workup "
        "(imaging, biomarkers, biopsy targets), discuss systemic therapy "
        "options at a high level, and comment on treatment urgency and "
        "expected outcomes." + DETAIL_REQUIREMENTS
    ),
    "Surgeon": (
        "You are a senior cardiothoracic / general surgeon. Assess whether "
        "any invasive procedure (biopsy, resection, drainage, "
        "bronchoscopy, lymph node sampling) is indicated based on imaging, "
        "vitals, and lab data. Consider surgical risk in light of `vitals` "
        "and comorbidities. State clearly whether surgery is necessary, "
        "elective, or not indicated, and provide a concise risk-benefit "
        "discussion in prose." + DETAIL_REQUIREMENTS
    ),
    "Pharmacist": (
        "You are a senior clinical pharmacist. Review `medications` already "
        "prescribed and any drugs proposed by other agents in the "
        "transcript. Check drug-drug interactions, dose adjustments for "
        "renal/hepatic function (inferred conservatively from labs and "
        "history), adverse effect risks, monitoring parameters (e.g. LFTs, "
        "renal panel, glucose), and lower-cost therapeutic alternatives. "
        "Recommend a complete medication plan with explicit rationale."
        + DETAIL_REQUIREMENTS
    ),
    "Insurance/Cost Agent": (
        "You are a healthcare financial advisor on the MDT. Estimate the "
        "total cost of the proposed diagnostic and treatment pathway, "
        "broken down into consultation, imaging, labs, procedures, drugs, "
        "and follow-up. Provide a payer coverage likelihood (low / medium "
        "/ high) for each major component and approximate USD ranges "
        "(citing public US AWP / international references qualitatively, "
        "e.g. 'azithromycin 250mg daily ~$10/month'). Suggest equivalent "
        "lower-cost alternatives that do not compromise quality, and give "
        "an out-of-pocket estimate range." + DETAIL_REQUIREMENTS
    ),
}


MODERATOR_SYSTEM_PROMPT = """You are the MDT chairperson. Synthesize the detailed debate from all six agents (both rounds). Produce a structured JSON output as defined below. Resolve conflicts by weighing evidence, and if no consensus exists, state the disagreement explicitly. Provide final diagnosis confidence, a prioritized treatment plan, surgery necessity, cost-risk analysis, and any unresolved issues.

Return ONLY a single JSON object that matches this schema exactly (no prose, no markdown fences):
{
  "patient_id": "string",
  "date_of_mdt": "YYYY-MM-DD",
  "diagnosis": {
    "primary_diagnosis": "detailed name",
    "confidence": 0,
    "differentials": [
      {"diagnosis": "name", "confidence": 0}
    ],
    "evidence_summary": "Key imaging/lab/history findings supporting primary diagnosis"
  },
  "treatment_plan": {
    "recommendations": ["step 1", "step 2"],
    "alternative_plan": "description",
    "surgery": {
      "necessary": false,
      "procedure": null,
      "urgency": "elective/urgent/emergent/none",
      "risks": ["risk1", "risk2"]
    },
    "pharmacotherapy": {
      "regimen": "drug names + doses",
      "duration_weeks": 0,
      "monitoring": "what labs/follow-up"
    }
  },
  "cost_analysis": {
    "estimated_total_usd": 0,
    "insurance_coverage_likelihood": "low/medium/high",
    "out_of_pocket_estimate_usd": 0,
    "cost_effectiveness_note": "string"
  },
  "unresolved_issues": ["question1", "question2"],
  "agent_consensus_notes": "Summary of agreements and major disagreements from Round 2"
}
Confidence values must be integers 0-100. Numeric cost fields must be plain numbers (no currency strings). If a value is genuinely unknown, choose the most defensible estimate and explain it inside the relevant string field rather than leaving the JSON malformed."""


RAG_VALIDATOR_SYSTEM_PROMPT = """You are the MDT chairperson reviewing your own draft consensus against k similar historical cases retrieved from the hospital RAG corpus.

Inputs you will receive:
- DRAFT_CONSENSUS_JSON: your earlier output for this patient.
- PATIENT_SUMMARY: the same clinical summary the agents debated.
- RETRIEVED_CASES: k examples, each with their clinical presentation, final diagnosis, reasoning, ICD-10 codes, and a similarity score (cosine, higher = more similar).

Rules:
1. If a clear majority of retrieved cases support the primary diagnosis (semantically equivalent labels count), KEEP it and raise diagnosis.confidence by 5-15 (cap at 95).
2. If retrieved cases consistently point to a different label, REPLACE primary_diagnosis with the better-supported label and set a defensible confidence (40-80 depending on agreement strength). Update differentials, evidence_summary, and pharmacotherapy.regimen as needed to stay internally consistent.
3. If retrieved cases are mixed or only weakly relevant, KEEP the diagnosis but lower confidence by 5-10.
4. Preserve all other fields from the draft (treatment_plan, cost_analysis, unresolved_issues, agent_consensus_notes) unless a corrected diagnosis logically requires changing them.
5. Add EXACTLY two new fields at the top level:
     "rag_validation_notes": string explaining the comparison and decision (cite MRNos),
     "retrieved_cases_summary": list of {"mrno": int, "diagnosis": "string", "score": number}
6. Confidence values remain integers 0-100. Numeric cost fields remain plain numbers.

Return ONLY a single JSON object that matches the original consensus schema PLUS the two new fields. No prose, no markdown fences."""


# ---------------------------------------------------------------------------
# Patient summary builder
# ---------------------------------------------------------------------------


def _safe(value: Any, default: str = "n/a") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _format_vitals(vitals: Dict[str, Any]) -> str:
    if not vitals:
        return "Vitals: not provided."
    parts = [
        f"BP {_safe(vitals.get('blood_pressure'))}",
        f"pulse {_safe(vitals.get('pulse'))}{_safe(vitals.get('pulse_unit'), '')}",
        f"resp rate {_safe(vitals.get('respiratory_rate'))}"
        f"{_safe(vitals.get('respiratory_rate_unit'), '')}",
        f"temp {_safe(vitals.get('temperature'))}"
        f"{_safe(vitals.get('temperature_unit'), '')}",
        f"weight {_safe(vitals.get('weight'))}"
        f"{_safe(vitals.get('weight_unit'), '')}",
        f"height {_safe(vitals.get('height'))}"
        f"{_safe(vitals.get('height_unit'), '')}",
    ]
    return "Vitals: " + ", ".join(parts) + "."


def _format_labs(labs: List[Dict[str, Any]]) -> str:
    if not labs:
        return "Lab results: none provided."
    lines = ["Lab results:"]
    for entry in labs:
        date = entry.get("date", "n/a")
        cpt_name = entry.get("cpt_name", "Unknown panel")
        results = entry.get("results", {}) or {}
        if not results:
            lines.append(f"  - {cpt_name} ({date}): no results recorded")
            continue
        for analyte, payload in results.items():
            payload = payload or {}
            value = payload.get("result", "n/a")
            unit = payload.get("unit", "") or ""
            normal = payload.get("normal_range") or ["", ""]
            normal_str = (
                f" (normal {normal[0]}-{normal[1]})"
                if any(str(n).strip() for n in normal)
                else ""
            )
            lines.append(
                f"  - {analyte}: {value}{unit}{normal_str} "
                f"[panel: {cpt_name}, date: {date}]"
            )
    return "\n".join(lines)


def _format_radiology(reports: List[Dict[str, Any]]) -> str:
    if not reports:
        return "Radiology reports: none provided."
    lines = ["Radiology reports:"]
    for r in reports:
        lines.append(
            f"- {r.get('cpt_name', 'Imaging')} " f"({r.get('date', 'date n/a')})"
        )
        if r.get("technique"):
            lines.append(f"  Technique: {r['technique']}")
        if r.get("result"):
            lines.append(f"  Findings: {r['result'].strip()}")
        if r.get("conclusion"):
            lines.append(f"  Conclusion: {r['conclusion'].strip()}")
    return "\n".join(lines)


def _format_meds(meds: List[Any]) -> str:
    if not meds:
        return "Current medications: none recorded in structured field."
    lines = ["Current medications:"]
    for m in meds:
        if isinstance(m, dict):
            lines.append(
                f"  - {m.get('name', 'unknown')} "
                f"{m.get('dose', '')} {m.get('frequency', '')}".strip()
            )
        else:
            lines.append(f"  - {m}")
    return "\n".join(lines)


def _format_encounters(encounters: List[Dict[str, Any]]) -> str:
    if not encounters:
        return "Recent encounters: none."
    lines = ["Recent encounters:"]
    for e in encounters:
        notes = (e.get("notes") or "").strip()
        if len(notes) > 700:
            notes = notes[:700] + "... [truncated]"
        lines.append(
            f"- {e.get('date', 'date n/a')} "
            f"({e.get('clinician', 'clinician n/a')}):\n  {notes}"
        )
    return "\n".join(lines)


def build_patient_summary(data: Dict[str, Any]) -> str:
    """Render patient JSON into a compact, agent-friendly text block."""

    pid = data.get("patient_id", "unknown")
    personal = data.get("personal_information", {}) or {}
    history = data.get("known_medical_history", []) or []
    symptoms = data.get("current_symptoms", []) or []

    header = (
        f"Patient ID: {pid}\n"
        f"Demographics: {personal.get('gender', 'unknown')} "
        f"age {personal.get('age', 'unknown')} "
        f"(DOB {personal.get('dob', 'unknown')})."
    )
    history_block = "Known medical history:\n" + (
        "\n".join(f"  - {h}" for h in history) if history else "  - none recorded"
    )
    symptoms_block = "Current symptoms:\n" + (
        "\n".join(f"  - {s}" for s in symptoms) if symptoms else "  - none recorded"
    )

    return "\n\n".join(
        [
            header,
            _format_vitals(data.get("vitals", {}) or {}),
            history_block,
            symptoms_block,
            _format_labs(data.get("lab_results", []) or []),
            _format_radiology(data.get("radiology_reports", []) or []),
            _format_meds(data.get("medications", []) or []),
            _format_encounters(data.get("recent_encounters", []) or []),
        ]
    )


# ---------------------------------------------------------------------------
# OpenAI call helpers
# ---------------------------------------------------------------------------


def call_agent(
    messages: List[Dict[str, str]],
    *,
    temperature: float = TEMP_R1,
    max_tokens: int = MAX_TOKENS_AGENT,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    """Call the chat completion endpoint with retry + exponential backoff."""

    last_err: Optional[Exception] = None
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                raise OpenAIError("Empty completion content from model")
            return content.strip()
        except OpenAIError as exc:
            last_err = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(2**attempt)
            else:
                break
        except Exception as exc:  # network etc.
            last_err = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(2**attempt)
            else:
                break

    err_msg = f"OpenAI API call failed after {RETRY_ATTEMPTS + 1} attempts: {last_err}"
    st.error(err_msg)
    raise RuntimeError(err_msg) from last_err


def _format_round_transcript(transcript: List[Dict[str, str]]) -> str:
    blocks = []
    for entry in transcript:
        blocks.append(
            f"### {entry['agent']} ({entry.get('timestamp', '')})\n{entry['content']}"
        )
    return "\n\n".join(blocks)


def _build_round1_messages(
    agent: str,
    patient_summary: str,
    prior_in_round: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    user_parts = [
        "PATIENT DATA SUMMARY (extracted from JSON):",
        patient_summary,
    ]
    if prior_in_round:
        user_parts.append(
            "Earlier opinions in this same Round 1 (build on or push back on them):"
        )
        user_parts.append(_format_round_transcript(prior_in_round))
    user_parts.append(
        f"Now, as the {agent}, deliver your Round 1 opinion. "
        "Follow every response requirement in your role description."
    )
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPTS[agent]},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _build_round2_messages(
    agent: str,
    patient_summary: str,
    round1: List[Dict[str, str]],
    prior_in_round2: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    user_parts = [
        "PATIENT DATA SUMMARY (extracted from JSON):",
        patient_summary,
        "FULL ROUND 1 TRANSCRIPT (all six agents):",
        _format_round_transcript(round1),
    ]
    if prior_in_round2:
        user_parts.append("Earlier Round 2 rebuttals so far:")
        user_parts.append(_format_round_transcript(prior_in_round2))
    user_parts.append(
        f"As the {agent}, produce your Round 2 reply. You MUST:\n"
        "1. Quote and respond to at least one specific point from another "
        "agent (name them).\n"
        "2. Refine your own Round 1 recommendation with sharper numbers "
        "(updated confidence, exact dose ranges, etc.).\n"
        "3. Identify exactly one unresolved question and propose how to "
        "answer it (test, imaging, consult).\n"
        "Keep all the response requirements from your role."
    )
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPTS[agent]},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_round(
    round_num: int,
    patient_summary: str,
    round1_transcript: Optional[List[Dict[str, str]]] = None,
    *,
    progress_cb=None,
) -> List[Dict[str, str]]:
    """Run a single debate round sequentially over ``AGENT_ORDER``."""

    transcript: List[Dict[str, str]] = []
    for idx, agent in enumerate(AGENT_ORDER):
        if progress_cb:
            progress_cb(round_num, idx, agent, "running")

        if round_num == 1:
            messages = _build_round1_messages(agent, patient_summary, transcript)
            temperature = TEMP_R1
        else:
            assert round1_transcript is not None
            messages = _build_round2_messages(
                agent, patient_summary, round1_transcript, transcript
            )
            temperature = TEMP_R2

        content = call_agent(messages, temperature=temperature)
        transcript.append(
            {
                "agent": agent,
                "content": content,
                "timestamp": _now_iso(),
            }
        )
        if progress_cb:
            progress_cb(round_num, idx, agent, "done")

    return transcript


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort JSON parser for moderator output."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass

    return {
        "_parse_error": "Moderator response was not valid JSON.",
        "raw_response": text,
    }


@st.cache_resource(show_spinner="Loading RAG index...")
def get_retriever() -> Optional[RAGRetriever]:
    """Load the FAISS-backed retriever once per Streamlit session.

    Returns ``None`` if the index cannot be loaded; callers should treat that
    as "RAG validation unavailable" and fall back to the draft consensus.
    """
    try:
        return RAGRetriever(api_key=API_KEY)
    except Exception as exc:  # noqa: BLE001 - surfaced once via UI warning
        st.warning(f"RAG index unavailable, validation phase will be skipped: {exc}")
        return None


def _format_retrieved_cases(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "No retrieved cases."
    blocks = []
    for c in cases:
        meta = c.get("metadata", {}) or {}
        icd = ", ".join(meta.get("icd_codes") or []) or "n/a"
        blocks.append(
            f"### Case rank {c.get('rank')}  "
            f"(MRNo {meta.get('mrno', 'n/a')}, score {c.get('score', 0):.3f})\n"
            f"Diagnosis: {meta.get('diagnosis', 'n/a')}\n"
            f"ICD-10: {icd}\n"
            f"Reasoning: {meta.get('reasoning', 'n/a')}\n"
            f"---\n{c.get('document', '')}"
        )
    return "\n\n".join(blocks)


def validate_with_rag(
    draft_report: Dict[str, Any],
    patient_data: Dict[str, Any],
    retriever: RAGRetriever,
    k: int = RAG_K,
) -> Dict[str, Any]:
    """Run the second-pass validator using k retrieved similar cases.

    Returns a new consensus dict with the same schema plus
    ``rag_validation_notes`` and ``retrieved_cases_summary``. The list of raw
    retrieved cases (with full documents) is attached under the private key
    ``_retrieved_cases`` so the UI can render them.
    """
    clinical_text = build_clinical_text(patient_data)

    exclude: List[int] = []
    pid_raw = patient_data.get("patient_id")
    try:
        if pid_raw is not None:
            exclude.append(int(pid_raw))
    except (TypeError, ValueError):
        pass

    cases = retriever.retrieve(clinical_text, k=k, exclude_mrnos=exclude or None)

    user_payload = (
        f"PATIENT_SUMMARY:\n{build_patient_summary(patient_data)}\n\n"
        f"DRAFT_CONSENSUS_JSON:\n{json.dumps(draft_report, indent=2)}\n\n"
        f"RETRIEVED_CASES (k={len(cases)}):\n{_format_retrieved_cases(cases)}\n\n"
        "Now produce the validated consensus JSON. Return ONLY the JSON."
    )
    messages = [
        {"role": "system", "content": RAG_VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]
    raw = call_agent(
        messages,
        temperature=TEMP_VALIDATOR,
        max_tokens=MAX_TOKENS_VALIDATOR,
        response_format={"type": "json_object"},
    )
    validated = _extract_json_object(raw)
    validated["_retrieved_cases"] = cases
    return validated


def persist_approved_case(
    patient_data: Dict[str, Any],
    validated_report: Dict[str, Any],
    retriever: RAGRetriever,
) -> Dict[str, Any]:
    """Append an approved patient + diagnosis to the on-disk RAG index.

    Returns a dict with the assigned ``mrno`` and the diagnosis used for the
    new metadata entry.
    """
    clinical_text = build_clinical_text(patient_data)
    full_document = report_to_full_document(clinical_text, validated_report)

    diag = validated_report.get("diagnosis", {}) or {}
    primary = str(diag.get("primary_diagnosis", "") or "")
    reasoning = str(diag.get("evidence_summary", "") or "")

    icd_codes: List[str] = []
    if isinstance(diag.get("icd_codes"), list):
        icd_codes = [str(c) for c in diag["icd_codes"] if c]
    elif isinstance(validated_report.get("icd_codes"), list):
        icd_codes = [str(c) for c in validated_report["icd_codes"] if c]

    metadata = {
        "mrno": next_mrno(retriever._metadata),  # noqa: SLF001 - intentional
        "diagnosis": primary,
        "reasoning": reasoning,
        "icd_codes": icd_codes,
    }
    assigned = append_document(retriever, clinical_text, full_document, metadata)
    return {"mrno": assigned, "diagnosis": primary}


def call_moderator(
    patient_summary: str,
    round1: List[Dict[str, str]],
    round2: List[Dict[str, str]],
    patient_id: str,
) -> Dict[str, Any]:
    user_payload = (
        f"Patient ID: {patient_id}\n"
        f"Date of MDT (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
        f"PATIENT DATA SUMMARY:\n{patient_summary}\n\n"
        f"ROUND 1 TRANSCRIPT:\n{_format_round_transcript(round1)}\n\n"
        f"ROUND 2 TRANSCRIPT:\n{_format_round_transcript(round2)}\n\n"
        "Now produce the final consensus JSON object. Return ONLY the JSON."
    )
    messages = [
        {"role": "system", "content": MODERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]
    raw = call_agent(
        messages,
        temperature=TEMP_MOD,
        max_tokens=MAX_TOKENS_MODERATOR,
        response_format={"type": "json_object"},
    )
    return _extract_json_object(raw)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def report_to_markdown(report: Dict[str, Any]) -> str:
    if "_parse_error" in report:
        return (
            "# MDT Consensus Report\n\n"
            f"> Parsing error: {report['_parse_error']}\n\n"
            "## Raw moderator response\n\n```\n"
            + str(report.get("raw_response", ""))
            + "\n```\n"
        )

    diag = report.get("diagnosis", {}) or {}
    plan = report.get("treatment_plan", {}) or {}
    surgery = plan.get("surgery", {}) or {}
    pharma = plan.get("pharmacotherapy", {}) or {}
    cost = report.get("cost_analysis", {}) or {}

    differentials = (
        "\n".join(
            f"- {d.get('diagnosis', 'n/a')} ({d.get('confidence', 'n/a')}%)"
            for d in (diag.get("differentials") or [])
        )
        or "- none"
    )

    recommendations = (
        "\n".join(
            f"{i + 1}. {step}"
            for i, step in enumerate(plan.get("recommendations") or [])
        )
        or "_none_"
    )

    risks = ", ".join(surgery.get("risks") or []) or "n/a"

    unresolved = (
        "\n".join(f"- {q}" for q in (report.get("unresolved_issues") or [])) or "- none"
    )

    md = f"""# MDT Consensus Report

- **Patient ID:** {report.get('patient_id', 'n/a')}
- **Date of MDT:** {report.get('date_of_mdt', 'n/a')}

## Diagnosis

- **Primary:** {diag.get('primary_diagnosis', 'n/a')} ({diag.get('confidence', 'n/a')}% confidence)
- **Differentials:**
{differentials}

**Evidence summary:** {diag.get('evidence_summary', 'n/a')}

## Treatment plan

**Recommendations:**

{recommendations}

**Alternative plan:** {plan.get('alternative_plan', 'n/a')}

### Surgery

- Necessary: **{surgery.get('necessary', 'n/a')}**
- Procedure: {surgery.get('procedure', 'n/a')}
- Urgency: {surgery.get('urgency', 'n/a')}
- Risks: {risks}

### Pharmacotherapy

- Regimen: {pharma.get('regimen', 'n/a')}
- Duration (weeks): {pharma.get('duration_weeks', 'n/a')}
- Monitoring: {pharma.get('monitoring', 'n/a')}

## Cost analysis

- Estimated total (USD): **${cost.get('estimated_total_usd', 'n/a')}**
- Insurance coverage likelihood: {cost.get('insurance_coverage_likelihood', 'n/a')}
- Out-of-pocket estimate (USD): ${cost.get('out_of_pocket_estimate_usd', 'n/a')}
- Note: {cost.get('cost_effectiveness_note', 'n/a')}

## Unresolved issues

{unresolved}

## Agent consensus notes

{report.get('agent_consensus_notes', 'n/a')}
"""
    return md


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "patient_data": None,
        "round1_transcript": [],
        "round2_transcript": [],
        "final_report": None,
        "validated_report": None,
        "rag_examples": [],
        "user_verdict": None,
        "rag_persist_message": None,
        "debate_complete": False,
        "patient_source": None,
        # form: dynamic row counters
        "form_lab_count": 1,
        "form_rad_count": 1,
        "form_med_count": 0,
        "form_enc_count": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_debate_state() -> None:
    st.session_state.round1_transcript = []
    st.session_state.round2_transcript = []
    st.session_state.final_report = None
    st.session_state.validated_report = None
    st.session_state.rag_examples = []
    st.session_state.user_verdict = None
    st.session_state.rag_persist_message = None
    st.session_state.debate_complete = False


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Patient input")
        st.caption(
            "API key is read from the `OPENAI_API_KEY` environment variable. "
            "It is never requested in the UI."
        )

        uploaded = st.file_uploader(
            "Upload patient JSON",
            type=["json"],
            help="Upload a patient data JSON file conforming to the project schema.",
        )
        if uploaded is not None:
            try:
                st.session_state.patient_data = json.loads(
                    uploaded.read().decode("utf-8")
                )
                st.session_state.patient_source = uploaded.name
                _reset_debate_state()
                st.success(f"Loaded `{uploaded.name}`.")
            except json.JSONDecodeError as exc:
                st.error(f"Could not parse uploaded JSON: {exc}")

        if st.button(
            "Load sample patient",
            help=f"Load `{SAMPLE_PATIENT_PATH.as_posix()}`.",
            use_container_width=True,
        ):
            try:
                with SAMPLE_PATIENT_PATH.open("r", encoding="utf-8") as fh:
                    st.session_state.patient_data = json.load(fh)
                st.session_state.patient_source = SAMPLE_PATIENT_PATH.as_posix()
                _reset_debate_state()
                st.success(
                    f"Loaded sample patient from `{SAMPLE_PATIENT_PATH.as_posix()}`."
                )
            except FileNotFoundError:
                st.error(f"Sample file not found at {SAMPLE_PATIENT_PATH}.")
            except json.JSONDecodeError as exc:
                st.error(f"Sample file is not valid JSON: {exc}")

        if st.session_state.patient_data is not None:
            with st.expander("Edit loaded JSON"):
                edited = st.text_area(
                    "Patient JSON",
                    value=json.dumps(st.session_state.patient_data, indent=2),
                    height=300,
                    label_visibility="collapsed",
                )
                if st.button("Apply edits"):
                    try:
                        st.session_state.patient_data = json.loads(edited)
                        _reset_debate_state()
                        st.success("Patient JSON updated.")
                    except json.JSONDecodeError as exc:
                        st.error(f"Edits not applied: invalid JSON ({exc}).")

        st.divider()

        run_disabled = st.session_state.patient_data is None
        run_clicked = st.button(
            "Start Detailed MDT Debate",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
            help=(
                "Runs Round 1 + Round 2 + Moderator. "
                "Expect 30-60 seconds and ~10-15K tokens of usage."
            ),
        )
        if run_clicked:
            _run_debate()

        if st.session_state.debate_complete:
            if st.button(
                "Clear debate (keep patient)",
                use_container_width=True,
            ):
                _reset_debate_state()
                st.rerun()

        if st.session_state.patient_source:
            st.caption(f"Loaded source: `{st.session_state.patient_source}`")


def _run_debate() -> None:
    data = st.session_state.patient_data
    patient_summary = build_patient_summary(data)
    patient_id = str(data.get("patient_id", "unknown"))

    _reset_debate_state()

    total_steps = len(AGENT_ORDER) * 2 + 2
    progress_bar = st.progress(0.0, text="Preparing MDT debate...")

    def _bump(step_idx: int, label: str) -> None:
        progress_bar.progress(
            min(step_idx / total_steps, 1.0),
            text=label,
        )

    with st.status("Running MDT debate...", expanded=True) as status:
        try:
            status.update(label="Round 1 - initial positions", state="running")
            r1_completed = 0

            def cb_r1(round_num: int, idx: int, agent: str, phase: str) -> None:
                nonlocal r1_completed
                if phase == "running":
                    st.write(f"Round 1 - {agent} thinking...")
                else:
                    r1_completed += 1
                    st.write(f"Round 1 - {agent} done.")
                    _bump(r1_completed, f"Round 1 ({r1_completed}/{len(AGENT_ORDER)})")

            r1 = run_round(1, patient_summary, progress_cb=cb_r1)
            st.session_state.round1_transcript = r1

            status.update(label="Round 2 - rebuttals & deepening", state="running")
            r2_completed = 0

            def cb_r2(round_num: int, idx: int, agent: str, phase: str) -> None:
                nonlocal r2_completed
                if phase == "running":
                    st.write(f"Round 2 - {agent} thinking...")
                else:
                    r2_completed += 1
                    st.write(f"Round 2 - {agent} done.")
                    _bump(
                        len(AGENT_ORDER) + r2_completed,
                        f"Round 2 ({r2_completed}/{len(AGENT_ORDER)})",
                    )

            r2 = run_round(2, patient_summary, r1, progress_cb=cb_r2)
            st.session_state.round2_transcript = r2

            status.update(label="Moderator synthesis", state="running")
            st.write("Moderator synthesising consensus report...")
            report = call_moderator(patient_summary, r1, r2, patient_id)
            st.session_state.final_report = report
            _bump(
                len(AGENT_ORDER) * 2 + 1,
                "Moderator draft complete",
            )

            status.update(label="RAG validation", state="running")
            st.write("Retrieving similar cases and re-evaluating diagnosis...")
            retriever = get_retriever()
            if retriever is not None and "_parse_error" not in report:
                try:
                    validated = validate_with_rag(report, data, retriever)
                    st.session_state.validated_report = validated
                    st.session_state.rag_examples = validated.get(
                        "_retrieved_cases", []
                    )
                    st.write(
                        f"Validated against {len(st.session_state.rag_examples)} "
                        "retrieved cases."
                    )
                except Exception as exc:  # noqa: BLE001 - surfaced in UI
                    st.warning(f"RAG validation failed, using draft consensus: {exc}")
                    st.session_state.validated_report = None
                    st.session_state.rag_examples = []
            else:
                if "_parse_error" in report:
                    st.info("Skipping RAG validation: draft consensus did not parse.")
                st.session_state.validated_report = None
                st.session_state.rag_examples = []

            st.session_state.debate_complete = True
            _bump(total_steps, "Done")
            status.update(label="MDT debate complete.", state="complete")
        except Exception as exc:  # noqa: BLE001 - surfaced to UI
            status.update(label=f"Debate failed: {exc}", state="error")
            st.exception(exc)
        finally:
            progress_bar.empty()


def _render_patient_tab() -> None:
    data = st.session_state.patient_data
    if not data:
        st.info(
            "No patient data loaded yet. "
            "Use the **Enter Patient Data** tab to fill in a form, "
            "upload a JSON file, or click **Load sample patient** in the sidebar."
        )
        return

    personal = data.get("personal_information", {}) or {}
    vitals = data.get("vitals", {}) or {}
    history = data.get("known_medical_history", []) or []
    symptoms = data.get("current_symptoms", []) or []

    top1, top2, top3 = st.columns(3)
    with top1:
        st.subheader("Patient")
        st.markdown(f"**ID:** `{data.get('patient_id', 'n/a')}`")
        st.markdown(f"**Gender:** {personal.get('gender', 'n/a')}")
        st.markdown(f"**Age:** {personal.get('age', 'n/a')}")
        st.markdown(f"**DOB:** {personal.get('dob', 'n/a')}")
        if data.get("last_visit"):
            st.markdown(f"**Last visit:** {data['last_visit']}")

    with top2:
        st.subheader("Vitals")
        st.markdown(f"- BP: {_safe(vitals.get('blood_pressure'))}")
        st.markdown(
            f"- Pulse: {_safe(vitals.get('pulse'))} "
            f"{_safe(vitals.get('pulse_unit'), '')}"
        )
        st.markdown(
            f"- Resp rate: {_safe(vitals.get('respiratory_rate'))} "
            f"{_safe(vitals.get('respiratory_rate_unit'), '')}"
        )
        st.markdown(
            f"- Temp: {_safe(vitals.get('temperature'))} "
            f"{_safe(vitals.get('temperature_unit'), '')}"
        )
        st.markdown(
            f"- Weight: {_safe(vitals.get('weight'))} "
            f"{_safe(vitals.get('weight_unit'), '')}"
        )
        st.markdown(
            f"- Height: {_safe(vitals.get('height'))} "
            f"{_safe(vitals.get('height_unit'), '')}"
        )

    with top3:
        st.subheader("History & symptoms")
        if history:
            for h in history:
                st.markdown(f"- {h}")
        else:
            st.markdown("_No history recorded._")
        st.markdown("**Current symptoms:**")
        if symptoms:
            for s in symptoms:
                st.markdown(f"- {s}")
        else:
            st.markdown("_None recorded._")

    st.divider()

    st.subheader("Lab results")
    labs = data.get("lab_results", []) or []
    if labs:
        rows: List[Dict[str, Any]] = []
        for entry in labs:
            for analyte, payload in (entry.get("results") or {}).items():
                payload = payload or {}
                rows.append(
                    {
                        "Date": entry.get("date", ""),
                        "Panel": entry.get("cpt_name", ""),
                        "Analyte": analyte,
                        "Result": payload.get("result", ""),
                        "Unit": payload.get("unit", ""),
                        "Normal range": " - ".join(
                            str(x) for x in (payload.get("normal_range") or ["", ""])
                        ).strip(" -"),
                    }
                )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.markdown("_No structured analytes parsed._")
    else:
        st.markdown("_No lab results recorded._")

    st.subheader("Radiology reports")
    radiology = data.get("radiology_reports", []) or []
    if not radiology:
        st.markdown("_None._")
    for r in radiology:
        with st.expander(
            f"{r.get('cpt_name', 'Imaging')} - {r.get('date', 'date n/a')}"
        ):
            if r.get("technique"):
                st.markdown(f"**Technique:** {r['technique']}")
            if r.get("result"):
                st.markdown("**Findings:**")
                st.write(r["result"].strip())
            if r.get("conclusion"):
                st.markdown("**Conclusion:**")
                st.write(r["conclusion"].strip())

    st.subheader("Medications")
    meds = data.get("medications", []) or []
    if meds:
        for m in meds:
            st.markdown(f"- {m}")
    else:
        st.markdown("_None recorded in structured field._")

    st.subheader("Recent encounters")
    encounters = data.get("recent_encounters", []) or []
    if not encounters:
        st.markdown("_None._")
    for e in encounters:
        with st.expander(
            f"{e.get('date', 'date n/a')} - {e.get('clinician', 'clinician n/a')}"
        ):
            st.write((e.get("notes") or "").strip())

    with st.expander("Raw patient JSON"):
        st.json(data)


def _render_round_transcript(
    transcript: List[Dict[str, str]], round_label: str
) -> None:
    if not transcript:
        st.info(f"{round_label} has not been run yet.")
        return
    for entry in transcript:
        with st.expander(f"{entry['agent']} - {entry.get('timestamp', '')}"):
            st.markdown(entry["content"])


def _render_debate_tab() -> None:
    if (
        not st.session_state.round1_transcript
        and not st.session_state.round2_transcript
    ):
        st.info(
            "Run the debate from the sidebar to populate Round 1 and Round 2 transcripts."
        )
        return
    r1_tab, r2_tab = st.tabs(["Round 1 - Initial positions", "Round 2 - Rebuttals"])
    with r1_tab:
        _render_round_transcript(st.session_state.round1_transcript, "Round 1")
    with r2_tab:
        _render_round_transcript(st.session_state.round2_transcript, "Round 2")


def _render_retrieved_cases_panel(cases: List[Dict[str, Any]]) -> None:
    if not cases:
        st.caption("No retrieved cases.")
        return
    for c in cases:
        meta = c.get("metadata", {}) or {}
        header = (
            f"Rank {c.get('rank', '?')} - MRNo {meta.get('mrno', 'n/a')} "
            f"- {meta.get('diagnosis', 'n/a')} (score {c.get('score', 0):.3f})"
        )
        with st.expander(header):
            icd = ", ".join(meta.get("icd_codes") or []) or "n/a"
            st.markdown(f"**ICD-10:** {icd}")
            if meta.get("reasoning"):
                st.markdown(f"**Reasoning:** {meta['reasoning']}")
            st.markdown("**Document:**")
            st.code(c.get("document", ""))


def _render_verification_controls(
    report: Dict[str, Any], patient_data: Dict[str, Any]
) -> None:
    """Approve / Disapprove buttons that gate writing back to the RAG index."""
    st.subheader("Verify result")
    verdict = st.session_state.get("user_verdict")
    persist_msg = st.session_state.get("rag_persist_message")

    if verdict == "approved":
        st.success(persist_msg or "Result approved. Added to RAG corpus.")
        return
    if verdict == "rejected":
        st.warning("Result disapproved. RAG corpus was not updated.")
        return

    st.caption(
        "If you approve, this patient (clinical summary + final diagnosis) is "
        "added to the RAG index so future cases can retrieve it. If you "
        "disapprove, nothing is written."
    )

    btn_l, btn_r = st.columns(2)
    with btn_l:
        approve = st.button(
            "Approve & add to RAG",
            type="primary",
            use_container_width=True,
            key="approve_btn",
        )
    with btn_r:
        disapprove = st.button(
            "Disapprove",
            use_container_width=True,
            key="disapprove_btn",
        )

    if approve:
        retriever = get_retriever()
        if retriever is None:
            st.error("RAG index not available; cannot persist this case.")
            return
        try:
            info = persist_approved_case(patient_data, report, retriever)
            st.session_state.user_verdict = "approved"
            st.session_state.rag_persist_message = (
                f"Added to RAG corpus as MRNo {info['mrno']} "
                f"(diagnosis: {info['diagnosis']})."
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - shown to user
            st.error(f"Failed to add case to RAG corpus: {exc}")
    elif disapprove:
        st.session_state.user_verdict = "rejected"
        st.rerun()


def _render_consensus_tab() -> None:
    draft = st.session_state.final_report
    validated = st.session_state.validated_report
    if not draft and not validated:
        st.info("The moderator has not produced a consensus report yet.")
        return

    using_validated = validated is not None and "_parse_error" not in validated
    report = validated if using_validated else draft

    if report is None or "_parse_error" in report:
        bad = report or {}
        st.error(bad.get("_parse_error", "Consensus report unavailable."))
        with st.expander("Raw moderator response"):
            st.code(bad.get("raw_response", ""))
        return

    if using_validated:
        st.success(
            "Showing RAG-validated consensus (k-shot grounded against retrieved "
            "similar cases)."
        )
        if draft and "_parse_error" not in draft:
            try:
                draft_conf = int(
                    (draft.get("diagnosis") or {}).get("confidence", 0) or 0
                )
                final_conf = int(
                    (report.get("diagnosis") or {}).get("confidence", 0) or 0
                )
                delta = final_conf - draft_conf
                draft_dx = (draft.get("diagnosis") or {}).get("primary_diagnosis", "")
                final_dx = (report.get("diagnosis") or {}).get("primary_diagnosis", "")
                if draft_dx and final_dx and draft_dx != final_dx:
                    st.caption(
                        f"Diagnosis revised after RAG: '{draft_dx}' -> '{final_dx}'."
                    )
                if delta != 0:
                    st.caption(
                        f"Confidence change after RAG: {draft_conf}% -> "
                        f"{final_conf}% ({delta:+d})."
                    )
            except (TypeError, ValueError):
                pass
    else:
        st.warning(
            "Showing draft consensus only - RAG validation was unavailable or "
            "skipped."
        )

    diag = report.get("diagnosis", {}) or {}
    plan = report.get("treatment_plan", {}) or {}
    surgery = plan.get("surgery", {}) or {}
    pharma = plan.get("pharmacotherapy", {}) or {}
    cost = report.get("cost_analysis", {}) or {}

    head_l, head_r = st.columns(2)
    with head_l:
        st.markdown(f"**Patient ID:** `{report.get('patient_id', 'n/a')}`")
    with head_r:
        st.markdown(f"**Date of MDT:** {report.get('date_of_mdt', 'n/a')}")

    st.subheader("Diagnosis")
    primary_col, conf_col = st.columns([3, 1])
    with primary_col:
        st.markdown(f"**Primary:** {diag.get('primary_diagnosis', 'n/a')}")
        st.caption(diag.get("evidence_summary", ""))
    with conf_col:
        try:
            conf_val = int(diag.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf_val = 0
        st.metric("Confidence", f"{conf_val}%")

    differentials = diag.get("differentials") or []
    if differentials:
        st.markdown("**Differentials:**")
        for d in differentials:
            st.markdown(
                f"- {d.get('diagnosis', 'n/a')} " f"({d.get('confidence', 'n/a')}%)"
            )

    st.subheader("Treatment plan")
    st.markdown("**Recommendations:**")
    for i, step in enumerate(plan.get("recommendations") or [], 1):
        st.markdown(f"{i}. {step}")
    if plan.get("alternative_plan"):
        st.markdown(f"**Alternative plan:** {plan['alternative_plan']}")

    surg_col, pharm_col = st.columns(2)
    with surg_col:
        st.markdown("#### Surgery")
        st.markdown(f"- Necessary: **{surgery.get('necessary', 'n/a')}**")
        st.markdown(f"- Procedure: {surgery.get('procedure', 'n/a')}")
        st.markdown(f"- Urgency: {surgery.get('urgency', 'n/a')}")
        risks = surgery.get("risks") or []
        if risks:
            st.markdown("- Risks: " + ", ".join(risks))
    with pharm_col:
        st.markdown("#### Pharmacotherapy")
        st.markdown(f"- Regimen: {pharma.get('regimen', 'n/a')}")
        st.markdown(f"- Duration (weeks): {pharma.get('duration_weeks', 'n/a')}")
        st.markdown(f"- Monitoring: {pharma.get('monitoring', 'n/a')}")

    st.subheader("Cost analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Estimated total (USD)",
            f"${cost.get('estimated_total_usd', 'n/a')}",
        )
    with c2:
        st.metric(
            "Coverage likelihood",
            str(cost.get("insurance_coverage_likelihood", "n/a")),
        )
    with c3:
        st.metric(
            "Out-of-pocket (USD)",
            f"${cost.get('out_of_pocket_estimate_usd', 'n/a')}",
        )
    if cost.get("cost_effectiveness_note"):
        st.caption(cost["cost_effectiveness_note"])

    st.subheader("Unresolved issues")
    unresolved = report.get("unresolved_issues") or []
    if unresolved:
        for q in unresolved:
            st.markdown(f"- {q}")
    else:
        st.markdown("_None._")

    if report.get("agent_consensus_notes"):
        st.subheader("Agent consensus notes")
        st.write(report["agent_consensus_notes"])

    if using_validated and report.get("rag_validation_notes"):
        st.subheader("RAG validation notes")
        st.write(report["rag_validation_notes"])

    rag_cases = st.session_state.get("rag_examples") or []
    if rag_cases:
        with st.expander(f"Retrieved similar cases ({len(rag_cases)})"):
            _render_retrieved_cases_panel(rag_cases)

    if st.session_state.get("patient_data"):
        _render_verification_controls(report, st.session_state.patient_data)

    with st.expander("Raw consensus JSON"):
        st.json({k: v for k, v in report.items() if not k.startswith("_")})

    public_report = {k: v for k, v in report.items() if not k.startswith("_")}
    json_bytes = json.dumps(public_report, indent=2).encode("utf-8")
    md_bytes = report_to_markdown(public_report).encode("utf-8")
    pid = public_report.get("patient_id", "patient")
    dl_l, dl_r = st.columns(2)
    with dl_l:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name=f"mdt_report_{pid}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_r:
        st.download_button(
            "Download Markdown",
            data=md_bytes,
            file_name=f"mdt_report_{pid}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _fk(section: str, idx: int, field: str) -> str:
    """Build a unique widget key for the patient entry form."""
    return f"pf_{section}_{idx}_{field}"


def _render_patient_form_tab() -> None:
    """Render a structured form that builds patient_data from scratch."""

    st.subheader("Enter patient data manually")
    st.caption(
        "Fill in the fields below and click **Apply patient data** at the bottom. "
        "All sections are optional except Patient ID. "
        "You can also upload a JSON or load the sample patient from the sidebar."
    )

    # ------------------------------------------------------------------
    # 1. Patient identity & demographics
    # ------------------------------------------------------------------
    st.markdown("#### Patient identity")
    col_id, col_gender, col_dob, col_age = st.columns([1, 1, 1, 1])
    with col_id:
        patient_id = st.text_input("Patient ID", value="1", key="pf_patient_id")
    with col_gender:
        gender = st.selectbox(
            "Gender",
            ["", "Female", "Male", "Other", "Unknown"],
            key="pf_gender",
        )
    with col_dob:
        dob = st.text_input("Date of birth (YYYY-MM-DD)", value="", key="pf_dob")
    with col_age:
        age_raw = st.text_input("Age (years)", value="", key="pf_age")

    # ------------------------------------------------------------------
    # 2. Vitals
    # ------------------------------------------------------------------
    st.markdown("#### Vitals")
    v1, v2, v3 = st.columns(3)
    with v1:
        bp = st.text_input("Blood pressure (e.g. 110/70)", value="", key="pf_bp")
        pulse = st.text_input("Pulse (/min)", value="", key="pf_pulse")
    with v2:
        temp = st.text_input("Temperature", value="", key="pf_temp")
        temp_unit = st.selectbox("Temp unit", ["C", "F"], key="pf_temp_unit")
    with v3:
        rr = st.text_input("Respiratory rate (/min)", value="", key="pf_rr")
        wt_col, ht_col = st.columns(2)
        with wt_col:
            weight = st.text_input("Weight", value="", key="pf_weight")
        with ht_col:
            height = st.text_input("Height", value="", key="pf_height")
    wunit_col, hunit_col = st.columns(2)
    with wunit_col:
        weight_unit = st.selectbox("Weight unit", ["kg", "lb"], key="pf_weight_unit")
    with hunit_col:
        height_unit = st.selectbox("Height unit", ["cm", "in"], key="pf_height_unit")

    # ------------------------------------------------------------------
    # 3. Medical history & symptoms
    # ------------------------------------------------------------------
    st.markdown("#### Medical history & current symptoms")
    hist_col, symp_col = st.columns(2)
    with hist_col:
        history_raw = st.text_area(
            "Known medical history (one entry per line)",
            height=100,
            placeholder="Known case of diabetes, hypertension ...",
            key="pf_history",
        )
    with symp_col:
        symptoms_raw = st.text_area(
            "Current symptoms (one per line)",
            height=100,
            placeholder="Shortness of breath\nProductve cough",
            key="pf_symptoms",
        )

    # ------------------------------------------------------------------
    # 4. Lab results (dynamic rows)
    # ------------------------------------------------------------------
    st.markdown("#### Lab results")
    st.caption("Each row is one analyte result. Leave blank rows to skip them.")
    add_lab, rem_lab = st.columns([1, 1])
    with add_lab:
        if st.button("+ Add lab row", key="pf_add_lab"):
            st.session_state.form_lab_count += 1
            st.rerun()
    with rem_lab:
        if st.session_state.form_lab_count > 0:
            if st.button("- Remove last lab row", key="pf_rem_lab"):
                st.session_state.form_lab_count = max(
                    0, st.session_state.form_lab_count - 1
                )
                st.rerun()

    for i in range(st.session_state.form_lab_count):
        with st.container(border=True):
            lc1, lc2, lc3, lc4, lc5 = st.columns([2, 2, 1, 1, 1])
            with lc1:
                st.text_input(
                    "Panel / test name",
                    placeholder="ESR (Erythrocytes Sedimentation Rate)",
                    key=_fk("lab", i, "panel"),
                )
            with lc2:
                st.text_input(
                    "Analyte name",
                    placeholder="ESR",
                    key=_fk("lab", i, "analyte"),
                )
            with lc3:
                st.text_input(
                    "Result",
                    placeholder="104",
                    key=_fk("lab", i, "result"),
                )
            with lc4:
                st.text_input(
                    "Unit",
                    placeholder="mm/hr",
                    key=_fk("lab", i, "unit"),
                )
            with lc5:
                st.text_input(
                    "Date (YYYY-MM-DD)",
                    placeholder="2025-12-02",
                    key=_fk("lab", i, "date"),
                )

    # ------------------------------------------------------------------
    # 5. Radiology reports (dynamic rows)
    # ------------------------------------------------------------------
    st.markdown("#### Radiology reports")
    add_rad, rem_rad = st.columns([1, 1])
    with add_rad:
        if st.button("+ Add radiology row", key="pf_add_rad"):
            st.session_state.form_rad_count += 1
            st.rerun()
    with rem_rad:
        if st.session_state.form_rad_count > 0:
            if st.button("- Remove last radiology row", key="pf_rem_rad"):
                st.session_state.form_rad_count = max(
                    0, st.session_state.form_rad_count - 1
                )
                st.rerun()

    for i in range(st.session_state.form_rad_count):
        with st.container(border=True):
            rc1, rc2 = st.columns([3, 1])
            with rc1:
                st.text_input(
                    "Study name",
                    placeholder="C.T. Chest High Resolution (HR Chest)",
                    key=_fk("rad", i, "name"),
                )
            with rc2:
                st.text_input(
                    "Date",
                    placeholder="2026-03-09",
                    key=_fk("rad", i, "date"),
                )
            st.text_area(
                "Technique",
                placeholder="Multiple axial sections through chest without IV contrast (HRCT protocol).",
                height=60,
                key=_fk("rad", i, "technique"),
            )
            st.text_area(
                "Findings",
                placeholder="Air trapping with bronchial wall thickening in bilateral lung fields ...",
                height=80,
                key=_fk("rad", i, "result"),
            )
            st.text_area(
                "Conclusion / impression",
                placeholder="Findings suggestive of inflammatory/infectious process ...",
                height=80,
                key=_fk("rad", i, "conclusion"),
            )

    # ------------------------------------------------------------------
    # 6. Medications (dynamic rows)
    # ------------------------------------------------------------------
    st.markdown("#### Current medications")
    add_med, rem_med = st.columns([1, 1])
    with add_med:
        if st.button("+ Add medication row", key="pf_add_med"):
            st.session_state.form_med_count += 1
            st.rerun()
    with rem_med:
        if st.session_state.form_med_count > 0:
            if st.button("- Remove last medication row", key="pf_rem_med"):
                st.session_state.form_med_count = max(
                    0, st.session_state.form_med_count - 1
                )
                st.rerun()

    for i in range(st.session_state.form_med_count):
        with st.container(border=True):
            mc1, mc2, mc3 = st.columns([2, 1, 1])
            with mc1:
                st.text_input(
                    "Drug name",
                    placeholder="Isoniazid",
                    key=_fk("med", i, "name"),
                )
            with mc2:
                st.text_input(
                    "Dose",
                    placeholder="300 mg",
                    key=_fk("med", i, "dose"),
                )
            with mc3:
                st.text_input(
                    "Frequency",
                    placeholder="once daily",
                    key=_fk("med", i, "frequency"),
                )

    # ------------------------------------------------------------------
    # 7. Recent encounters (dynamic rows)
    # ------------------------------------------------------------------
    st.markdown("#### Recent encounters")
    add_enc, rem_enc = st.columns([1, 1])
    with add_enc:
        if st.button("+ Add encounter row", key="pf_add_enc"):
            st.session_state.form_enc_count += 1
            st.rerun()
    with rem_enc:
        if st.session_state.form_enc_count > 0:
            if st.button("- Remove last encounter row", key="pf_rem_enc"):
                st.session_state.form_enc_count = max(
                    0, st.session_state.form_enc_count - 1
                )
                st.rerun()

    for i in range(st.session_state.form_enc_count):
        with st.container(border=True):
            ec1, ec2 = st.columns([2, 2])
            with ec1:
                st.text_input(
                    "Date (ISO)",
                    placeholder="2025-12-03T12:07:43+00:00",
                    key=_fk("enc", i, "date"),
                )
            with ec2:
                st.text_input(
                    "Clinician",
                    placeholder="Dr. Smith",
                    key=_fk("enc", i, "clinician"),
                )
            st.text_area(
                "Notes",
                placeholder="Patient seen. Vitally stable ...",
                height=90,
                key=_fk("enc", i, "notes"),
            )

    # ------------------------------------------------------------------
    # Submit: assemble JSON from all widgets and set session state
    # ------------------------------------------------------------------
    st.divider()
    if st.button(
        "Apply patient data",
        type="primary",
        use_container_width=True,
        key="pf_submit",
    ):
        pid = (st.session_state.get("pf_patient_id") or "").strip() or "1"

        # personal information
        try:
            age_val: Any = int(age_raw.strip()) if age_raw.strip() else None
        except ValueError:
            age_val = age_raw.strip() or None

        personal = {
            "name": "",
            "gender": gender or "",
            "dob": (dob or "").strip(),
            "age": age_val,
        }

        # vitals — convert numeric strings to floats where possible
        def _to_float_or_none(s: str) -> Any:
            try:
                return float(s.strip())
            except (ValueError, AttributeError):
                return None if not s.strip() else s.strip()

        vitals: Dict[str, Any] = {
            "blood_pressure": bp.strip() or None,
            "temperature": _to_float_or_none(temp),
            "temperature_unit": temp_unit,
            "pulse": _to_float_or_none(pulse),
            "pulse_unit": "/min",
            "respiratory_rate": _to_float_or_none(rr),
            "respiratory_rate_unit": "/min",
            "weight": _to_float_or_none(weight),
            "weight_unit": weight_unit,
            "height": _to_float_or_none(height),
            "height_unit": height_unit,
            "timestamp": None,
        }

        # history & symptoms (one per line, filter blanks)
        history_list = [l.strip() for l in history_raw.splitlines() if l.strip()]
        symptoms_list = [l.strip() for l in symptoms_raw.splitlines() if l.strip()]

        # lab results
        lab_results: List[Dict[str, Any]] = []
        for i in range(st.session_state.form_lab_count):
            analyte = (st.session_state.get(_fk("lab", i, "analyte")) or "").strip()
            result_str = (st.session_state.get(_fk("lab", i, "result")) or "").strip()
            if not analyte and not result_str:
                continue
            try:
                result_val: Any = float(result_str) if result_str else None
            except ValueError:
                result_val = result_str or None
            panel = (
                st.session_state.get(_fk("lab", i, "panel")) or ""
            ).strip() or analyte
            lab_results.append(
                {
                    "cpt_id": f"manual_{i:04d}",
                    "cpt_name": panel,
                    "date": (st.session_state.get(_fk("lab", i, "date")) or "").strip()
                    or None,
                    "results": {
                        analyte
                        or "result": {
                            "result": result_val,
                            "unit": (
                                st.session_state.get(_fk("lab", i, "unit")) or ""
                            ).strip(),
                            "normal_range": ["", ""],
                        }
                    },
                }
            )

        # radiology reports
        radiology: List[Dict[str, Any]] = []
        for i in range(st.session_state.form_rad_count):
            name = (st.session_state.get(_fk("rad", i, "name")) or "").strip()
            findings = (st.session_state.get(_fk("rad", i, "result")) or "").strip()
            conclusion = (
                st.session_state.get(_fk("rad", i, "conclusion")) or ""
            ).strip()
            if not name and not findings and not conclusion:
                continue
            radiology.append(
                {
                    "cpt_id": f"manual_rad_{i:04d}",
                    "cpt_name": name or "Imaging",
                    "technique": (
                        st.session_state.get(_fk("rad", i, "technique")) or ""
                    ).strip(),
                    "result": findings,
                    "conclusion": conclusion,
                    "system_conclusion": "",
                    "file_path": "",
                    "date": (st.session_state.get(_fk("rad", i, "date")) or "").strip()
                    or None,
                }
            )

        # medications
        medications: List[Dict[str, str]] = []
        for i in range(st.session_state.form_med_count):
            drug = (st.session_state.get(_fk("med", i, "name")) or "").strip()
            if not drug:
                continue
            medications.append(
                {
                    "name": drug,
                    "dose": (st.session_state.get(_fk("med", i, "dose")) or "").strip(),
                    "frequency": (
                        st.session_state.get(_fk("med", i, "frequency")) or ""
                    ).strip(),
                }
            )

        # recent encounters
        encounters: List[Dict[str, str]] = []
        for i in range(st.session_state.form_enc_count):
            notes = (st.session_state.get(_fk("enc", i, "notes")) or "").strip()
            enc_date = (st.session_state.get(_fk("enc", i, "date")) or "").strip()
            clinician = (st.session_state.get(_fk("enc", i, "clinician")) or "").strip()
            if not notes and not clinician:
                continue
            encounters.append(
                {
                    "date": enc_date or None,
                    "clinician": clinician,
                    "notes": notes,
                    "symptoms": [],
                }
            )

        patient_data: Dict[str, Any] = {
            "patient_id": pid,
            "personal_information": personal,
            "vitals": vitals,
            "known_medical_history": history_list,
            "current_symptoms": symptoms_list,
            "lab_results": lab_results,
            "radiology_reports": radiology,
            "medications": medications,
            "recent_encounters": encounters,
            "last_visit": encounters[0].get("date") if encounters else None,
        }

        st.session_state.patient_data = patient_data
        st.session_state.patient_source = "manual form"
        _reset_debate_state()
        st.success(
            f"Patient data applied (ID: `{pid}`). "
            "Switch to the **Patient Summary** tab to review, "
            "then run the debate from the sidebar."
        )


def main() -> None:
    _init_session_state()

    st.title("AI Clinical MDT Simulator")
    st.markdown(
        "Six clinical AI agents (Radiologist, Pathologist, Oncologist, "
        "Surgeon, Pharmacist, Insurance/Cost Agent) hold a structured "
        "two-round debate on the uploaded patient case, then a moderator "
        "produces a consensus report. "
        f"Model: `{MODEL}`."
    )

    _render_sidebar()

    entry_tab, summary_tab, debate_tab, consensus_tab = st.tabs(
        [
            "Enter Patient Data",
            "Patient Summary",
            "Debate Transcript",
            "Consensus Report",
        ]
    )
    with entry_tab:
        _render_patient_form_tab()
    with summary_tab:
        _render_patient_tab()
    with debate_tab:
        _render_debate_tab()
    with consensus_tab:
        _render_consensus_tab()


if __name__ == "__main__":
    main()
