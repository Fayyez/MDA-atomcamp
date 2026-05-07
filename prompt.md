# Prompt: Implement AI Clinical MDT (Multi-Disciplinary Team) Simulator – Updated

## Project Goal
Build a **Streamlit** web application that uses the **OpenAI API** (model: `gpt-4o-mini`) to simulate a **detailed, multi‑round debate** between six clinical AI agents. The simulation mimics a real hospital MDT meeting for complex cases (oncology, cardiology, infectious disease, etc.). The application accepts a patient data JSON (lab results, imaging summaries, symptoms, notes, vitals, history) and orchestrates a rich, evidence‑based discussion among agents, ending with a structured consensus report.

**Key update**: The OpenAI API key is **read from environment variables** (`.env` file or system env) – **not** requested from the user in the UI.  
**Model**: `gpt-4o-mini` (cost‑effective, high detail).  
**Debate requirement**: Each agent must produce a **detailed, clinical‑grade** response (minimum 3‑5 sentences, with reasoning, citations of data, and explicit confidence levels).

## Provided Context
- You have an OpenAI API key stored in `OPENAI_API_KEY` environment variable.
- Patient data follows the given JSON schema (fields may vary, but key sections like `radiology_reports`, `lab_results`, `vitals`, `known_medical_history` are expected).
- The user will upload a JSON file via Streamlit. No manual key input is required.

## Technical Requirements
- **Language**: Python 3.9+
- **Libraries**: `streamlit`, `openai`, `json`, `datetime`, `typing`, `pydantic` (optional), `python-dotenv`, `markdown`
- **Model**: `openai.ChatCompletion` with `model="gpt-4o-mini"`
- **Temperature**: 0.2 for factual clinical reasoning (allow slight variability but avoid hallucinations)
- **UI**: Streamlit (file uploader, tabs, expanders, download buttons, progress indicators)

## Agent Roles & Detailed Prompts (System Messages)
Each agent has a **system prompt** that enforces detailed, evidence‑grounded reasoning. All agents must explicitly reference specific data fields from the uploaded JSON.

| Agent | System Prompt (core instructions) |
|-------|-----------------------------------|
| **Radiologist** | You are a senior radiologist. Interpret all `radiology_reports` in detail. Describe key findings (e.g., tree‑in‑bud pattern, lymphadenopathy, bronchiectasis). Give a differential diagnosis based on imaging alone. Rate your diagnostic confidence (0‑100%). Suggest additional imaging if needed. |
| **Pathologist** | You are a pathologist with expertise in lab data and histology. Analyze `lab_results` (abnormal values, trends, normal ranges), and any available cytology/histology (if present in the JSON). Correlate with `known_medical_history`. Provide a cellular/inflammatory/molecular interpretation. List missing tests that would help. |
| **Oncologist** | You are a consultant oncologist. Evaluate malignancy risk from all available data (imaging, labs, symptoms). If cancer is unlikely, state so and explain why. If suspicious, propose staging, biomarkers, and systemic therapy options. Always comment on treatment urgency and expected outcomes. Use explicit confidence. |
| **Surgeon** | You are a cardiothoracic/general surgeon. Assess the need for invasive procedures: biopsy, resection, drainage, etc. Consider patient vitals (`vitals`), surgical risks, and alternative diagnostic steps. State clearly if surgery is necessary, elective, or not indicated. Provide a risk‑benefit table in prose. |
| **Pharmacist** | You are a clinical pharmacist. Review `medications` (current) and any proposed drugs from other agents. Check interactions, dosing adjustments (renal/hepatic – infer from labs/vitals), adverse effects, cost‑effective alternatives, and monitoring parameters. Recommend a complete medication plan with rationales. |
| **Insurance/Cost Agent** | You are a healthcare financial advisor. Estimate the total cost of the proposed diagnostic and treatment pathway (consultation, procedures, drugs, follow‑up). Provide a payer coverage likelihood (e.g., low/medium/high) for each major component. Suggest lower‑cost alternatives without compromising quality. Quantify out‑of‑pocket ranges where possible. |

## Debate Flow – Detailed & Multi‑Round
To ensure depth, implement **two structured rounds** plus a **moderator synthesis**.

### Round 1 – Initial Position (Sequential)
- Order: Radiologist → Pathologist → Oncologist → Surgeon → Pharmacist → Insurance/Cost Agent
- For each agent:
  - Build a message list containing:
    - The agent’s system prompt.
    - A structured summary of the patient data relevant to that agent (extracted programmatically).
    - The full transcripts of all previous agents in the current round (so they can build on earlier points).
  - Call OpenAI with `model="gpt-4o-mini"`, `temperature=0.2`, `max_tokens=800`.
  - Append the agent’s response to session state (as a transcript with role, agent name, timestamp).

### Round 2 – Rebuttal & Deepening (Optional but highly recommended)
- After Round 1, **automatically** trigger a second round in the same order.
- In Round 2, each agent is given the **full transcript of Round 1** (all agents) and is asked to:
  - Respond to at least one specific point made by another agent (e.g., “I disagree with the oncologist’s suggestion of biopsy because…”).
  - Refine their own previous recommendation with more precise numbers (e.g., confidence updated from 70% to 85% after seeing pathologist’s lab interpretation).
  - Identify one unresolved question and propose how to answer it.
- Temperature for Round 2: `0.25` (slightly more creative for counterarguments).
- Store Round 2 responses in session state under a separate key.

### Moderator / Consensus Synthesis
- After both rounds, call a **Moderator** agent with system prompt:
  ```
  You are the MDT chairperson. Synthesize the detailed debate from all six agents (both rounds). Produce a structured JSON output as defined below. Resolve conflicts by weighing evidence, and if no consensus exists, state the disagreement explicitly. Provide final diagnosis confidence, a prioritized treatment plan, surgery necessity, cost‑risk analysis, and any unresolved issues.
  ```
- Use `response_format={"type": "json_object"}` if available (OpenAI API), otherwise parse JSON from the response.
- The final JSON must follow the schema provided in the next section.

## Output Schema (Final Consensus Report)
The moderator must output exactly this structure (or as close as possible). The Streamlit UI will parse and display it.

```json
{
  "patient_id": "string",
  "date_of_mdt": "YYYY-MM-DD",
  "diagnosis": {
    "primary_diagnosis": "detailed name",
    "confidence": 0-100,
    "differentials": [
      {"diagnosis": "name", "confidence": 0-100}
    ],
    "evidence_summary": "Key imaging/lab/history findings supporting primary diagnosis"
  },
  "treatment_plan": {
    "recommendations": ["step 1", "step 2", ...],
    "alternative_plan": "description",
    "surgery": {
      "necessary": true/false,
      "procedure": "name or null",
      "urgency": "elective/urgent/emergent/none",
      "risks": ["risk1", "risk2"]
    },
    "pharmacotherapy": {
      "regimen": "drug names + doses",
      "duration_weeks": number,
      "monitoring": "what labs/follow-up"
    }
  },
  "cost_analysis": {
    "estimated_total_usd": number,
    "insurance_coverage_likelihood": "low/medium/high",
    "out_of_pocket_estimate_usd": number,
    "cost_effectiveness_note": "string"
  },
  "unresolved_issues": ["question1", "question2"],
  "agent_consensus_notes": "Summary of agreements and major disagreements from Round 2"
}
```

## Streamlit UI Implementation Details

### Environment Setup
- Use `python-dotenv` to load `.env` file containing `OPENAI_API_KEY=your-key-here`.
- If key is missing, `st.error("OpenAI API key not found in environment")` and exit gracefully.
- No sidebar input for API key.

### Main Page Layout
- **Header**: Title, brief description.
- **Sidebar**:
  - File uploader (`.json`) – load patient data.
  - Expandable JSON editor to modify loaded data (optional).
  - Button: **Start Detailed MDT Debate** (only enabled when data loaded).
- **Main Area** with tabs:
  1. **Patient Summary** – formatted display of vitals, labs, radiology, history, medications.
  2. **Debate Transcript** – two sub‑tabs: Round 1 and Round 2. Each shows expanders for every agent’s full response.
  3. **Consensus Report** – nicely formatted JSON with cards for diagnosis, treatment, cost. Download buttons (JSON, Markdown).
- **Progress** – show spinner and progress bar during API calls (Rounds 1 & 2 can take 10‑20 seconds; inform user).

### State Management (`st.session_state`)
- `patient_data`: dict (loaded JSON)
- `round1_transcript`: list of dicts `{"agent": str, "content": str, "timestamp": str}`
- `round2_transcript`: list of dicts (same structure)
- `final_report`: dict (moderator output)
- `debate_complete`: bool

### API Call Helpers
- `call_agent(messages, model="gpt-4o-mini", temperature=0.2, max_tokens=800) -> str`
  - Includes retry logic (max 2 retries, exponential backoff).
  - Logs errors to `st.error` but does not crash whole app.
- For moderator: `call_moderator(transcript_round1, transcript_round2) -> dict`

## Detailed Debate Requirements (for each agent)
To ensure the debate is **detailed**, each agent’s system prompt must explicitly request:
- Minimum 3 clinical findings referenced by name and value.
- Explicit confidence levels (e.g., “80% confident of MAC due to tree‑in‑bud pattern”).
- If data is missing, state “insufficient data to conclude X, recommend Y test”.
- For the Insurance Agent: include approximate drug costs (can use public US average wholesale prices or international references). For example: “Azithromycin 250mg daily: ~$10/month”.

### Example of expected detail (Radiologist, Round 1):
> “The HRCT chest shows extensive tree‑in‑bud nodules in all lobes, air trapping, and mild cylindrical bronchiectasis in lower lobes. Mediastinal and left hilar lymphadenopathy (short axis ~12mm). These findings are highly suggestive of an infectious bronchiolitis, with nontuberculous mycobacteria (NTM) being most likely (confidence 75%). Tuberculosis is a key differential (20%), and less likely hypersensitivity pneumonitis (5%). I recommend sputum AFB culture and PCR for MAC.”

## Integration with Provided Sample JSON
Given the sample patient (60F, diabetes, hypertension, rheumatoid arthritis, shortness of breath, HRCT with tree‑in‑bud, ESR 104, Vit D 9.1, already started ATT and steroids):
- The debate should reveal that ATT was prescribed empirically for TB, but the pattern is more consistent with MAC.
- The pharmacist must flag the ongoing insulin, steroids, and ATT → risk of hyperglycemia, hepatotoxicity.
- The surgeon should note that no lung surgery is indicated; however, the toe drainage (mentioned in encounter) is already handled.
- The insurance agent should note that ATT is low‑cost but MAC regimens (azithromycin/ethambutol/rifampin) are also cheap.
- Final report: primary diagnosis = MAC pulmonary disease (confidence 80%), continue ATT? No, switch to MAC regimen, add vitamin D, monitor LFTs.

## Additional Features (Stretch, but encouraged)
- **Cost slider**: Allow user to adjust drug cost assumptions (e.g., by country).
- **Voice summary** (using gTTS) of the final report.
- **Export as PDF** (using `fpdf` or `reportlab`).
- **History of past MDTs** (save to local JSON).

## Acceptance Criteria (Final)
1. The application loads the sample JSON without errors.
2. Clicking “Start Detailed MDT Debate” runs Round 1 and Round 2, showing progress.
3. Each agent’s response is detailed (≥3 sentences, references specific data, includes confidence).
4. The moderator produces a valid JSON matching the schema.
5. The UI displays all rounds and the final report clearly.
6. The API key is **never** exposed or requested in the UI – only from env.
7. The model used is `gpt-4o-mini` (check via `model` parameter).

## Deliverable
- Single file `app.py` with all code.
- `requirements.txt`.
- `.env.example` file (with `OPENAI_API_KEY=your_key_here`).
- Short `README.md` explaining setup and run.

**Now generate the complete code and documentation based on this detailed prompt.**