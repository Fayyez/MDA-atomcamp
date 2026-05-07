# AI Clinical MDT Simulator

A Streamlit web app that simulates a hospital **Multi-Disciplinary Team (MDT)**
meeting using six clinical AI agents powered by OpenAI's `gpt-4o-mini`.
Upload a patient JSON, click **Start Detailed MDT Debate**, and the app runs
two structured debate rounds plus a moderator synthesis, producing a
downloadable consensus report (JSON + Markdown).

## Agents

| Agent | Role |
|-------|------|
| Radiologist | Interprets imaging, gives a radiologic differential. |
| Pathologist | Interprets labs, suggests missing tests, correlates with history. |
| Oncologist | Stratifies malignancy risk and proposes work-up / therapy. |
| Surgeon | Decides if invasive procedures are indicated. |
| Pharmacist | Reviews / proposes a medication plan with interactions and monitoring. |
| Insurance / Cost Agent | Estimates costs and coverage likelihood. |

## Debate flow

1. **Round 1 - initial positions**: agents speak in fixed order, each one
   sees all earlier opinions in the round.
2. **Round 2 - rebuttal & deepening**: each agent sees the full Round 1
   transcript and must push back on at least one specific point, refine
   their own confidence, and flag one unresolved question.
3. **Moderator synthesis**: a chairperson agent fuses everything into a
   structured JSON consensus report (diagnosis, treatment plan, surgery,
   pharmacotherapy, cost analysis, unresolved issues).

The model is `gpt-4o-mini` throughout; temperatures are `0.2` for Round 1
and the moderator, and `0.25` for Round 2 to encourage productive
counter-arguments.

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create your `.env` from the template and add your OpenAI key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

The key is loaded by `python-dotenv` at startup. **It is never requested
in the UI.** If `OPENAI_API_KEY` is missing the app shows an error and
exits gracefully.

## Run

```bash
streamlit run app.py
```

Then in the sidebar either:

- click **Load sample patient** to load `data/sample-patient-info.json`, or
- upload your own patient JSON,

and click **Start Detailed MDT Debate**. Round 1 + Round 2 + moderator
typically take 30-60 seconds.

## Patient JSON schema

The app expects fields like `patient_id`, `personal_information`,
`vitals`, `lab_results`, `radiology_reports`, `medications`,
`recent_encounters`, `current_symptoms`, `known_medical_history`. All
fields are optional - missing sections render as "not provided" and the
agents are instructed to recommend follow-up tests when data is missing
rather than hallucinate values. See `data/sample-patient-info.json` for
a full example.

## Output

The **Consensus Report** tab shows the structured moderator output as
cards (diagnosis, treatment plan, surgery, pharmacotherapy, cost) plus
the raw JSON, with download buttons for both `.json` and `.md`.

## Notes

- Uses the **OpenAI Python SDK v1+** (`from openai import OpenAI`). The
  legacy `openai.ChatCompletion` API is no longer supported by recent SDK
  versions.
- API calls have automatic retries with exponential backoff (up to 2
  retries per call).
- The moderator call uses `response_format={"type": "json_object"}` to
  force JSON output, with a fallback regex parser for safety.
- This tool is a research / educational simulator. **It is not a medical
  device and must not be used for actual clinical decision making.**

## Potential extensions (TODO)

- PDF export (e.g. via `reportlab`).
- Voice summary via `gTTS`.
- Country-specific cost slider for the Insurance/Cost agent.
- Persistent local history of past MDTs.

## File layout

```
app.py                      # the whole Streamlit app
requirements.txt            # streamlit + openai + python-dotenv
.env.example                # OPENAI_API_KEY placeholder
data/sample-patient-info.json  # example patient payload
```
