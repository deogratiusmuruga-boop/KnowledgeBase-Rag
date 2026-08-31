# ElderDocAI-System

An **evidence-grounded, rule-bounded** RAG (Retrieval-Augmented Generation) assistant for elderly care. The system retrieves from a caregiver knowledge base, grounds LLM answers with verifiable evidence, gates output through a reliability/adaptive-decision layer, and is served to users through a React frontend.

> **Note:** `SCI_PAPER_DRAFT.md` contains the research-manuscript draft that previously lived here. This README is the project-level reference for the codebase.

---

## Features

- **Hybrid retrieval** — dense vectors (FAISS) + BM25 lexical search + cross-encoder reranking.
- **Evidence-grounded generation** — LLM answers forced to cite the retrieved SOURCE only (temperature 0).
- **Reliability gating** — accept / refine / re-retrieve / reject policy with configurable scores & thresholds.
- **Adaptive care context** — maps longitudinal records into care-state windows, transitions, and adaptive assistance plans (ElderDocAI).
- **Rule-bounded safety** — deterministic no-diagnosis / no-risk-prediction guardrails in generated plans.
- **User-profile personalization** — locale-aware (ko/en), speech-speed, chronic conditions, medications.
- **REST API** — FastAPI backend + SQLite persistence.
- **React frontend** — Create React App (`react-scripts`), voice/chat UI with a digital-human avatar.

---

## Tech Stack

### Backend
- **Python 3.12/3.14**, **FastAPI 0.140**, **Uvicorn 0.51**, **SQLAlchemy 2.0**, **Pydantic 2.13**
- **sentence-transformers 5.6**, **FAISS 1.14**, **rank-bm25 0.2**, **scikit-learn 1.9**, **torch 2.13**
- **Ollama 0.6** (local LLM, e.g. Llama 3.2)

### Frontend
- **react-scripts** (Create React App), **React 18**, **react-dom**, **lucide-react** (icons)

---

## Repository Layout

```
api/                 FastAPI routers (main, profile, medication, appointment, reminder, deps)
models/              SQLAlchemy models (user, medication, appointment)
scripts/             RAG pipeline + evaluation/build/reliability scripts
config/              runtime configuration
data/                generated indices, text, vector DB, gold-QA set
datasets/            Synthea FHIR patient records
evaluation/          gold-QA / retrieval queries
frontend/            React UI (src/screens, components, services)
tests/               retrieval / BM25 tests
create_tables.py     one-time DB table creation
database.py          SQLAlchemy engine / session / SQLite connection
carebuddy.db         generated SQLite database (git-ignored)
dataset_inventory.md NIA caregiver-manual inventory
SCI_PAPER_DRAFT.md   research manuscript
```

---

## Quickstart

### 1. Backend (Python)

Create a venv and install dependencies. The venv packages are already installed under `venv/` (no `requirements.txt` yet). Required versions: `fastapi==0.140`, `uvicorn==0.51`, `sqlalchemy==2.0.51`, `pydantic==2.13.4`, `sentence-transformers==5.6`, `faiss-cpu==1.14.3`, `rank-bm25==0.2.2`, `scikit-learn==1.9.0`, `torch==2.13`, `ollama==0.6`.

> Recommended: consolidate these into a `requirements.txt` for reproducible rebuilds.

Initialize the SQLite database:
```bash
python create_tables.py
```

Start the API (default port 8000):
```bash
uvicorn api.main:app --reload
```

### 2. Frontend (React / react-scripts)

From `frontend/`:
```bash
npm start
```
Open http://localhost:3000. Point the frontend at the backend via `frontend/.env`:
```
# copy from `.env.example`
REACT_APP_API_URL=http://localhost:8000
```
`src/services/ragApi.js` and `src/services/reminderApi.js` read `process.env.REACT_APP_API_URL`, falling back to `http://127.0.0.1:8000`.

---

## Database Schema (SQLite, FK-enforced)

- `user_profiles` — `id`, `age`, `location`, `chronic_conditions`, `medications`, `preferred_language`, `speech_speed`
- `medications` — `id`, `user_id → user_profiles.id`, `medicine_name`, `dosage`, `time`, `frequency`
- `appointments` — `id`, `user_id → user_profiles.id`, `title`, `appointment_date`, `appointment_time`, `location`, `notes`

Engine URL: `sqlite:///./carebuddy.db`.

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/`                      | health check |
| POST | `/ask`                   | ask the RAG (question, user_id, user_profile, conversation_history) |
| POST | `/profile/`              | create user profile |
| POST | `/medications/`          | save a medication |
| GET  | `/medications/{user_id}` | list a user's medications |
| POST | `/appointments/`         | create appointment |
| GET  | `/appointments/`         | list appointments |
| DELETE | `/appointments/{id}`   | delete appointment |
| GET  | `/reminders/{user_id}`   | merged medication + appointment reminders |

All user-scoped routes validate that the user exists. CORS allows `localhost:5173`, `127.0.0.1:5173`, `localhost:3000`, with `allow_credentials=True`.

**Note:** `/ask` derives response language from the saved profile's `preferred_language` (default `en`); the request-body `language` / `preferred_language` fields are presently ignored by the backend.

---

## Configuration

`config/reliability_config.json`:

```json
{
  "reliability_weights": { "authority": 0.3, "relevance": 0.3, "support": 0.2, "coverage": 0.1, "consistency": 0.1 },
  "decision_thresholds": { "accept": 0.8, "refine": 0.65, "re_retrieve": 0.45 }
}
```

---

## Pipeline (scripts/)

- **Ingest:** `extract_text.py`, `clean_text.py`, `chunk_text.py`
- **Index:** `build_bm25_index.py`, `build_faiss_index.py`, `generate_embeddings.py`
- **Retrieval:** `hybrid_retrieval.py`, `hybrid_retriever.py`, `reranker.py`, `test_retrieval.py`
- **Grounding:** `build_grounded_prompt.py`, `build_context.py`, `evidence_aggregation.py`
- **Reliability / decisions:** `reliability_config.py`, `reliability_evaluation.py`, `adaptive_decision_controller.py`, `authority_mapping.py`
- **Service / chat:** `rag_chat.py`, `carebuddy_service.py`
- **Evaluation:** `evaluate_*` scripts, `data/gold_qa_evaluation.json`

See `dataset_inventory.md` for the NIA/NIH source books composing the caregiver knowledge base.

---

## Tests

Scripts under `tests/`, plus `test_retrieval.py` and `test_bm25.py`. Unit tests are provided under `tests/` (e.g., `tests/test_api.py`), and PASS/FAIL-style evaluation scripts run under `scripts/` (e.g., `evaluate_retrieval.py`, `evaluate_latency.py`, `evaluate_carebuddy.py`).

---

## Status

Internal research prototype. Manuscript draft tracked separately in `SCI_PAPER_DRAFT.md`.