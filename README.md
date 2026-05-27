# LLM Evaluation Platform

A full-stack platform to evaluate LLM-based applications using industry frameworks.

## Stack

| Layer      | Tech                                      |
|------------|-------------------------------------------|
| Frontend   | React (Vite), plain CSS-in-JS             |
| Backend    | FastAPI (Python), session-based auth      |
| Database   | PostgreSQL                                |
| Evaluation | DeepEval / Arize Phoenix / TruLens / Opik |

---

## 1. Database Setup

```bash
psql -U postgres

CREATE DATABASE evaluation_db;
\c evaluation_db
\i database/schema.sql
```

---

## 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

## 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will run at `http://localhost:5173`

---

## 4. User Flow

1. **Register / Login** — Create account or sign in
2. **Dashboard** — View all your projects and previous runs
3. **Create Project** — 3-step wizard:
   - Step 1: Project details (name, language, LLM model, etc.)
   - Step 2: Choose evaluation framework
   - Step 3: Configure metrics (relevancy, faithfulness, custom)
4. **API Endpoint** — A unique URL is generated for your project
5. **Evaluate** — POST either your conversation data or your chatbot message to the endpoint
6. **View Results** — Dashboard link appears on your project page after evaluation completes

---

## 5. Evaluation API

### Endpoint
```
POST http://localhost:8000/evaluate/{framework}/{your-api-key}
```

### Option 1: Direct evaluation payload
```json
{
  "session_label": "my-test-run",
  "conversation": [
    {
      "user_input": "What is the capital of France?",
      "llm_output": "The capital of France is Paris.",
      "context": ["France is a country in Western Europe. Its capital is Paris."],
      "expected_output": "Paris"
    }
  ]
}
```

### Option 2: Chatbot proxy payload
Use this when you want your app or chatbot to call the generated evaluation URL directly. The backend will:
1. Forward the request to the chatbot API
2. Capture the chatbot reply
3. Start evaluation using the framework selected for that project

```json
{
  "message": "Hi there, what is the capital of India?",
  "reset": false
}
```

Optional fields supported in proxy mode:
- `session_label`
- `context`
- `expected_output`

### Response
```json
{
  "response": "The capital of India is New Delhi.",
  "evaluation": {
    "run_id": "uuid-here",
    "status": "pending",
    "framework": "arize_phoenix",
    "status_url": "http://localhost:8000/evaluate/run/uuid-here/status"
  }
}
```

### Poll Status
```
GET http://localhost:8000/evaluate/run/{run_id}/status
```

---

## 6. Adding Evaluation Logic

Each framework has a stub file in `backend/evaluators/`:

| File                        | Framework     |
|-----------------------------|---------------|
| `deepeval_runner.py`        | DeepEval      |
| `arize_phoenix_runner.py`   | Arize Phoenix |
| `trulens_runner.py`         | TruLens       |
| `opik_runner.py`            | Opik (Comet)  |

Each file has:
- Detailed comments with example implementation
- A `run_*()` async function that accepts `(run_id, project_id, config, payload)`
- Must return `(dashboard_url: str, external_run_id: str)`

---

## 7. Project Structure

```
eval-platform/
├── database/
│   └── schema.sql              # Full PostgreSQL schema
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt
│   ├── core/
│   │   ├── config.py           # DB config & settings
│   │   └── auth.py             # Session auth utilities
│   ├── routers/
│   │   ├── auth.py             # Register / Login / Logout
│   │   ├── projects.py         # Project CRUD
│   │   └── evaluate.py         # Evaluation endpoint
│   └── evaluators/
│       ├── deepeval_runner.py
│       ├── arize_phoenix_runner.py
│       ├── trulens_runner.py
│       └── opik_runner.py
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── api/
        │   └── client.js       # All API calls
        ├── context/
        │   └── AuthContext.jsx # Session state
        └── pages/
            ├── AuthPage.jsx          # Login / Register
            ├── DashboardPage.jsx     # Project list
            ├── CreateProjectPage.jsx # 3-step wizard
            ├── ApiKeyPage.jsx        # Endpoint display + docs
            └── ProjectDetailPage.jsx # Runs + dashboard links
```


# deepeval Evaluation Platform

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd EvaluationFrameworkBackend
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Fill in `.env`
```bash
cp .env.example .env
# Edit .env with your actual values
```

### 5. Run setup script
```bash
chmod +x setup.sh && ./setup.sh
```

### 6. Set up Confident AI Dashboard (DeepEval)
Get your free API key from **https://app.confident-ai.com → Settings → API Keys**

Add to `.env`:
```
CONFIDENT_AI_API_KEY=your_key_here
```

Then register it with DeepEval:
```bash
source venv/bin/activate
python3 setup_confident_ai.py
```

### 7. Start the server
```bash
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

---

## How the Confident AI Dashboard works

Every time you run an evaluation via `POST /evaluate/deepeval/{unique_id}`, results are automatically pushed to Confident AI. The response includes a `dashboard_url` that opens directly to that test run:

```json
{
  "dashboard_url": "https://app.confident-ai.com/test-runs/...",
  "status": "completed"
}
```

Each team member needs their own Confident AI account and API key — results are tied to the account whose key is in `.env`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DB_NAME` | ✅ | PostgreSQL database name |
| `DB_USER` | ✅ | PostgreSQL username |
| `DB_PASSWORD` | ✅ | PostgreSQL password |
| `DB_HOST` | ✅ | Database host |
| `DB_PORT` | ✅ | Database port |
| `SECRET_KEY` | ✅ | JWT secret key |
| `VERTEX_API_KEY` | ✅ | Vertex AI proxy API key |
| `VERTEX_API_BASE` | ✅ | Vertex AI proxy base URL |
| `CONFIDENT_AI_API_KEY` | ✅ | Confident AI API key for DeepEval dashboard |
