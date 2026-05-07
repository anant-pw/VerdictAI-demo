# VerdictAI — LLM Evaluation Framework

> Systematic, multi-judge evaluation of LLM outputs with hallucination detection, relevance scoring, and regression tracking.

🔗 **[Live Demo →](https://YOUR_APP.streamlit.app)** &nbsp;|&nbsp; Built with Python · LangChain · Groq · Streamlit · SQLite

---

## What it does

VerdictAI replaces subjective "does this look right?" prompt testing with structured, repeatable LLM evaluation. Define test cases in YAML, run them against any Groq-hosted model, and get scored, tracked, auditable results — every time.

```
YAML test suite → Runner → Heuristic assertions → Multi-judge scoring
                                                 → Hallucination detection
                                                 → Relevance scoring
                                                 → Verdict + regression check
                                                 → Streamlit dashboard
```

---

## Architecture

```
verdictai/
├── tests/suites/          # YAML test suites (hallucination, safety, format, RAG)
├── runner/                # Suite loader, assertion engine, Groq model client
├── judge/                 # LLM-as-judge, multi-judge consensus, relevance, hallucination
├── memory/                # SQLite store, regression tracker, self-heal trigger
├── database/              # DatabaseManager — test_runs, test_results, llm_calls, scores
├── reports/               # Allure writer, CSV exporter
└── dashboard/             # Streamlit dashboard (app.py)
    └── app.py             # Overview, Suite Breakdown, Trends, Detail, Run Suite tab
```

### Six evaluation layers

| Layer | What it does |
|---|---|
| **1. YAML loader** | Parses test suites — input, expected_behavior, assertions, scoring_mode |
| **2. Heuristic assertions** | Fast rule checks (contains, max_length, regex) before LLM calls |
| **3. Multi-judge scoring** | Groq + Cerebras in parallel; consensus score, disagreement detection |
| **4. Relevance scoring** | sentence-transformers cosine similarity vs expected behavior |
| **5. Hallucination detection** | Claim extraction + batched NLI verification (~60% fewer tokens) |
| **6. Regression tracker** | Score drop detection across runs; Jira ticket on consecutive failures |

---

## Live Demo

The dashboard is deployed on Streamlit Community Cloud.

**[→ Open VerdictAI Dashboard](https://YOUR_APP.streamlit.app)**

The **▶️ Run Suite** tab lets you run a live evaluation:
1. Paste your [Groq API key](https://console.groq.com) (free)
2. Pick a test suite (format, hallucination, RAG, safety)
3. Click **Run Suite** — watch each test case scored in real time

> Demo uses `llama-3.1-8b-instant` via Groq free tier. Keys are not stored.

---

## Run locally

```bash
git clone https://github.com/anant-pw/verdictai
cd verdictai
pip install -r requirements.txt

# Add your keys
cp config.env.example config.env
# edit config.env → GROQ_API_KEY=gsk_...

# Run a suite
python -m runner.main --suite tests/suites/hallucination.yaml

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Test suite format

```yaml
# tests/suites/my_suite.yaml
test_cases:
  - id: factual_001
    input: "What is the capital of France?"
    expected_behavior: "Paris is the capital of France"
    judge_threshold: 70
    scoring_mode: full        # full | judge_only
    assertions:
      - type: contains
        value: "Paris"
      - type: max_length
        value: 300
```

`scoring_mode: full` — runs judge + relevance + hallucination (factual suites)  
`scoring_mode: judge_only` — runs judge only (safety, format suites)

---

## Key design decisions

**Why multi-judge?** Single LLM judges are inconsistent. Running Groq + Cerebras in parallel and averaging scores reduces variance and flags genuine disagreements (gap > 20 points).

**Why heuristics first?** LLM calls cost tokens and time. Fast regex/contains checks catch obvious failures before invoking any judge.

**Why SQLite?** Zero-config, file-based, portable. Swappable for Postgres via `DatabaseManager` without touching eval logic.

**Why YAML test suites?** Non-engineers can write and review test cases. No Python required to extend coverage.

---

## Stack

| Component | Technology |
|---|---|
| LLM provider | Groq (llama-3.1-8b-instant, llama-3.3-70b) |
| Orchestration | LangChain LCEL |
| Relevance scoring | sentence-transformers |
| Dashboard | Streamlit + Plotly |
| Storage | SQLite via DatabaseManager |
| Reporting | Allure + CSV |
| CI | GitHub Actions |
| Hosting | Streamlit Community Cloud |

---

## Project status

Feature-complete for portfolio and interview demonstration purposes. Architectural extensions in scope: swap SQLite → Supabase for persistent multi-session storage; add OpenAI/Gemini judge providers; expose eval API via FastAPI.

---

## Author

**Anant Jain** — QA Engineer / AI Engineer  
[Portfolio](https://anant-pw.github.io) · [LinkedIn](https://www.linkedin.com/in/anant-jain-40760719/) · [GitHub](https://github.com/anant-pw)
