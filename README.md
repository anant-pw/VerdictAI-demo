# VerdictAI — LLM Evaluation Framework

> **Your LLM passed. But did it really?**

VerdictAI is an open-source framework for systematically evaluating LLM outputs — built for QA engineers and AI teams who need structured, repeatable proof that their models are behaving correctly.

It runs YAML-defined test suites against any LLM, scores responses using multiple judges in parallel, detects hallucinations, tracks regressions across runs, and visualises everything in a Streamlit dashboard.

---

## Real Test Run Results

| Suite | Cases | Pass Rate | Scoring Mode |
|---|---|---|---|
| Hallucination | 15 | 80% | full (judge + relevance + hallucination) |
| Safety | 10 | 80% | judge_only |
| Format | 10 | 40% | judge_only |
| RAG | 4 | 75% | full |
| **Total** | **39** | **65.4%** | |

- **Avg LLM latency:** 381ms per call
- **Multi-judge agreement rate:** 95% (Groq + Cerebras agreed on 57/60 cases)
- **Avg judge score:** 96.1 / 100
- The 35% failures are not bugs — they are the framework correctly catching real LLM weaknesses in format compliance and context boundaries.

---

## Architecture

```
verdictai/
├── runner/
│   ├── main.py              # CLI entry point
│   ├── runner.py            # Core eval loop — orchestrates all scorers
│   ├── loader.py            # YAML suite loader
│   ├── assertions.py        # Heuristic checks (contains, not_contains, max_length)
│   ├── groq_model.py        # Model-under-test caller
│   ├── retry_utils.py       # Exponential backoff + inter-case sleep
│   └── logger.py            # JSONL structured logging
├── judge/
│   ├── multi_judge.py       # Parallel multi-judge scoring (Groq + Cerebras/SambaNova)
│   ├── llm_judge.py         # Single judge scorer + response parser
│   ├── groq_client.py       # Groq API client with retry decorator
│   ├── hallucination_detector.py  # Claim extraction + batched verification
│   ├── relevance_scorer.py  # Cosine similarity via sentence-transformers
│   └── judge_cache.py       # SQLite-backed result cache
├── memory/
│   ├── store.py             # Run history persistence
│   ├── regression.py        # Score drop detection across runs
│   └── jira_client.py       # Auto-create Jira tickets on FAIL
├── reports/
│   ├── allure_writer.py     # Allure report integration
│   └── cli_reporter.py      # Terminal output formatter
├── database/
│   └── models.py            # SQLAlchemy models (eval_runs, test_results, scores, llm_calls)
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── tests/suites/
│   ├── hallucination.yaml   # 15 factual accuracy test cases
│   ├── safety.yaml          # 10 safety / jailbreak test cases
│   ├── format.yaml          # 10 format compliance test cases
│   └── rag.yaml             # 4 RAG context fidelity test cases
└── config.env.template      # API keys template
```

---

## How It Works

Each test case flows through a 7-stage pipeline:

```
YAML Input
    │
    ▼
1. Heuristic Assertions    contains / not_contains / max_length
    │ FAIL → immediate FAIL verdict
    ▼
2. LLM Response            Groq (llama-3.1-8b-instant)
    │
    ▼
3. Multi-Judge Scoring     Groq + Cerebras in parallel → consensus score
    │
    ▼
4. Relevance Scoring       Cosine similarity via sentence-transformers
    │ (skipped for judge_only suites)
    ▼
5. Hallucination Detection Extract claims → batch-verify against source → support rate
    │ (skipped for judge_only suites)
    ▼
6. Verdict Engine          Aggregate all signals → PASS / FAIL + reason
    │
    ▼
7. Storage + Reporting     SQLite → Streamlit dashboard + Allure report + JSONL log
```

### Scoring Modes

| Mode | When to use | Scorers active |
|---|---|---|
| `full` | Factual suites (hallucination, RAG) | Judge + Relevance + Hallucination |
| `judge_only` | Behaviour suites (safety, format) | Judge only — cosine similarity is meaningless for refusals/format checks |

Set per test case in YAML:
```yaml
- id: my_test
  scoring_mode: judge_only   # or: full
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/anant-pw/verdictai.git
cd verdictai
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp config.env.template config.env
```

Open `config.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_key_here          # Required — model under test + judge
CEREBRAS_API_KEY=your_cerebras_key_here  # Recommended — second judge (generous free tier)
SAMBANOVA_API_KEY=your_sambanova_key     # Optional — fallback second judge
```

Get free keys:
- Groq → https://console.groq.com
- Cerebras → https://cloud.cerebras.ai
- SambaNova → https://cloud.sambanova.ai

### 3. Run a suite

```bash
python -m runner.main --suite tests/suites/hallucination.yaml
```

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## Writing Test Cases

```yaml
suite: my_suite
description: What this suite tests.

test_cases:

  - id: tc_001
    scoring_mode: full            # full | judge_only
    input: "Who invented Python?"
    expected_behavior: "Answer must mention Guido van Rossum as the creator of Python."
    judge_threshold: 80           # minimum judge score to PASS (0-100)
    assertions:
      - type: contains
        value: "Guido"
      - type: not_contains
        value: "Elon"
      - type: max_length
        value: 500
```

### Assertion types

| Type | Description | Value |
|---|---|---|
| `contains` | Response must include this string (case-insensitive) | string |
| `not_contains` | Response must NOT include this string | string |
| `max_length` | Response character count must be ≤ this | integer |
| `min_length` | Response character count must be ≥ this | integer |

---

## Running Suites for Regression Data

Run each suite **twice** to generate meaningful regression tracking:

```bash
python reset_db.py

# Hallucination — run twice
python -m runner.main --suite tests/suites/hallucination.yaml
python -m runner.main --suite tests/suites/hallucination.yaml

# Safety
python -m runner.main --suite tests/suites/safety.yaml
python -m runner.main --suite tests/suites/safety.yaml

# Format
python -m runner.main --suite tests/suites/format.yaml
python -m runner.main --suite tests/suites/format.yaml

# RAG
python -m runner.main --suite tests/suites/rag.yaml
python -m runner.main --suite tests/suites/rag.yaml
```

The second run of each suite compares scores against the first and flags any drops > 10 points as regressions.

---

## Multi-Judge System

VerdictAI runs two LLM judges in parallel and averages their scores:

```
Input + Response + Expected Behavior
            │
    ┌───────┴────────┐
    ▼                ▼
  Groq            Cerebras        ← parallel, async
  judge           judge
    │                │
    └───────┬────────┘
            ▼
     Consensus Score (avg)
     Disagreement flag if gap > threshold
```

- If both judges agree → consensus score used
- If judges disagree by > 20 points → disagreement flagged in results
- If one judge errors (429 / timeout) → surviving judge's score used alone
- If both judges error → heuristic result used, judge score = N/A

---

## Hallucination Detection

Uses a two-step batched approach to minimise API calls:

**Step 1 — Claim extraction** (1 API call)
```
"Who invented Python?" → "The Python programming language was created by Guido van Rossum in 1991..."
                       → ["Guido van Rossum created Python", "Python was created in 1991"]
```

**Step 2 — Batched verification** (1 API call for all claims, not N)
```
Claims verified against source_context:
  ✅ SUPPORTED    — claim is stated or implied in source
  ❌ CONTRADICTED — source disagrees with claim
  ⚠️  UNSUPPORTED — claim not mentioned in source

Hallucination Score = (supported / total) × 100
```

> Previous approach used N separate API calls per claim. Batching reduced token usage by ~60%.

---

## Rate Limit Handling

VerdictAI includes built-in protection against free tier API rate limits:

- **2 second sleep** between test cases (configurable in `retry_utils.py`)
- **Exponential backoff** with up to 4 retry attempts on 429 errors
- **Provider fallback** — Cerebras → SambaNova → single-judge if second judge unavailable

Free tier limits for reference:

| Provider | RPM | Daily |
|---|---|---|
| Groq (llama-3.1-8b) | 30 | 14,400 |
| Cerebras (llama3.1-8b) | generous | generous |
| SambaNova | ~20 | ~600 |

A full 39-case run across all suites completes in ~14 minutes within free tier limits.

---

## Jira Integration

When a test FAILs, VerdictAI can automatically create a Jira ticket:

```env
JIRA_URL=https://yourproject.atlassian.net
JIRA_EMAIL=your@email.com
JIRA_API_TOKEN=your_token
JIRA_PROJECT_KEY=VQA
```

Ticket includes: test ID, input, response, judge score, failure reason.

---

## CI/CD

GitHub Actions workflow runs all suites on every push to `main`:

```yaml
# .github/workflows/eval.yml
- name: Run VerdictAI
  run: python -m runner.main --suite tests/suites/hallucination.yaml
```

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Features:
- Suite-level pass/fail metrics
- Per-test case drill-down (judge score, relevance, hallucination, reasoning)
- Judge score distribution charts
- Regression history across runs
- Full CSV export

> **Note:** `relevance` and `hallucination` columns show `N/A` for `judge_only` suites — this is expected behaviour, not missing data.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM providers | Groq, Cerebras, SambaNova, Gemini |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Orchestration | LangChain |
| Tracing | LangSmith |
| Storage | SQLite (via SQLAlchemy) |
| Dashboard | Streamlit + Plotly |
| Reports | Allure |
| Retry | tenacity |
| CI/CD | GitHub Actions |

---

## Limitations

- Judge scoring is probabilistic — not a deterministic guarantee
- Relevance scoring (cosine similarity) only applies to factual suites — meaningless for behaviour suites
- Hallucination detection accuracy depends on quality of `source_context` / `expected_behavior` in YAML
- SQLite is sufficient for development and portfolio use — would need migration for production scale
- Free API tiers have daily limits — not suitable for large-scale continuous eval without paid keys

---

## Author

**Anant Jain** — Principal QA Automation Engineer upskilling into AI-augmented testing  
GitHub: [anant-pw](https://github.com/anant-pw)  
Portfolio: [anant-pw.github.io](https://anant-pw.github.io)  
LinkedIn: [linkedin.com/in/anant-jain-40760719](https://linkedin.com/in/anant-jain-40760719)

---

## License

MIT — free to use, fork, and build on.
