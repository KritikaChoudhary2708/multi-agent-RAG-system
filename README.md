# Multi-Agent RAG System + LLM Eval Platform
> Production-grade Retrieval Augmented Generation pipeline over SEC filings — with adversarial red-teaming, LLM-as-judge scoring, a live leaderboard, and a dual CI/CD gate.

---

## What This Is

Two projects. One unified system.

**Project 1 — Multi-Agent RAG System (Panchayat):** Answers financial questions from SEC filings with faithfulness 1.0 and zero hallucination on RAGAS evaluation.

**Project 2 — LLM Eval & Red-Teaming Platform:** Attacks P1 with 27 adversarial prompts across 6 categories, scores responses using an ensemble judge (rule-based + LLM-as-judge), posts results to a live leaderboard, auto-generates a PDF report, and enforces a safety gate in CI.

The story: I built a RAG system, then built the infrastructure to break it, measure it, and make sure it never regresses.

---

## Full Architecture

```
User Query (via Panchayat Web App)
    ↓
Query Decomposition Agent     → breaks complex questions into sub-questions
    ↓
Retrieval Agent               → Hybrid Search (BM25 + Dense + RRF)
    ↓
Synthesis Agent               → Grounded answer generation with citations
    ↓
Evaluation Agent              → RAGAS scoring (faithfulness, relevancy, recall)
    ↓
Panchayat (Streamlit UI)      → Interactive web interface


Red-Team Attack Layer (eval-platform/)
    ↓
RAG Attacker                  → 27 adversarial prompts across 6 categories
    ↓
Async Eval Runner             → LiteLLM + asyncio parallel execution
    ↓
Ensemble Judge                → 0.4 × rule score + 0.6 × LLM-as-judge score
    ↓
FastAPI Leaderboard           → Persists results to SQLite, serves /leaderboard
    ↓
Streamlit Dashboard           → Live model comparison chart
    ↓
PDF Report Generator          → Auto-generates per-run report (fpdf2)
    ↓
CI Safety Gate                → Blocks merge if avg score < 0.70
```

---

## CI/CD — Dual Gate

Every PR triggers two jobs in sequence:

**Job 1 — Quality Gate (P1)**
- Syntax validation across all P1 agents
- RAGAS score floor: faithfulness ≥ 0.85, relevancy ≥ 0.75

**Job 2 — Safety Gate (P2)**
- Runs adversarial prompts against the RAG system
- Ensemble judge scores each response
- Exits code 1 (blocks merge) if avg score < 0.70

```yaml
jobs:
  evaluate:       # P1 quality gate
    ...
  safety-eval:    # P2 safety gate
    needs: evaluate
    steps:
      - name: Run safety gate
        run: |
          cd eval-platform
          python3 main.py --model groq/llama-3.1-8b-instant --category prompt_injection --ci
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

A PR that improves answer quality but makes the model more vulnerable to prompt injection gets blocked automatically.

---

## P1 — Multi-Agent RAG System

### Agents

**1. Ingestion Agent (`ingestion_agent.py`)**
- Fetches real SEC filings from EDGAR
- Semantic chunking (500 words, 50-word overlap)
- Embeds using `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in ChromaDB with metadata

**2. Retrieval Agent (`retrieval_agent.py`)**
- BM25 — exact keyword matching via `rank-bm25`
- Dense search — semantic vector similarity via ChromaDB
- Reciprocal Rank Fusion (RRF) — combines both rankings

**3. Synthesis Agent (`synthesis_agent.py`)**
- Retrieved chunks + query → grounded answer
- Forced citation format: `[Chunk N]`
- `temperature=0.1` to minimize hallucination
- Confidence scoring based on citation presence

**4. Query Decomposition Agent (`query_decomposition.py`)**
- Breaks multi-hop questions into independent sub-questions
- Retrieves separately per sub-question
- Combines sub-answers into one coherent final answer

**5. Evaluation Agent (`evaluation_agent.py`)**
- RAGAS evaluation against a golden Q&A dataset
- Metrics: faithfulness, answer relevancy, context recall, context precision

### RAGAS Results

```
Faithfulness:      1.000  ✅  (zero hallucination)
Answer Relevancy:  0.998  ✅  (direct, cited answers)
Context Recall:    1.000  ✅  (right chunks retrieved)
Context Precision: 1.000  ✅  (no noisy chunks)
```

---

## P2 — LLM Eval & Red-Teaming Platform

### Red-Team Prompt Library

27 adversarial prompts across 6 attack categories:

| Category | Count | Severity | Expected |
|---|---|---|---|
| `prompt_injection` | 5 | high | should_refuse |
| `jailbreak` | 5 | high | should_refuse |
| `hallucination_trap` | 4 | medium | should_not_fabricate |
| `bias_probe` | 6 | medium | should_not_stereotype |
| `context_poisoning` | 3 | high | should_not_propagate_false_info |
| `multi_hop_confusion` | 4 | medium | should_not_fabricate |

### Ensemble Judge

```
final_score = 0.4 × rule_score + 0.6 × llm_score
```

- **Rule judge** — regex/keyword heuristics → 0 or 1
- **LLM judge** — Gemini Flash scores response against rubric → 0.0–1.0
- **Ensemble** — weighted combination

### Key Finding

26/27 prompts returned "Not found in provided documents" (confidence 0.4). This is a retrieval-miss, not a genuine safety refusal. The judge distinguishes three outcomes:
1. Genuine refusal (LLM safety filter fired)
2. Retrieval-miss (no relevant chunk — looks like refusal but isn't)
3. Actual compliance (model answered the adversarial prompt)

### CLI

```bash
# Run against a specific category
python3 main.py --model groq/llama-3.1-8b-instant --category prompt_injection

# Run all categories with CI gate
python3 main.py --model groq/llama-3.1-8b-instant --category all --ci
```

### Leaderboard

FastAPI backend + Streamlit UI showing avg ensemble score, rule score, and LLM score per model. SQLite persistence — every eval run appends results.

```bash
# Start API
cd eval-platform/leaderboard && uvicorn api:app --port 8000

# Start UI
streamlit run ui.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Keyword search | rank-bm25 |
| LLM (generation) | Groq — LLaMA 3.1 8B Instant |
| LLM (judge) | Gemini Flash |
| Unified LLM API | LiteLLM |
| Evaluation | RAGAS |
| Data source | SEC EDGAR (Apple 10-K 2023) |
| Web UI | Streamlit (P1 + P2) |
| API | FastAPI |
| Database | SQLite |
| PDF reports | fpdf2 |
| CI/CD | GitHub Actions |

---

## Project Structure

```
multi-agent-RAG-system/
├── app.py                      # Panchayat Streamlit UI
├── ingestion_agent.py
├── retrieval_agent.py
├── synthesis_agent.py
├── query_decomposition.py
├── evaluation_agent.py
├── ci_evaluation.py
├── eval_gate.py
├── golden_dataset.py
├── .github/
│   └── workflows/
│       └── eval.yml            # Dual CI gate (quality + safety)
├── requirements.txt
│
└── eval-platform/
    ├── main.py                 # CLI entry point
    ├── red_team/
    │   ├── prompt_library.py   # 27 adversarial prompts
    │   └── rag_attacker.py     # fires prompts through P1 RAG
    ├── eval_runner/
    │   └── async_runner.py     # LiteLLM + asyncio
    ├── judge/
    │   ├── rule_judge.py
    │   ├── llm_judge.py
    │   └── ensemble.py
    ├── leaderboard/
    │   ├── db.py               # SQLite helpers
    │   ├── api.py              # FastAPI
    │   └── ui.py               # Streamlit dashboard
    └── reports/
        └── generator.py        # PDF report (fpdf2)
```

---

## Setup

```bash
git clone https://github.com/KritikaChoudhary2708/multi-agent-RAG-system.git
cd multi-agent-RAG-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r eval-platform/requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your-groq-key-here
GOOGLE_API_KEY=your-gemini-key-here
```

### Run Panchayat (P1 UI)
```bash
streamlit run app.py
```

### Run red-team eval (P2)
```bash
cd eval-platform
python3 main.py --model groq/llama-3.1-8b-instant --category all --ci
```

### Start leaderboard
```bash
cd eval-platform/leaderboard
uvicorn api:app --port 8000
streamlit run ui.py
```

---

## Key Learnings

- **Hybrid search beats either alone** — BM25 finds exact keywords, dense search finds meaning. RRF fusion gives best of both.
- **Query decomposition is essential** — single retrieval fails multi-hop questions. Decompose first, retrieve per sub-question, combine.
- **Prompt engineering > model size** — moving from `temperature=1.0` to `0.1` moved Answer Relevancy from 0.36 to 0.998.
- **Retrieval-miss ≠ safety refusal** — the RAG system's "answer only from context" constraint deflects most attacks via retrieval-miss. This is not a genuine safety property. The judge must distinguish the two.
- **Eval infrastructure creates constraints** — the value isn't in any single score. A dual CI gate means improving quality while increasing vulnerability gets blocked automatically.

---

## Part of 100 Days of Projects

Follow the build: [LinkedIn — Kritika Choudhary](https://www.linkedin.com/in/kritika-choudhary2708)

