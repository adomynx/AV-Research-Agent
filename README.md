# 🚗 AV Research Agent

A multi-agent research pipeline built with **LangChain** and **Streamlit** for automated literature review and technical analysis in autonomous driving and perception systems.

## Overview

AV Research Agent automates the process of researching autonomous driving topics by coordinating three specialized AI agents that plan, gather, and compile information into structured technical reports.

Upload your own research papers (PDFs) for **RAG-powered analysis** — the agents will cross-reference web findings with your uploaded documents for deeper, citation-backed insights.

## Architecture

The pipeline is a **stateful LangGraph agent**, not a fixed chain. After each
research round a Reflection step decides whether the evidence is good enough or
whether to loop back and dig deeper — so the path depends on the topic.

```
        ┌──────┐     ┌──────────┐     ┌─────────┐
  START │ plan │ ──▶ │ research │ ──▶ │ reflect │
        └──────┘     └──────────┘     └────┬────┘
          🎯            🔍 web+RAG      🤔  │
                          ▲                 │
                          │  gaps found     │ sufficient
                          └─────────────────┤
                                            ▼
                                       ┌─────────┐
                                       │ compile │ ──▶ END
                                       └─────────┘
                                          📝
```

| Node | Agent | Role |
|------|-------|------|
| `plan` | 🎯 Triage | Decomposes the topic into 4-6 focused sub-queries |
| `research` | 🔍 Research | Web search (DuckDuckGo) + RAG (FAISS); extracts typed facts with sources |
| `reflect` | 🤔 Reflection | Judges evidence coverage; emits targeted follow-up queries on gaps (bounded loop) |
| `compile` | 📝 Editor | Compiles facts into a cited, structured report |

The `reflect → research` edge is conditional: the graph loops only while gaps
remain and `iteration < max_iterations`, guaranteeing termination.

## Features

- **Agentic Graph (LangGraph)**: Plan → Research → **Reflect ↺** → Compile, with a bounded reflection loop that adaptively digs deeper when evidence is thin — real agency, not a fixed chain
- **Provider-Agnostic**: Swap OpenAI ↔ Anthropic (Claude) with one env var — benchmark GPT vs Claude on the same eval harness
- **Free Local Embeddings**: RAG uses a local FastEmbed (ONNX) model by default — no embeddings API key, no per-vector cost
- **Typed, Validated I/O**: Agents emit Pydantic-validated structured output (`ResearchPlan`, `FactSet`) — no fragile JSON parsing
- **Evaluation Harness**: Scores every report for citation grounding, structure, source grounding, faithfulness, and topic relevance — with a reproducible before/after scorecard
- **Observability**: Per-run token / cost / latency tracking, plus optional LangSmith tracing
- **RAG Integration**: Upload research papers (PDF) for semantic search during analysis
- **Web Search**: Automated web research using DuckDuckGo for current information
- **AV Domain Focus**: Pre-tuned for perception, sensor fusion, ODD, V&V, and safety-critical systems
- **Structured Reports**: Generates markdown reports with executive summaries, comparison tables, and citations
- **Interactive UI**: Built with Streamlit — monitor each agent's progress, token cost, and RAG/trace status in real-time
- **Downloadable Output**: Export reports as `.md` files

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/adomynx/AV-Research-Agent.git
cd AV-Research-Agent
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your environment

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

You can also enter the key directly in the app sidebar. LangSmith tracing is
optional — set the `LANGCHAIN_*` vars in `.env` to enable it.

**Switching LLM provider:** set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic`
in `.env` (and the matching `*_API_KEY`), or pick it in the app sidebar. Claude
keys come from https://console.anthropic.com — note that's the **API**, separate
from a Claude Pro subscription. Embeddings run locally for free regardless of
provider, so RAG never needs an OpenAI key.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 5. Run the evaluation harness

```bash
# Offline: deterministic metrics only (no API key, no network, free)
python -m evals.run_eval

# Live: regenerate reports with the real agents + LLM-as-judge metrics
python -m evals.run_eval --live
```

Scorecards (JSON + Markdown) are written to `evals/results/`. Use them to
measure a real before/after when you change prompts, models, or retrieval.

## Usage

1. Enter a research topic (or pick from the examples in the sidebar)
2. Optionally upload research papers (PDFs) for RAG-powered analysis
3. Click **Start Research** and watch the agents work
4. View the compiled report in the **Report** tab
5. Download the report as markdown

### Example Topics

- BEVFusion vs PointPillars for 3D object detection
- ODD exit detection strategies in autonomous vehicles
- LiDAR-camera fusion architectures for perception
- Safety validation methods for AV perception systems
- Transformer architectures in autonomous driving

## Tech Stack

| Component         | Technology                          |
|-------------------|-------------------------------------|
| LLM Framework     | LangChain                           |
| Agent Orchestration | LangGraph (stateful graph + reflection loop) |
| LLM               | OpenAI (GPT-4o-mini) or Anthropic (Claude) — configurable |
| Embeddings        | Local FastEmbed (BAAI/bge-small-en-v1.5), free; OpenAI optional |
| Vector Store      | FAISS                               |
| Web Search        | DuckDuckGo                          |
| PDF Parsing       | PyPDF                               |
| Validation        | Pydantic v2 (structured output)     |
| Evaluation        | Custom harness (deterministic + LLM-judge) |
| Observability     | Token/cost tracking + LangSmith     |
| Frontend          | Streamlit                           |
| Language          | Python 3.10+                        |

## Project Structure

```
av-research-agent/
├── app.py                   # Main Streamlit application
├── agents/
│   ├── graph.py             # LangGraph agentic pipeline with reflection loop
│   ├── triage.py            # Topic decomposition agent (structured output)
│   ├── researcher.py        # Web search + RAG research agent (structured output)
│   ├── reflection.py        # Evidence-gap reflection agent (drives the loop)
│   └── editor.py            # Report compilation agent
├── schemas/
│   └── models.py            # Pydantic data contracts (ResearchPlan, Fact, FactSet)
├── utils/
│   ├── llm.py               # Provider-agnostic LLM + embedding factories
│   ├── pdf_loader.py        # PDF ingestion and chunking
│   ├── rag_engine.py        # FAISS-based RAG engine (local embeddings)
│   └── observability.py     # Token / cost / latency tracking + LangSmith
├── evals/
│   ├── dataset.py           # Golden evaluation cases + report fixtures
│   ├── metrics.py           # Deterministic + LLM-as-judge metrics
│   ├── run_eval.py          # Eval runner → scorecard
│   └── results/             # Generated scorecards (JSON + Markdown)
├── .env.example
├── requirements.txt
└── README.md
```

## Evaluation

The project treats quality as something you **measure**, not vibe-check.
`evals/run_eval.py` scores generated reports against a golden dataset:

| Metric | Type | What it checks |
|--------|------|----------------|
| `citation_coverage` | deterministic | Every inline `[citation]` maps to a collected source (catches hallucinated cites) |
| `structure_coverage` | deterministic | Required report sections are present |
| `source_grounding` | deterministic | Every entry in the Sources section maps to a real fact |
| `faithfulness` | LLM-as-judge | Report claims are supported by the collected facts |
| `topic_relevance` | LLM-as-judge | Report actually answers the requested topic |

Deterministic metrics run offline with no API key; LLM-as-judge metrics run
when `OPENAI_API_KEY` is set. Each run writes a timestamped scorecard so you
can compare before/after any change.

