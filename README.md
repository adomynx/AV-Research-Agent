# 🚗 AV Research Agent

A multi-agent research pipeline built with **LangChain** and **Streamlit** for automated literature review and technical analysis in autonomous driving and perception systems.

## Overview

AV Research Agent automates the process of researching autonomous driving topics by coordinating three specialized AI agents that plan, gather, and compile information into structured technical reports.

Upload your own research papers (PDFs) for **RAG-powered analysis** — the agents will cross-reference web findings with your uploaded documents for deeper, citation-backed insights.

## Architecture

```
┌──────────────────────────────────────────────┐
│                Research Topic                 │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│            🎯 Triage Agent                    │
│  Decomposes topic into 4-6 focused queries   │
│  covering perception, safety, benchmarks     │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│           🔍 Research Agent                   │
│  Web Search (DuckDuckGo) + RAG (FAISS)       │
│  Extracts structured facts with sources      │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│            📝 Editor Agent                    │
│  Compiles facts into structured report       │
│  with comparison tables and citations        │
└──────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Pipeline**: Triage → Research → Editor workflow with real-time status tracking
- **RAG Integration**: Upload research papers (PDF) for semantic search during analysis
- **Web Search**: Automated web research using DuckDuckGo for current information
- **AV Domain Focus**: Pre-tuned for perception, sensor fusion, ODD, V&V, and safety-critical systems
- **Structured Reports**: Generates markdown reports with executive summaries, comparison tables, and citations
- **Interactive UI**: Built with Streamlit — monitor each agent's progress in real-time
- **Downloadable Output**: Export reports as `.md` files

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/adomynx/AV-Research-Agent.git
cd AV-Research-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or enter it directly in the app sidebar.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

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
| LLM               | OpenAI GPT-4o-mini                  |
| Embeddings        | OpenAI text-embedding-3-small       |
| Vector Store      | FAISS                               |
| Web Search        | DuckDuckGo                          |
| PDF Parsing       | PyPDF                               |
| Frontend          | Streamlit                           |
| Language          | Python 3.10+                        |

## Project Structure

```
av-research-agent/
├── app.py                   # Main Streamlit application
├── agents/
│   ├── __init__.py
│   ├── triage.py            # Topic decomposition agent
│   ├── researcher.py        # Web search + RAG research agent
│   └── editor.py            # Report compilation agent
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py        # PDF ingestion and chunking
│   └── rag_engine.py        # FAISS-based RAG engine
├── requirements.txt
└── README.md
```

