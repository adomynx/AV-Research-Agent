"""
FastAPI backend — exposes the agentic research pipeline as a REST API.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health    — liveness + which provider/model is configured
    POST /research  — run the full agentic pipeline for a topic

This is the "production shape" layer: the same graph that powers the Streamlit
UI is now callable by any client (or another service), containerized via the
Dockerfile, and deployable behind a URL.
"""

from __future__ import annotations

from dotenv import load_dotenv

# Load .env before importing modules that read provider config at import time.
load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents.graph import run_research  # noqa: E402
from utils.llm import has_llm_credentials, provider_label  # noqa: E402
from utils.observability import track_run  # noqa: E402

app = FastAPI(
    title="AV Research Agent API",
    version="1.0.0",
    description="Multi-agent research pipeline for autonomous driving & perception.",
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Research topic to investigate.")
    max_iterations: int = Field(
        2, ge=1, le=4, description="Max research/reflection rounds (bounds the loop)."
    )


class ResearchResponse(BaseModel):
    topic: str
    provider: str
    research_rounds: int
    queries: list[str]
    facts: list[dict]
    report: str
    decision_log: list[str]
    tokens: int


@app.get("/health")
def health() -> dict:
    """Liveness probe + current provider configuration."""
    return {
        "status": "ok",
        "provider": provider_label(),
        "credentials_configured": has_llm_credentials(),
    }


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    """Run the full Plan → Research → Reflect ↺ → Compile pipeline."""
    if not has_llm_credentials():
        raise HTTPException(
            status_code=400,
            detail="No API key configured for the selected LLM provider.",
        )

    with track_run("api_research") as stats:
        state = run_research(req.topic, max_iterations=req.max_iterations)

    return ResearchResponse(
        topic=req.topic,
        provider=provider_label(),
        research_rounds=state.get("iteration", 0),
        queries=state.get("all_queries", []),
        facts=[f.model_dump(mode="json") for f in state.get("facts", [])],
        report=state.get("report", ""),
        decision_log=state.get("log", []),
        tokens=stats.total_tokens,
    )
