"""
Research Agent — Searches the web and uploaded papers (via RAG) to collect
structured facts with source attribution for autonomous driving topics.

Fact extraction is bound to the ``FactSet`` schema via structured output, so
every fact is a validated ``Fact`` (typed relevance, non-empty fields) rather
than an untyped dict.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults

from schemas.models import Fact, FactSet, Relevance
from utils.llm import get_chat_model

SYSTEM_PROMPT = """You are an autonomous driving research analyst.
Given search results and/or paper excerpts about a specific query, extract
the most important facts as structured data.

FOCUS ON:
- Quantitative results (mAP, IoU, FPS, latency, accuracy)
- Architecture details and innovations
- Dataset and benchmark comparisons
- Deployment constraints (compute, real-time requirements)
- Safety-critical findings
- Limitations and failure cases

Each fact must have:
- title: Short descriptive title (max 10 words)
- content: The key finding or information (2-3 sentences max)
- source: Where this information came from (a URL, paper name, or site)
- relevance: "high", "medium", or "low"

Return 3-5 facts per query. Prefer high and medium relevance."""


class ResearchAgent:
    """
    The Research Agent is the second stage of the pipeline.
    For each sub-query from the Triage Agent, it:
    1. Searches the web using DuckDuckGo
    2. Queries uploaded papers via RAG (if available)
    3. Extracts structured facts with source attribution
    """

    def __init__(self, rag_engine=None):
        llm = get_chat_model(temperature=0.2)
        self.rag_engine = rag_engine
        self.search_tool = DuckDuckGoSearchResults(max_results=5)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Research query: {query}
Overall topic: {topic}

Web search results:
{search_results}

{rag_context}

Extract the key facts:"""),
        ])
        self.chain = self.prompt | llm.with_structured_output(FactSet)

    def _web_search(self, query: str) -> str:
        """Execute web search and return results as text."""
        try:
            return str(self.search_tool.invoke(query))
        except Exception as e:  # noqa: BLE001
            print(f"[ResearchAgent] Web search error: {e}")
            return "No web results available."

    def _rag_search(self, query: str) -> str:
        """Query uploaded papers via RAG engine."""
        if not self.rag_engine:
            return ""
        try:
            results = self.rag_engine.query(query)
            return f"Paper excerpts (from uploaded PDFs):\n{results}" if results else ""
        except Exception as e:  # noqa: BLE001
            print(f"[ResearchAgent] RAG search error: {e}")
            return ""

    def research(self, query: str, topic: str) -> list[Fact]:
        """
        Research a specific sub-query using web search and RAG.

        Args:
            query: The focused sub-query to research
            topic: The overall research topic for context

        Returns:
            List of validated ``Fact`` objects (high/medium relevance preferred).
        """
        search_results = self._web_search(query)
        rag_context = self._rag_search(query)

        try:
            fact_set: FactSet = self.chain.invoke({
                "query": query,
                "topic": topic,
                "search_results": search_results,
                "rag_context": rag_context or "No uploaded papers available.",
            })
            keep = fact_set.high_and_medium()
            return keep if keep else fact_set.facts[:3]
        except Exception as e:  # noqa: BLE001
            print(f"[ResearchAgent] Extraction error: {e}")
            return [Fact(
                title="Search completed",
                content=f"Found results for: {query}. Manual review recommended.",
                source="Web search",
                relevance=Relevance.MEDIUM,
            )]
