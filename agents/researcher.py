"""
Research Agent — Searches the web and uploaded papers (via RAG) to collect
structured facts with source attribution for autonomous driving topics.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchResults


class ResearchAgent:
    """
    The Research Agent is the second stage of the pipeline.
    For each sub-query from the Triage Agent, it:
    1. Searches the web using DuckDuckGo
    2. Queries uploaded papers via RAG (if available)
    3. Extracts structured facts with source attribution
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", rag_engine=None):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.2
        )
        self.rag_engine = rag_engine
        self.search_tool = DuckDuckGoSearchResults(max_results=5)
        self.parser = JsonOutputParser()

        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an autonomous driving research analyst. 
Given search results and/or paper excerpts about a specific query, extract 
the most important facts as structured data.

FOCUS ON:
- Quantitative results (mAP, IoU, FPS, latency, accuracy)
- Architecture details and innovations
- Dataset and benchmark comparisons
- Deployment constraints (compute, real-time requirements)
- Safety-critical findings
- Limitations and failure cases

Return a JSON array of fact objects. Each fact must have:
- "title": Short descriptive title (max 10 words)
- "content": The key finding or information (2-3 sentences max)
- "source": Where this information came from
- "relevance": "high", "medium", or "low"

Return 3-5 facts per query. ONLY high and medium relevance.
Return ONLY the JSON array, no explanation."""),
            ("human", """Research query: {query}
Overall topic: {topic}

Web search results:
{search_results}

{rag_context}

Extract the key facts:""")
        ])

        self.chain = self.extraction_prompt | self.llm | self.parser

    def _web_search(self, query: str) -> str:
        """Execute web search and return results as text."""
        try:
            results = self.search_tool.invoke(query)
            return str(results)
        except Exception as e:
            print(f"[ResearchAgent] Web search error: {e}")
            return "No web results available."

    def _rag_search(self, query: str) -> str:
        """Query uploaded papers via RAG engine."""
        if not self.rag_engine:
            return ""

        try:
            results = self.rag_engine.query(query)
            if results:
                return f"Paper excerpts (from uploaded PDFs):\n{results}"
            return ""
        except Exception as e:
            print(f"[ResearchAgent] RAG search error: {e}")
            return ""

    def research(self, query: str, topic: str) -> list[dict]:
        """
        Research a specific sub-query using web search and RAG.

        Args:
            query: The focused sub-query to research
            topic: The overall research topic for context

        Returns:
            List of structured fact dictionaries
        """
        # Gather information from both sources
        search_results = self._web_search(query)
        rag_context = self._rag_search(query)

        try:
            facts = self.chain.invoke({
                "query": query,
                "topic": topic,
                "search_results": search_results,
                "rag_context": rag_context if rag_context else "No uploaded papers available."
            })

            if isinstance(facts, list):
                # Filter to high and medium relevance only
                filtered = [
                    f for f in facts
                    if isinstance(f, dict) and f.get("relevance") in ("high", "medium")
                ]
                return filtered if filtered else facts[:3]

            return []

        except Exception as e:
            print(f"[ResearchAgent] Extraction error: {e}")
            return [{
                "title": "Search completed",
                "content": f"Found results for: {query}. Manual review recommended.",
                "source": "Web search",
                "relevance": "medium"
            }]
