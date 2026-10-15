"""
Agentic research graph (LangGraph).

This replaces the old fixed Triage → Research → Editor chain with a stateful
graph that can *loop*:

        ┌──────┐     ┌──────────┐     ┌─────────┐
  START │ plan │ ──▶ │ research │ ──▶ │ reflect │
        └──────┘     └──────────┘     └────┬────┘
                          ▲                 │
                          │  gaps found     │ sufficient
                          └─────────────────┤
                                            ▼
                                       ┌─────────┐
                                       │ compile │ ──▶ END
                                       └─────────┘

The Reflection node inspects the evidence after each research round and, if it
finds gaps, emits targeted follow-up queries that feed back into Research. The
loop is bounded by ``max_iterations`` so it always terminates.

This is genuine agency: the path through the graph depends on the quality of
the evidence, not a hardcoded sequence.
"""

from __future__ import annotations

from typing import Callable, Optional, TypedDict

from langgraph.graph import START, END, StateGraph

from agents.triage import TriageAgent
from agents.researcher import ResearchAgent
from agents.reflection import ReflectionAgent
from agents.editor import EditorAgent
from schemas.models import Fact


class ResearchState(TypedDict, total=False):
    topic: str
    pending_queries: list[str]   # queries the next research round will run
    all_queries: list[str]       # every query ever run (for metrics/trace)
    facts: list[Fact]            # accumulated across rounds
    iteration: int               # completed research rounds
    max_iterations: int
    sufficient: bool
    missing_aspects: list[str]
    report: str
    log: list[str]               # human-readable decision trace


def build_graph(
    rag_engine=None,
    max_iterations: int = 2,
    on_event: Optional[Callable[[str], None]] = None,
):
    """Compile the agentic research graph with its agents bound in."""
    triage = TriageAgent()
    researcher = ResearchAgent(rag_engine=rag_engine)
    reflection = ReflectionAgent()
    editor = EditorAgent()

    def _log(state: ResearchState, msg: str) -> list[str]:
        if on_event:
            on_event(msg)
        return state.get("log", []) + [msg]

    def plan_node(state: ResearchState) -> dict:
        queries = triage.decompose(state["topic"])
        return {
            "pending_queries": queries,
            "all_queries": queries,
            "log": _log(state, f"🎯 Triage planned {len(queries)} sub-queries"),
        }

    def research_node(state: ResearchState) -> dict:
        new_facts: list[Fact] = []
        for query in state.get("pending_queries", []):
            new_facts.extend(researcher.research(query, state["topic"]))
        facts = state.get("facts", []) + new_facts
        iteration = state.get("iteration", 0) + 1
        return {
            "facts": facts,
            "iteration": iteration,
            "pending_queries": [],
            "log": _log(
                state,
                f"🔍 Research round {iteration}: +{len(new_facts)} facts "
                f"(total {len(facts)})",
            ),
        }

    def reflect_node(state: ResearchState) -> dict:
        verdict = reflection.assess(state["topic"], state.get("facts", []))
        if verdict.sufficient:
            return {
                "sufficient": True,
                "missing_aspects": [],
                "log": _log(state, "✅ Reflection: evidence sufficient — compiling"),
            }
        follow_up = verdict.follow_up_queries[:3]
        return {
            "sufficient": False,
            "missing_aspects": verdict.missing_aspects,
            "pending_queries": follow_up,
            "all_queries": state.get("all_queries", []) + follow_up,
            "log": _log(
                state,
                f"🤔 Reflection: gaps {verdict.missing_aspects} → "
                f"{len(follow_up)} follow-up query(ies)",
            ),
        }

    def compile_node(state: ResearchState) -> dict:
        report = editor.compile_report(state["topic"], state.get("facts", []))
        return {"report": report, "log": _log(state, "📝 Editor compiled final report")}

    def route_after_reflect(state: ResearchState) -> str:
        can_loop = (
            not state.get("sufficient", True)
            and state.get("iteration", 0) < state.get("max_iterations", max_iterations)
            and bool(state.get("pending_queries"))
        )
        return "research" if can_loop else "compile"

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("compile", compile_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "reflect")
    graph.add_conditional_edges(
        "reflect", route_after_reflect, {"research": "research", "compile": "compile"}
    )
    graph.add_edge("compile", END)
    return graph.compile()


def run_research(
    topic: str,
    rag_engine=None,
    max_iterations: int = 2,
    on_event: Optional[Callable[[str], None]] = None,
) -> ResearchState:
    """
    Run the full agentic research pipeline for a topic.

    Returns the final graph state, including ``report``, ``facts``,
    ``all_queries``, ``iteration`` and the decision ``log``.
    """
    graph = build_graph(rag_engine, max_iterations, on_event)
    initial: ResearchState = {
        "topic": topic,
        "facts": [],
        "all_queries": [],
        "pending_queries": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "log": [],
    }
    return graph.invoke(initial)


if __name__ == "__main__":
    # Offline structural check — builds the graph and prints its shape without
    # making any API calls. Model construction needs *a* key present (it does
    # not call the network), so set a dummy one if none is configured.
    import os

    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "sk-offline-structural-test")

    compiled = build_graph()
    g = compiled.get_graph()
    print("Nodes:", [n for n in g.nodes])
    print("Edges:")
    for e in g.edges:
        arrow = f"  {e.source} -> {e.target}"
        if e.conditional:
            arrow += "  (conditional)"
        print(arrow)
