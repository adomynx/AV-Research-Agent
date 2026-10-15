"""
Reflection Agent — the component that gives the pipeline *agency*.

After a round of research, it inspects the facts collected so far and decides,
like a human analyst would: "Is this enough to write a solid report, or is
there an obvious gap I should go back and investigate?"

If there's a gap, it emits targeted follow-up queries. The graph feeds those
back into the Research agent, so the system adaptively digs deeper on hard
topics instead of always running the same fixed set of searches.
"""

from langchain_core.prompts import ChatPromptTemplate

from schemas.models import Fact, ReflectionVerdict
from utils.llm import get_chat_model

SYSTEM_PROMPT = """You are a meticulous research lead for autonomous driving and
perception systems. You are given a research TOPIC and the FACTS gathered so far.

Decide whether the facts are sufficient to write a thorough, balanced technical
report that covers: core architecture/approach, quantitative benchmarks, and
real-world deployment trade-offs (and safety, if relevant).

Be demanding but realistic:
- If a major angle is missing (e.g. no benchmark numbers, only one side of a
  comparison, no deployment/limitations), mark it NOT sufficient and propose up
  to 3 targeted follow-up search queries that would close the biggest gaps.
- If the facts already cover the topic well, mark it sufficient and return no
  follow-up queries.

Do not request follow-ups just to pad the report."""


class ReflectionAgent:
    """Assesses evidence coverage and proposes follow-up research when needed."""

    def __init__(self):
        llm = get_chat_model(temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "TOPIC: {topic}\n\nFACTS COLLECTED ({n_facts}):\n{facts}"),
        ])
        self.chain = self.prompt | llm.with_structured_output(ReflectionVerdict)

    def assess(self, topic: str, facts: list[Fact]) -> ReflectionVerdict:
        """
        Judge whether the collected facts are enough, or more research is needed.

        Falls back to "sufficient" on error so the pipeline always terminates.
        """
        facts_text = "\n".join(
            f"- [{f.relevance.value}] {f.title}: {f.content} (source: {f.source})"
            for f in facts
        ) or "(no facts collected yet)"

        try:
            return self.chain.invoke({
                "topic": topic,
                "n_facts": len(facts),
                "facts": facts_text,
            })
        except Exception as e:  # noqa: BLE001
            print(f"[ReflectionAgent] Error: {e}")
            return ReflectionVerdict(sufficient=True, missing_aspects=[], follow_up_queries=[])
