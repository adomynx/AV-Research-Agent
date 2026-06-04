"""
Editor Agent — Compiles collected research facts into a structured,
publication-ready research report with proper citations and analysis.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class EditorAgent:
    """
    The Editor Agent is the final stage of the pipeline.
    It takes all collected facts and compiles them into a structured
    research report with executive summary, analysis, comparison tables,
    and cited sources.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.4,
            max_tokens=4000
        )

        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior technical writer specializing in autonomous 
driving and perception systems research. You compile research facts into 
clear, structured reports.

REPORT STRUCTURE:
1. **Title** — Clear, descriptive research title
2. **Executive Summary** — 3-4 sentence overview of key findings
3. **Background** — Brief context on why this topic matters in AV
4. **Key Findings** — Organized by theme, with quantitative data where available
5. **Architecture Comparison** — Table comparing approaches (if applicable)
6. **Challenges & Limitations** — Real-world deployment concerns
7. **Conclusion & Outlook** — Summary and future directions
8. **Sources** — Numbered list of all sources cited

FORMATTING RULES:
- Use markdown formatting
- Include comparison tables where data supports it
- Cite sources inline as [Source Name]
- Highlight quantitative metrics in bold
- Keep language technical but accessible
- Total length: 800-1200 words"""),
            ("human", """Research Topic: {topic}

Collected Facts:
{facts}

Compile these facts into a structured research report:""")
        ])

    def compile_report(self, topic: str, facts: list[dict]) -> str:
        """
        Compile collected facts into a structured research report.

        Args:
            topic: The original research topic
            facts: List of fact dictionaries from the Research Agent

        Returns:
            Markdown-formatted research report string
        """
        # Format facts for the prompt
        formatted_facts = self._format_facts(facts)

        try:
            response = self.report_prompt | self.llm
            result = response.invoke({
                "topic": topic,
                "facts": formatted_facts
            })
            return result.content

        except Exception as e:
            print(f"[EditorAgent] Report generation error: {e}")
            return self._fallback_report(topic, facts)

    def _format_facts(self, facts: list[dict]) -> str:
        """Format fact dictionaries into readable text for the prompt."""
        lines = []
        for i, fact in enumerate(facts, 1):
            title = fact.get("title", "Untitled")
            content = fact.get("content", "")
            source = fact.get("source", "Unknown")
            relevance = fact.get("relevance", "medium")

            lines.append(
                f"Fact {i} [{relevance.upper()}]:\n"
                f"  Title: {title}\n"
                f"  Content: {content}\n"
                f"  Source: {source}\n"
            )
        return "\n".join(lines)

    def _fallback_report(self, topic: str, facts: list[dict]) -> str:
        """Generate a basic report if LLM compilation fails."""
        report = f"# Research Report: {topic}\n\n"
        report += "## Executive Summary\n"
        report += f"This report presents findings from automated research on **{topic}**.\n\n"
        report += "## Key Findings\n\n"

        for i, fact in enumerate(facts, 1):
            report += f"### {i}. {fact.get('title', 'Finding')}\n"
            report += f"{fact.get('content', '')}\n"
            report += f"*Source: {fact.get('source', 'Unknown')}*\n\n"

        report += "## Sources\n\n"
        sources = list(set(f.get("source", "Unknown") for f in facts))
        for i, source in enumerate(sources, 1):
            report += f"{i}. {source}\n"

        return report
