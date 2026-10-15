import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # pick up provider + API keys from .env for the running app

from agents.graph import run_research
from utils.pdf_loader import load_pdf_documents
from utils.rag_engine import RAGEngine
from utils.observability import track_run, enable_langsmith
from utils.llm import has_llm_credentials, provider_label

st.set_page_config(
    page_title="AV Research Agent",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide default chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

:root {
    --accent: #22d3ee;
    --indigo: #6366f1;
    --violet: #a855f7;
    --amber: #f59e0b;
    --green: #34d399;
    --surface: #14172a;
    --surface-2: #1b1f38;
    --border: #2a2f52;
    --muted: #96a0c8;
}

.block-container { padding-top: 2rem; max-width: 1100px; }

/* ---- Hero ---- */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #0e7490 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 10px 40px rgba(34, 211, 238, 0.08);
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0; color: #fff; letter-spacing: -0.02em; }
.hero p { color: #cbd5f5; margin: 0.35rem 0 0; font-size: 1.02rem; }

.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    color: #e7eaf6; font-size: 0.8rem; font-weight: 500;
    padding: 0.32rem 0.7rem; border-radius: 999px; backdrop-filter: blur(6px);
}

/* ---- Agent cards ---- */
.agent-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.8rem; }
.agent-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.1rem; position: relative; overflow: hidden;
    transition: transform .15s ease, border-color .15s ease;
}
.agent-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.agent-card::before { content:""; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--indigo), var(--accent)); }
.agent-card .ic { font-size: 1.5rem; }
.agent-card .nm { font-weight: 700; color: #fff; margin: 0.4rem 0 0.3rem; font-size: 0.98rem; }
.agent-card .ds { color: var(--muted); font-size: 0.82rem; line-height: 1.45; }

/* ---- Timeline (decision trace) ---- */
.timeline { margin: 0.4rem 0 0.6rem; }
.tl-item {
    position: relative; padding: 0.7rem 0.9rem 0.7rem 2.4rem;
    border-left: 2px solid var(--border); margin-left: 0.6rem;
}
.tl-item:last-child { border-left-color: transparent; }
.tl-item::before {
    content: ""; position: absolute; left: -0.56rem; top: 0.95rem;
    width: 1rem; height: 1rem; border-radius: 50%;
    background: var(--dot, var(--accent)); box-shadow: 0 0 0 4px rgba(34,211,238,0.12);
}
.tl-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--dot, var(--accent));
    border-radius: 10px; padding: 0.6rem 0.85rem; color: #dfe4f7; font-size: 0.88rem; line-height: 1.5;
}
.tl-plan    { --dot: var(--indigo); }
.tl-research{ --dot: var(--accent); }
.tl-reflect { --dot: var(--amber); }
.tl-compile { --dot: var(--green); }

/* ---- Stat tiles ---- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.7rem; margin: 0.3rem 0 0.6rem; }
.stat-tile {
    background: var(--surface); border: 1px solid var(--border); border-radius: 13px;
    padding: 0.9rem 1rem; text-align: center; position: relative; overflow: hidden;
}
.stat-tile::after { content:""; position:absolute; bottom:0; left:0; right:0; height:3px; background: var(--accent2, var(--accent)); opacity:0.8; }
.stat-val { font-size: 1.6rem; font-weight: 800; color: #fff; line-height: 1; }
.stat-lbl { font-size: 0.72rem; color: var(--muted); margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.04em; }

/* ---- Fact / query cards ---- */
.fact-card {
    background: var(--surface-2); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); padding: 0.7rem 0.9rem; margin: 0.45rem 0;
    border-radius: 0 10px 10px 0; color: #dfe4f7; font-size: 0.86rem;
}
.fact-card small { color: var(--muted); }
.rel-high { color: var(--green); font-weight: 600; }
.rel-medium { color: var(--amber); font-weight: 600; }
.rel-low { color: var(--muted); }

/* ---- Report paper ---- */
.report-paper {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.6rem 2rem; margin-top: 0.5rem;
}

.status-running { color: var(--amber); font-weight: 600; }
.status-complete { color: var(--green); font-weight: 700; }

/* primary button polish */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, var(--indigo), var(--accent));
    border: none; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _chip(text: str) -> str:
    return f'<span class="chip">{text}</span>'


def _stage_class(line: str) -> str:
    if "Triage" in line:
        return "tl-plan"
    if "Research round" in line:
        return "tl-research"
    if "Reflection" in line:
        return "tl-reflect"
    if "Editor" in line or "compiled" in line:
        return "tl-compile"
    return "tl-research"


def _timeline_html(log: list[str]) -> str:
    items = "".join(
        f'<div class="tl-item {_stage_class(line)}"><div class="tl-card">{line}</div></div>'
        for line in log
    )
    return f'<div class="timeline">{items}</div>'


def _stats_html(tiles: list[tuple]) -> str:
    cells = "".join(
        f'<div class="stat-tile" style="--accent2:{accent}">'
        f'<div class="stat-val">{value}</div><div class="stat-lbl">{label}</div></div>'
        for label, value, accent in tiles
    )
    return f'<div class="stat-grid">{cells}</div>'


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for key, default in [
    ("research_results", None), ("research_log", []),
    ("rag_engine", None), ("uploaded_docs", []), ("run_meta", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    _default_idx = 1 if os.getenv("LLM_PROVIDER", "openai").lower() == "anthropic" else 0
    provider = st.radio(
        "LLM Provider",
        options=["OpenAI", "Anthropic (Claude)"],
        index=_default_idx,
        help="Pick the model provider. The pipeline is provider-agnostic.",
    )
    os.environ["LLM_PROVIDER"] = "anthropic" if provider.startswith("Anthropic") else "openai"

    if os.environ["LLM_PROVIDER"] == "anthropic":
        api_key = st.text_input("Anthropic API Key", type="password",
                                 help="sk-ant-... from console.anthropic.com")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    else:
        api_key = st.text_input("OpenAI API Key", type="password",
                                help="sk-... from platform.openai.com")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    if has_llm_credentials():
        st.success(f"✅ {provider_label()}", icon="🔑")
    else:
        st.warning("Add an API key to run research.")

    st.caption("📎 Embeddings run locally (free) — RAG needs no API key.")

    st.markdown("---")
    st.markdown("### 📄 Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDFs for RAG-powered analysis",
        type=["pdf"], accept_multiple_files=True,
        help="Upload AV/perception papers for deeper, cited context",
    )
    if uploaded_files:
        with st.spinner("Indexing PDFs (hybrid retrieval)..."):
            docs = load_pdf_documents(uploaded_files)
            st.session_state.uploaded_docs = docs
            st.session_state.rag_engine = RAGEngine(docs)
            st.success(f"✅ {len(uploaded_files)} paper(s) indexed")

    st.markdown("---")
    st.markdown("### 🔬 Example Topics")
    example_topics = [
        "BEVFusion vs PointPillars for 3D object detection",
        "ODD exit detection strategies in autonomous vehicles",
        "LiDAR-camera fusion architectures for perception",
        "Safety validation methods for AV perception systems",
        "Transformer architectures in autonomous driving perception",
    ]
    for topic in example_topics:
        if st.button(topic, key=topic, use_container_width=True):
            st.session_state["selected_topic"] = topic

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
rag_chip = "🟢 RAG active" if st.session_state.rag_engine else "⚪ RAG idle"
trace_chip = "🟢 LangSmith on" if enable_langsmith() else "⚪ Tracing off"
st.markdown(f"""
<div class="hero">
    <h1>🚗 AV Research Agent</h1>
    <p>An agentic research pipeline for autonomous driving &amp; perception — it plans, searches, reflects on gaps, and compiles cited reports.</p>
    <div class="chip-row">
        {_chip("🧠 LangGraph · reflection loop")}
        {_chip("🔀 Hybrid RAG + rerank")}
        {_chip("🧩 " + provider_label())}
        {_chip(rag_chip)}
        {_chip(trace_chip)}
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🏗️  How it works — the agent architecture"):
    st.markdown("""
    <div class="agent-grid">
      <div class="agent-card"><div class="ic">🎯</div><div class="nm">Triage</div>
        <div class="ds">Decomposes the topic into 4–6 focused, searchable sub-queries.</div></div>
      <div class="agent-card"><div class="ic">🔍</div><div class="nm">Research</div>
        <div class="ds">Web search + hybrid RAG (BM25 + dense, reranked); extracts typed facts with sources.</div></div>
      <div class="agent-card"><div class="ic">🤔</div><div class="nm">Reflection</div>
        <div class="ds">Judges evidence coverage and loops back with follow-up queries on gaps.</div></div>
      <div class="agent-card"><div class="ic">📝</div><div class="nm">Editor</div>
        <div class="ds">Compiles grounded facts into a structured, cited report.</div></div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
default_topic = st.session_state.get("selected_topic", "")
research_topic = st.text_input(
    "🔎 Research topic",
    value=default_topic,
    placeholder="e.g., LiDAR-camera fusion architectures for autonomous driving perception",
)

col_btn1, col_btn2, _ = st.columns([2, 2, 6])
with col_btn1:
    start_research = st.button("🚀 Start Research", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.research_results = None
        st.session_state.research_log = []
        st.session_state.run_meta = None
        st.rerun()


def _render_process(decision_log, sub_queries, all_facts, meta):
    st.markdown("#### 🔁 Agent Decision Trace")
    st.caption("Plan → Research → Reflect ↺ → Compile. The Reflect step loops back when it finds evidence gaps.")
    st.markdown(_timeline_html(decision_log), unsafe_allow_html=True)

    st.markdown("#### 📊 Run Metrics")
    st.markdown(_stats_html([
        ("Research Rounds", meta["rounds"], "#6366f1"),
        ("Queries", len(sub_queries), "#22d3ee"),
        ("Facts", len(all_facts), "#a855f7"),
        ("Sources", meta["sources"], "#34d399"),
        ("Tokens", f"{meta['tokens']:,}", "#f59e0b"),
        ("Est. Cost", f"${meta['cost']:.4f}", "#f472b6"),
    ]), unsafe_allow_html=True)

    with st.expander(f"🧭 Queries executed ({len(sub_queries)})"):
        for i, q in enumerate(sub_queries, 1):
            st.markdown(f'<div class="fact-card"><strong>Query {i}:</strong> {q}</div>', unsafe_allow_html=True)

    with st.expander(f"📋 Collected facts ({len(all_facts)})"):
        for f in all_facts:
            rel = f.relevance.value
            st.markdown(
                f'<div class="fact-card"><strong>{f.title}</strong> '
                f'<span class="rel-{rel}">● {rel}</span><br>{f.content}<br>'
                f'<small>Source: {f.source}</small></div>',
                unsafe_allow_html=True,
            )


def _render_report(report: str):
    if not report:
        st.info("Run a research query to generate a report.")
        return
    st.markdown(f'<div class="report-paper">{report}</div>', unsafe_allow_html=True)
    st.download_button(
        "📥 Download Report (.md)", data=report,
        file_name=f"av_research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
if start_research and research_topic:
    if not has_llm_credentials():
        st.error("⚠️ Please add your API key for the selected provider in the sidebar.")
        st.stop()

    tab_process, tab_report = st.tabs(["📊 Research Process", "📄 Report"])

    with tab_process:
        status = st.empty()
        progress = st.progress(0, text="Starting…")
        steps = {"n": 0}

        def on_event(msg: str):
            steps["n"] += 1
            status.markdown(f'<p class="status-running">⏳ {msg}</p>', unsafe_allow_html=True)
            progress.progress(min(steps["n"] / 6, 0.95), text=msg)

        with track_run("research_graph") as run_stats:
            final_state = run_research(
                research_topic, rag_engine=st.session_state.rag_engine,
                max_iterations=2, on_event=on_event,
            )
        progress.progress(1.0, text="Complete")
        status.markdown('<p class="status-complete">✅ Research complete</p>', unsafe_allow_html=True)

        sub_queries = final_state.get("all_queries", [])
        all_facts = final_state.get("facts", [])
        decision_log = final_state.get("log", [])
        meta = {
            "rounds": final_state.get("iteration", 0),
            "sources": len({f.source for f in all_facts}),
            "tokens": run_stats.total_tokens,
            "cost": run_stats.total_cost_usd,
        }

        st.session_state.research_results = final_state.get("report", "")
        st.session_state.research_log = decision_log
        st.session_state.run_meta = {"queries": sub_queries, "facts": all_facts, **meta}

        _render_process(decision_log, sub_queries, all_facts, meta)

    with tab_report:
        _render_report(st.session_state.research_results)

elif st.session_state.research_results:
    tab_process, tab_report = st.tabs(["📊 Research Process", "📄 Report"])
    with tab_process:
        m = st.session_state.run_meta or {}
        _render_process(
            st.session_state.research_log,
            m.get("queries", []), m.get("facts", []),
            {"rounds": m.get("rounds", 0), "sources": m.get("sources", 0),
             "tokens": m.get("tokens", 0), "cost": m.get("cost", 0.0)},
        )
    with tab_report:
        _render_report(st.session_state.research_results)

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown(
    "<p style='text-align:center; color:#5b6485; font-size:0.8rem; margin-top:2rem;'>"
    "Built with LangGraph · LangChain · hybrid RAG · provider-agnostic (OpenAI ↔ Claude)"
    "</p>",
    unsafe_allow_html=True,
)
