import streamlit as st
import os
import time
from datetime import datetime
from agents.triage import TriageAgent
from agents.researcher import ResearchAgent
from agents.editor import EditorAgent
from utils.pdf_loader import load_pdf_documents
from utils.rag_engine import RAGEngine

st.set_page_config(
    page_title="AV Research Agent",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .agent-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00d2ff;
    }
    .fact-card {
        background: #f8f9fa;
        border-left: 4px solid #00d2ff;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .status-running {
        color: #ffc107;
        font-weight: 600;
    }
    .status-complete {
        color: #28a745;
        font-weight: 600;
    }
    .metric-box {
        background: #f1f3f5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "research_log" not in st.session_state:
    st.session_state.research_log = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for LLM agents")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("### 📄 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDFs for RAG-powered analysis",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload AV/perception papers for deeper context"
    )
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            docs = load_pdf_documents(uploaded_files)
            st.session_state.uploaded_docs = docs
            st.session_state.rag_engine = RAGEngine(docs, api_key)
            st.success(f"✅ {len(uploaded_files)} paper(s) indexed")
    
    st.markdown("---")
    st.markdown("### 🔬 Example Topics")
    
    example_topics = [
        "BEVFusion vs PointPillars for 3D object detection",
        "ODD exit detection strategies in autonomous vehicles",
        "LiDAR-camera fusion architectures for perception",
        "Safety validation methods for AV perception systems",
        "Real-time object detection on edge devices for robotics",
        "Transformer architectures in autonomous driving perception"
    ]
    
    for topic in example_topics:
        if st.button(f"📌 {topic}", key=topic, use_container_width=True):
            st.session_state["selected_topic"] = topic

# --- Main Content ---
st.markdown('<p class="main-header">🚗 AV Research Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-agent research pipeline for autonomous driving & perception systems</p>', unsafe_allow_html=True)

# --- Architecture Overview ---
with st.expander("🏗️ Agent Architecture", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <p class="agent-name">🎯 Triage Agent</p>
            <p>Analyzes the research topic, decomposes it into focused sub-queries, 
            and plans the research strategy across perception, planning, and control domains.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <p class="agent-name">🔍 Research Agent</p>
            <p>Searches the web and uploaded papers (via RAG) for relevant information. 
            Extracts key findings, benchmarks, architectures, and datasets with source attribution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <p class="agent-name">📝 Editor Agent</p>
            <p>Compiles all collected facts into a structured research report with 
            title, executive summary, comparison tables, and cited sources.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- Research Input ---
default_topic = st.session_state.get("selected_topic", "")
research_topic = st.text_input(
    "🔎 Enter your research topic",
    value=default_topic,
    placeholder="e.g., LiDAR-camera fusion architectures for autonomous driving perception"
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    start_research = st.button("🚀 Start Research", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Clear Results", use_container_width=False):
        st.session_state.research_results = None
        st.session_state.research_log = []
        st.rerun()

# --- Research Pipeline ---
if start_research and research_topic:
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    st.session_state.research_log = []
    st.session_state.research_results = None
    
    tab_process, tab_report = st.tabs(["📊 Research Process", "📄 Report"])
    
    with tab_process:
        # --- Phase 1: Triage ---
        st.markdown("### Phase 1: Research Planning")
        with st.spinner("🎯 Triage Agent analyzing topic..."):
            triage = TriageAgent(api_key)
            sub_queries = triage.decompose(research_topic)
            st.session_state.research_log.append({
                "agent": "Triage",
                "action": "Decomposed topic into sub-queries",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        st.success(f"✅ Generated {len(sub_queries)} research sub-queries")
        for i, query in enumerate(sub_queries, 1):
            st.markdown(f"""<div class="fact-card">
                <strong>Sub-query {i}:</strong> {query}
            </div>""", unsafe_allow_html=True)
        
        # --- Phase 2: Research ---
        st.markdown("### Phase 2: Information Gathering")
        all_facts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        researcher = ResearchAgent(api_key, rag_engine=st.session_state.rag_engine)
        
        for i, query in enumerate(sub_queries):
            status_text.markdown(f'<p class="status-running">🔍 Researching: {query}</p>', unsafe_allow_html=True)
            
            facts = researcher.research(query, research_topic)
            all_facts.extend(facts)
            
            st.session_state.research_log.append({
                "agent": "Researcher",
                "action": f"Found {len(facts)} facts for: {query}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            progress_bar.progress((i + 1) / len(sub_queries))
            time.sleep(0.3)
        
        status_text.markdown(f'<p class="status-complete">✅ Collected {len(all_facts)} facts total</p>', unsafe_allow_html=True)
        
        with st.expander(f"📋 All Collected Facts ({len(all_facts)})", expanded=False):
            for fact in all_facts:
                st.markdown(f"""<div class="fact-card">
                    <strong>{fact['title']}</strong><br>
                    {fact['content']}<br>
                    <small>Source: {fact['source']}</small>
                </div>""", unsafe_allow_html=True)
        
        # --- Phase 3: Report Generation ---
        st.markdown("### Phase 3: Report Compilation")
        with st.spinner("📝 Editor Agent compiling report..."):
            editor = EditorAgent(api_key)
            report = editor.compile_report(research_topic, all_facts)
            
            st.session_state.research_results = report
            st.session_state.research_log.append({
                "agent": "Editor",
                "action": "Compiled final research report",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        st.success("✅ Research complete!")
        
        # --- Metrics ---
        st.markdown("### 📊 Research Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Sub-queries", len(sub_queries))
        with m2:
            st.metric("Facts Collected", len(all_facts))
        with m3:
            st.metric("Sources", len(set(f['source'] for f in all_facts)))
        with m4:
            rag_status = "Active" if st.session_state.rag_engine else "Inactive"
            st.metric("RAG Status", rag_status)
        
        # --- Activity Log ---
        with st.expander("📜 Agent Activity Log"):
            for log in st.session_state.research_log:
                st.markdown(f"**[{log['timestamp']}] {log['agent']}** — {log['action']}")
    
    with tab_report:
        if st.session_state.research_results:
            st.markdown(st.session_state.research_results)
            
            st.download_button(
                label="📥 Download Report (.md)",
                data=st.session_state.research_results,
                file_name=f"av_research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )

elif st.session_state.research_results:
    tab_process, tab_report = st.tabs(["📊 Research Process", "📄 Report"])
    
    with tab_process:
        if st.session_state.research_log:
            for log in st.session_state.research_log:
                st.markdown(f"**[{log['timestamp']}] {log['agent']}** — {log['action']}")
        else:
            st.info("Run a research query to see the process.")
    
    with tab_report:
        st.markdown(st.session_state.research_results)
        st.download_button(
            label="📥 Download Report (.md)",
            data=st.session_state.research_results,
            file_name=f"av_research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
    "Built with LangChain, Streamlit & RAG | AV Research Agent v1.0"
    "</p>",
    unsafe_allow_html=True
)
