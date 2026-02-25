"""
Streamlit UI for the RAG Medical Chatbot.

A chat-style interface that sends queries to the FastAPI backend
and displays AI-generated medical answers with source citations.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import time

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI — Medical Chatbot",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ─────────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #0D9488, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* ── Source cards ────────────────────────────────────────────────── */
    .source-card {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.4rem 0;
    }
    .source-card .source-type {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .source-pubmed { background: #1E3A5F; color: #60A5FA; }
    .source-medlineplus { background: #1A3A2A; color: #4ADE80; }
    .source-patient { background: #3B1F4B; color: #C084FC; }
    .source-card .source-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #E2E8F0;
        margin-top: 4px;
    }
    .source-card .source-preview {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4px;
        line-height: 1.4;
    }

    /* ── Disclaimer banner ──────────────────────────────────────────── */
    .disclaimer {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #CBD5E1;
        margin-bottom: 1rem;
    }

    /* ── Sidebar stats ──────────────────────────────────────────────── */
    .stat-box {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-box .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0D9488, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-box .stat-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ── Backend Configuration ──────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


def check_backend_health() -> dict | None:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.ConnectionError:
        return None
    return None


def query_backend(question: str) -> dict | None:
    """Send a question to the FastAPI backend."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code} — {response.text}")
            return None
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend. Make sure the FastAPI server is running.")
        return None
    except requests.Timeout:
        st.error("⚠️ Request timed out. The backend may be overloaded.")
        return None


def render_source_card(source: dict):
    """Render a source citation card."""
    source_type = source.get("source_type", "Unknown")
    css_class = {
        "PubMed": "source-pubmed",
        "MedlinePlus": "source-medlineplus",
        "Patient Records": "source-patient",
    }.get(source_type, "source-pubmed")

    st.markdown(f"""
    <div class="source-card">
        <span class="source-type {css_class}">{source_type}</span>
        <div class="source-title">{source.get("title", "Unknown")}</div>
        <div class="source-preview">{source.get("content_preview", "")}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAssist AI")
    st.markdown("---")

    # Backend status
    health = check_backend_health()
    if health:
        st.success("✅ Backend Connected")
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">LLM Model</div>
            <div class="stat-number" style="font-size: 0.9rem;">Mixtral 8×7B</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Backend Offline")
        st.caption("Start the server with:")
        st.code("uvicorn app.main:app --reload", language="bash")

    st.markdown("---")

    # Knowledge base info
    st.markdown("### 📚 Knowledge Base")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">30</div>
            <div class="stat-label">PubMed Articles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">15</div>
            <div class="stat-label">MedlinePlus</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">15</div>
        <div class="stat-label">Patient Records</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Queries counter
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{st.session_state.query_count}</div>
        <div class="stat-label">Queries This Session</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Example queries
    st.markdown("### 💡 Try Asking")
    example_queries = [
        "What are the symptoms of Type 2 diabetes?",
        "How is hypertension diagnosed and treated?",
        "Explain the management of COPD",
        "What medications are used for heart failure?",
        "Tell me about migraine prevention",
    ]
    for eq in example_queries:
        if st.button(eq, key=f"example_{eq[:20]}", use_container_width=True):
            st.session_state.example_query = eq

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()

# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MedAssist AI</h1>
    <p>RAG-powered Medical Chatbot • PubMed • MedlinePlus • Patient Records</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This chatbot provides general medical information for educational purposes only.
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# ── Chat History ───────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📄 Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    render_source_card(source)

# ── Handle example query from sidebar ──────────────────────────────────────────
if "example_query" in st.session_state:
    prompt = st.session_state.pop("example_query")
else:
    prompt = st.chat_input("Ask a medical question...")

# ── Process Query ──────────────────────────────────────────────────────────────
if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching medical knowledge base..."):
            start_time = time.time()
            result = query_backend(prompt)
            elapsed = time.time() - start_time

        if result:
            st.session_state.query_count += 1

            # Display answer
            answer = result["answer"]
            st.markdown(answer)
            st.caption(f"⏱️ Response time: {elapsed:.1f}s")

            # Display sources
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📄 Sources ({len(sources)})"):
                    for source in sources:
                        render_source_card(source)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            error_msg = "Sorry, I couldn't process your question. Please check that the backend is running."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
