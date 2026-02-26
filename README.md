# 🏥 RAG-Powered Medical Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built on a medical knowledge base sourced from **PubMed**, **MedlinePlus**, and **simulated patient records**. Powered by **LangChain** with **FAISS** vector search and **Mixtral 8×7B** (via Groq) for accurate, grounded medical answers.

> ⚠️ **Disclaimer**: This is a portfolio/demo project. It is NOT a substitute for professional medical advice.

---

## 🏗️ Architecture

```
┌──────────────────┐     HTTP/JSON     ┌──────────────────────────┐
│   Streamlit UI   │ ◄──────────────►  │   FastAPI Backend        │
│  (streamlit_app) │                   │   (app/main.py)          │
└──────────────────┘                   └────────┬─────────────────┘
                                                │
                                       ┌────────▼─────────────────┐
                                       │  LangChain RAG Pipeline  │
                                       │  (app/rag_chain.py)      │
                                       └────────┬─────────────────┘
                                                │
                              ┌─────────────────┼──────────────────┐
                              ▼                 ▼                  ▼
                     ┌──────────────┐  ┌────────────────┐  ┌─────────────┐
                     │ FAISS Vector │  │  Mixtral 8×7B  │  │   Medical   │
                     │    Store     │  │  (Groq API)    │  │   Prompt    │
                     └──────────────┘  └────────────────┘  └─────────────┘
```

## 📚 Knowledge Base

| Source | Documents | Description |
|--------|-----------|-------------|
| **PubMed** | 30 articles | Clinical research abstracts covering diabetes, hypertension, oncology, infectious disease, etc. |
| **MedlinePlus** | 15 articles | Consumer health guides on medications, nutrition, exercise, mental health, etc. |
| **Patient Records** | 15 records | Synthetic clinical notes with conditions, medications, and treatment plans |

## 🛠️ Tech Stack

- **LLM**: Mixtral 8×7B via [Groq](https://groq.com) (free tier)
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers, runs locally)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Orchestration**: LangChain (RetrievalQA chain)
- **Backend**: FastAPI with async endpoints
- **Frontend**: Streamlit with chat interface

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd swift-glenn
pip install -r requirements.txt
```

### 2. Set API Key

Get a free API key from [console.groq.com](https://console.groq.com) and add it to app/config

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Build the Vector Store

```bash
python scripts/build_vectorstore.py
```

This embeds all 60 documents into a FAISS index (~30 seconds on first run).

### 4. Start the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Launch the UI

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📁 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── config.py            # Environment & settings
│   ├── vectorstore.py       # FAISS loader
│   ├── rag_chain.py         # LangChain RAG pipeline + prompt
│   └── main.py              # FastAPI endpoints
├── data/
│   └── raw/
│       ├── pubmed_articles.json
│       ├── medlineplus_articles.json
│       └── patient_records.json
├── scripts/
│   └── build_vectorstore.py # Ingestion pipeline
├── streamlit_app.py         # Chat UI
├── requirements.txt
├── .env.example
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check + model status |
| `POST` | `/query`  | Submit a medical question |

*
