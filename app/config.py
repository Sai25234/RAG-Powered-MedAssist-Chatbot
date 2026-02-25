"""Configuration settings for the RAG Medical Chatbot."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# ── Embedding Settings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Vector Store Settings ──────────────────────────────────────────────────────
VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectorstore")

# ── RAG Settings ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 5

# ── Data Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PUBMED_PATH = os.path.join(DATA_DIR, "pubmed_articles.json")
MEDLINEPLUS_PATH = os.path.join(DATA_DIR, "medlineplus_articles.json")
PATIENT_RECORDS_PATH = os.path.join(DATA_DIR, "patient_records.json")
