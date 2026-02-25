"""
Build the FAISS vector store from medical knowledge base documents.

This script:
1. Loads PubMed articles, MedlinePlus articles, and patient records from JSON
2. Formats each document with metadata (source type, title)
3. Splits documents using RecursiveCharacterTextSplitter
4. Embeds using HuggingFace sentence-transformers (all-MiniLM-L6-v2)
5. Persists FAISS index to data/vectorstore/

Usage:
    python scripts/build_vectorstore.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTORSTORE_DIR,
    PUBMED_PATH,
    MEDLINEPLUS_PATH,
    PATIENT_RECORDS_PATH,
)
from app.vectorstore import get_embeddings


def load_pubmed_articles(filepath: str) -> list[Document]:
    """Load PubMed articles and convert to LangChain Documents."""
    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)

    documents = []
    for article in articles:
        content = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                "source_type": "PubMed",
                "title": article["title"],
                "pmid": article.get("pmid", ""),
            },
        )
        documents.append(doc)

    print(f"  ✓ Loaded {len(documents)} PubMed articles")
    return documents


def load_medlineplus_articles(filepath: str) -> list[Document]:
    """Load MedlinePlus articles and convert to LangChain Documents."""
    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)

    documents = []
    for article in articles:
        content = f"Title: {article['title']}\n\n{article['content']}"
        doc = Document(
            page_content=content,
            metadata={
                "source_type": "MedlinePlus",
                "title": article["title"],
                "url": article.get("url", ""),
            },
        )
        documents.append(doc)

    print(f"  ✓ Loaded {len(documents)} MedlinePlus articles")
    return documents


def load_patient_records(filepath: str) -> list[Document]:
    """Load synthetic patient records and convert to LangChain Documents."""
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    documents = []
    for record in records:
        conditions = ", ".join(record["conditions"])
        medications = ", ".join(record["medications"])
        content = (
            f"Patient ID: {record['patient_id']}\n"
            f"Age: {record['age']} | Gender: {record['gender']}\n"
            f"Conditions: {conditions}\n"
            f"Medications: {medications}\n\n"
            f"Clinical Notes: {record['clinical_notes']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "source_type": "Patient Records",
                "title": f"Patient {record['patient_id']}",
                "patient_id": record["patient_id"],
            },
        )
        documents.append(doc)

    print(f"  ✓ Loaded {len(documents)} patient records")
    return documents


def build_vectorstore():
    """Build and persist the FAISS vector store."""
    print("=" * 60)
    print("  RAG Medical Chatbot — Vector Store Builder")
    print("=" * 60)

    # ── Step 1: Load all documents ─────────────────────────────────────────
    print("\n📂 Loading documents...")
    all_documents = []
    all_documents.extend(load_pubmed_articles(PUBMED_PATH))
    all_documents.extend(load_medlineplus_articles(MEDLINEPLUS_PATH))
    all_documents.extend(load_patient_records(PATIENT_RECORDS_PATH))
    print(f"\n  Total documents loaded: {len(all_documents)}")

    # ── Step 2: Split documents into chunks ────────────────────────────────
    print(f"\n✂️  Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"  ✓ Created {len(chunks)} chunks")

    # ── Step 3: Create embeddings and FAISS index ──────────────────────────
    print(f"\n🧠 Creating embeddings (this may take a moment on first run)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  ✓ FAISS index created with {vectorstore.index.ntotal} vectors")

    # ── Step 4: Persist to disk ────────────────────────────────────────────
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"\n💾 Vector store saved to: {VECTORSTORE_DIR}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Vector store built successfully!")
    print(f"  • Documents: {len(all_documents)}")
    print(f"  • Chunks:    {len(chunks)}")
    print(f"  • Vectors:   {vectorstore.index.ntotal}")
    print(f"  • Location:  {VECTORSTORE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build_vectorstore()
