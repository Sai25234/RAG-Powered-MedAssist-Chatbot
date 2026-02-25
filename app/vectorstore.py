"""FAISS vector store loader for the RAG Medical Chatbot."""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL_NAME, VECTORSTORE_DIR


def get_embeddings():
    """Initialize and return the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore() -> FAISS:
    """Load the persisted FAISS vector store from disk.

    Returns:
        FAISS: The loaded vector store ready for similarity search.

    Raises:
        FileNotFoundError: If the vector store has not been built yet.
    """
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_DIR}'. "
            "Run 'python scripts/build_vectorstore.py' first."
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
