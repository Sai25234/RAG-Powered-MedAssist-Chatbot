"""FastAPI backend for the RAG Medical Chatbot."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_chain import build_rag_chain, query_rag

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global chain reference ─────────────────────────────────────────────────────
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG chain on startup."""
    global rag_chain
    logger.info("Loading RAG chain...")
    try:
        rag_chain = build_rag_chain()
        logger.info("RAG chain loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {e}")
        logger.error("Run 'python scripts/build_vectorstore.py' before starting the server.")
    except Exception as e:
        logger.error(f"Failed to load RAG chain: {e}")
    yield
    logger.info("Shutting down...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Medical Chatbot API",
    description="A retrieval-augmented generation chatbot powered by medical knowledge from PubMed, MedlinePlus, and patient records.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "What are the symptoms and treatment options for Type 2 diabetes?"}
        }


class SourceDocument(BaseModel):
    title: str
    source_type: str
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class HealthResponse(BaseModel):
    status: str
    model: str
    vectorstore_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model="mixtral-8x7b-32768",
        vectorstore_loaded=rag_chain is not None,
    )


@app.post("/query", response_model=QueryResponse, tags=["Chat"])
async def query_endpoint(request: QueryRequest):
    """Process a medical question through the RAG pipeline.

    Retrieves relevant documents from the medical knowledge base
    and generates a grounded answer using Mixtral 8×7B.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialized. Ensure the vector store is built and the GROQ_API_KEY is set.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Received query: {request.question[:100]}...")

    try:
        result = query_rag(rag_chain, request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
