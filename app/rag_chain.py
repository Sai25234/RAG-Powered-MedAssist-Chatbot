"""LangChain RAG pipeline for the Medical Chatbot."""

from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from langchain_core.prompts import PromptTemplate

from app.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, RETRIEVAL_TOP_K
from app.vectorstore import load_vectorstore


# ── Medical RAG Prompt ─────────────────────────────────────────────────────────
MEDICAL_PROMPT_TEMPLATE = """You are a knowledgeable and helpful medical assistant. Your role is to provide accurate, evidence-based medical information using ONLY the context provided below. Follow these guidelines strictly:

1. **Grounded Answers**: Base your response ONLY on the provided context. Do not fabricate or hallucinate information.
2. **Source Attribution**: Cite the source of your information (PubMed, MedlinePlus, or Patient Records) when relevant.
3. **Clinical Accuracy**: Use proper medical terminology while keeping explanations accessible.
4. **Safety Disclaimer**: For any clinical questions, remind users to consult their healthcare provider for personalized medical advice.
5. **Uncertainty**: If the provided context does not contain enough information to fully answer the question, clearly state what you can and cannot answer based on the available evidence.

Context from Medical Knowledge Base:
{context}

Patient Question: {question}

Medical Assistant Response:"""

MEDICAL_PROMPT = PromptTemplate(
    template=MEDICAL_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_rag_chain():
    """Build and return the LangChain RetrievalQA chain.

    Returns:
        RetrievalQA: Configured RAG chain with Mixtral 8x7B and FAISS retriever.
    """
    # Load the vector store
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )

    # Initialize Groq LLM (Mixtral 8x7B)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    # Build the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": MEDICAL_PROMPT},
    )

    return chain


def query_rag(chain, question: str) -> dict:
    """Query the RAG chain and format the response.

    Args:
        chain: The RetrievalQA chain instance.
        question: The user's medical question.

    Returs:
        dict: Contains 'answer' (str) and 'sources' (list of dicts).
    """
    result = chain.invoke({"query": question})

    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        meta = doc.metadata
        source_key = meta.get("title", meta.get("patient_id", "Unknown"))
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "title": source_key,
                "source_type": meta.get("source_type", "Unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

    return {
        "answer": result["result"],
        "sources": sources,
    }
