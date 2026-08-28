from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sentence_transformers import CrossEncoder


from agents.verifier import verify_answer
from agents.citation_agent import find_citations
from agents.router_agent import route_request
from agents.summarizer_agent import (
    summarize_documents,
    filter_summary_documents,
)

from agents.security_agent import (
    security_agent,
    security_rejection_node,
    route_after_security,
)

from graph.workflow import build_rag_workflow

# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DB_PATH = "vector_db"

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

OLLAMA_BASE_URL = "http://localhost:11434"

INITIAL_K = 20
FINAL_K = 8

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are an Enterprise Financial Document Question-Answering Assistant.

Your task is to answer the user's question using ONLY the information
provided in the context.

Instructions:

1. Read all retrieved context carefully before answering.
2. Combine information from multiple context chunks whenever necessary.
3. Answer the user's exact question directly and completely.

4. When the question asks what drove a change, first state the overall
   change, then state the main drivers and supporting figures that are
   explicitly available in the retrieved context.

5. Preserve the exact names and scope of financial categories.
   Do not combine, shorten, or reinterpret categories that have different
   meanings. For example, "Server products and cloud services" and
   "Server products" must be treated as separate categories.

6. Before producing the final answer, verify that every stated increase,
   decrease, percentage, and monetary value refers to the exact category
   associated with it in the retrieved documents.

7. If the retrieved context completely answers the question,
   provide the complete answer and STOP.

   Only state:
   "The provided documents do not specify the remaining information."
   if an essential part of the question is genuinely missing from every
   retrieved context chunk.

8. Do not infer, speculate, or assume anything that is not explicitly
   stated in the context.

9. Do not use outside knowledge.

10. Do not mention the context or say phrases such as
    "according to the provided context."

11. Keep the answer concise, factual, focused, and well-organized.

12. When answering financial questions, prefer reporting values exactly
    as they appear in the retrieved context. Do not compute differences
    unless they are explicitly stated in the retrieved documents.

13. When financial tables are labeled "In millions", preserve the units
    exactly as written. Do not change or reinterpret the units.

14. Use the fallback response only when the answer is completely absent
    from every retrieved chunk.

Fallback response:

I could not find this information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# HELPERS
# ============================================================

def remove_duplicate_documents(documents):

    unique_documents = []
    seen_content = set()

    for document in documents:

        content = document.page_content.strip()

        if not content:
            continue

        if content in seen_content:
            continue

        seen_content.add(content)

        unique_documents.append(
            document
        )

    return unique_documents


def create_retrieval_query(question):

    return f"""
Retrieve the document passages required to answer the following question.

User question:
{question}

Retrieval instructions:

- Retrieve passages that directly answer the question.
- Include supporting facts, values, dates, percentages, and explanations.
- If the question concerns company financial performance, prioritize
  consolidated company-level results before individual business segments.
- Also retrieve nearby supporting passages when the complete answer may
  span multiple chunks.
"""


def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        content = document.page_content.strip()

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page_number",
            document.metadata.get(
                "page",
                "Unknown page"
            )
        )

        context_parts.append(
            f"[Context Chunk {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# PRINT VERIFIED CITATIONS
# ============================================================

def print_verified_citations(citations):

    print(
        "\nVerified Sources:"
    )

    if not citations:

        print(
            "No direct supporting citation found."
        )

        return

    for index, citation in enumerate(
        citations,
        start=1
    ):

        print(
            f"{index}. "
            f"{citation.get('source', 'Unknown source')} "
            f"— Page "
            f"{citation.get('page', 'Unknown page')}"
        )


# ============================================================
# LOAD COMPONENTS
# ============================================================

print(
    "Loading embedding model..."
)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)


print(
    "Connecting to ChromaDB..."
)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

print(
    "Stored chunks:",
    vectorstore._collection.count()
)


print(
    "Loading reranker..."
)

reranker = CrossEncoder(
    RERANKER_MODEL
)

print(
    "Reranker ready!"
)


print(
    "Loading Llama 3.2..."
)

llm = ChatOllama(
    model=GENERATION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)


# ============================================================
# LANGGRAPH ADAPTERS
# ============================================================

class RetrievalAdapter:

    """
    Provides the .invoke() interface expected by workflow.py
    while preserving the same ChromaDB similarity-search
    behavior as the original generator.
    """

    def invoke(
        self,
        retrieval_query
    ):

        documents = vectorstore.similarity_search(
            retrieval_query,
            k=INITIAL_K
        )

        documents = remove_duplicate_documents(
            documents
        )

        return documents


retriever = RetrievalAdapter()


def graph_rerank_documents(
    question,
    documents
):

    """
    Rerank retrieved chunks using the CrossEncoder
    and keep the FINAL_K most relevant chunks.
    """

    if not documents:
        return []

    pairs = [
        (
            question,
            document.page_content
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    scored_documents = list(
        zip(
            scores,
            documents
        )
    )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        document
        for score, document
        in scored_documents[:FINAL_K]
    ]


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================
print(
    "Building LangGraph workflow..."
)

app = build_rag_workflow(

    retriever=retriever,

    rerank_documents=(
        graph_rerank_documents
    ),

    create_retrieval_query=(
        create_retrieval_query
    ),

    build_context=(
        build_context
    ),

    llm=llm,

    prompt_template=(
        PROMPT_TEMPLATE
    ),

    verify_answer=(
        verify_answer
    ),
    # Security Agent
    security_agent=(
        security_agent
    ),

    security_rejection_node=(
        security_rejection_node
    ),

    route_after_security=(
        route_after_security
    ),

    # Citation Agent
    find_citations=(
        find_citations
    ),

    # Router Agent
    route_request=(
        route_request
    ),

        # Summarizer Agent
    summarize_documents=(
        summarize_documents
    ),

    # Summary Scope Filter
    filter_summary_documents=(
        filter_summary_documents
    ),
)

print(
    "Enterprise Agentic RAG workflow is ready!"
)
# ============================================================
# QUESTION-ANSWERING LOOP
# ============================================================

while True:

    question = input(
        "\nAsk a question "
        "(type 'exit' to quit): "
    ).strip()


    if question.lower() == "exit":

        print(
            "Exiting Enterprise Agentic RAG."
        )

        break


    if not question:

        print(
            "Please enter a valid question."
        )

        continue


    try:

        # ----------------------------------------------------
        # RUN LANGGRAPH
        # ----------------------------------------------------

        result = app.invoke(
            {
                "question": question,
                "retry_count": 0,
            }
        )


        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        answer = result.get(
            "final_answer",
            result.get(
                "answer",
                (
                    "I could not find this information "
                    "in the provided documents."
                )
            )
        )


        # ----------------------------------------------------
        # VERIFICATION RESULT
        # ----------------------------------------------------

        verification = result.get(
            "verification",
            {}
        )


        # ----------------------------------------------------
        # CITATION RESULT
        # ----------------------------------------------------

        citations = result.get(
            "citations",
            []
        )


        # ----------------------------------------------------
        # DISPLAY FINAL RESULT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            answer
        )


        # Only Citation Agent-selected sources
        print_verified_citations(
            citations
        )


        print(
            "\nFinal verifier verdict:",
            verification.get(
                "verdict",
                "UNKNOWN"
            )
        )


        print(
            "Citation support:",
            (
                "FOUND"
                if citations
                else "NOT FOUND"
            )
        )


        print(
            "=" * 70
        )


    except Exception as error:

        print(
            "\nAn error occurred:"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )