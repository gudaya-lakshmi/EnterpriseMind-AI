from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DB_PATH = "vector_db"

EMBEDDING_MODEL = "nomic-embed-text"

OLLAMA_BASE_URL = "http://localhost:11434"

INITIAL_K = 20
FINAL_K = 8

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

print(
    "Stored chunks:",
    vectorstore._collection.count()
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker ready!")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(documents):

    unique_documents = []

    seen_content = set()

    for document in documents:

        content = document.page_content.strip()

        if not content:
            continue

        if content in seen_content:
            continue

        seen_content.add(content)

        unique_documents.append(document)

    return unique_documents


# ============================================================
# RERANK FUNCTION
# ============================================================

def rerank_documents(question, documents):

    pairs = []

    for document in documents:

        pairs.append(
            (
                question,
                document.page_content
            )
        )

    scores = reranker.predict(
        pairs
    )

    scored_documents = []

    for document, score in zip(
        documents,
        scores
    ):

        scored_documents.append(
            (
                float(score),
                document
            )
        )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_documents[:FINAL_K]


# ============================================================
# ASK QUESTION
# ============================================================

question = input(
    "\nEnter your question: "
).strip()


# ============================================================
# INITIAL RETRIEVAL
# ============================================================

print("\nInitial similarity search...")

documents = vectorstore.similarity_search(
    question,
    k=INITIAL_K
)

documents = remove_duplicates(
    documents
)

print(
    f"Retrieved "
    f"{len(documents)} "
    f"candidate chunks."
)


# ============================================================
# RERANK
# ============================================================

print("\nReranking chunks...")

reranked_documents = rerank_documents(
    question,
    documents
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)

print(
    f"TOP {len(reranked_documents)} "
    f"RERANKED RESULTS"
)

print("=" * 80)


for rank, (
    score,
    document
) in enumerate(
    reranked_documents,
    start=1
):

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

    print("\n" + "-" * 80)

    print(
        f"RANK {rank}"
    )

    print(
        f"Reranker Score: "
        f"{score:.4f}"
    )

    print(
        f"Source: {source}"
    )

    print(
        f"Page: {page}"
    )

    print("-" * 80)

    print(
        document.page_content[:900]
    )


print("\n" + "=" * 80)