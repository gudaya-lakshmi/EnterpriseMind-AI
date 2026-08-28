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
# LOAD MODELS + VECTOR DB
# ============================================================

print("Loading embedding model...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

print("Connecting to ChromaDB...")

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

stored_chunks = vectorstore._collection.count()

print(f"Stored chunks in ChromaDB: {stored_chunks}")

if stored_chunks == 0:
    print("ERROR: ChromaDB is empty.")
    raise SystemExit


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

    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    scored_documents = list(
        zip(scores, documents)
    )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_documents[:FINAL_K]


# ============================================================
# BENCHMARK QUESTIONS
# ============================================================

questions = [
    "What is Microsoft's mission?",

    "What is Microsoft's vision for AI?",

    "What is Microsoft’s primary business?",

    "How much did Microsoft's revenue increase in fiscal year 2025?",

    "How did revenue, operating income, and net income change in fiscal year 2025?",

    "What drove Intelligent Cloud revenue growth in fiscal year 2025?",

    "What factors drove Microsoft 365 Commercial revenue growth?",

    "What were the main drivers of More Personal Computing revenue growth?",

    "What changes did Microsoft make to its reportable segments in fiscal year 2025?",

    "What are Microsoft's core values according to the report?"
]


# ============================================================
# RESULTS STORAGE
# ============================================================

results_summary = []


# ============================================================
# RUN BENCHMARK
# ============================================================

print("\n" + "=" * 80)
print("RERANKER RETRIEVAL BENCHMARK STARTED")
print("=" * 80)


for question_number, question in enumerate(
    questions,
    start=1
):

    print("\n" + "=" * 80)
    print(f"QUESTION {question_number}")
    print("=" * 80)

    print(question)

    try:

        # ----------------------------------------------------
        # INITIAL SIMILARITY RETRIEVAL
        # ----------------------------------------------------

        documents = vectorstore.similarity_search(
            question,
            k=INITIAL_K
        )

        documents = remove_duplicates(
            documents
        )

        print(
            f"\nInitial candidate chunks: "
            f"{len(documents)}"
        )


        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        reranked_documents = rerank_documents(
            question,
            documents
        )

        print(
            f"Final reranked chunks: "
            f"{len(reranked_documents)}"
        )


        # ----------------------------------------------------
        # DISPLAY RERANKED RESULTS
        # ----------------------------------------------------

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
                    "Unknown"
                )
            )

            content = (
                document.page_content
                .strip()
            )

            print("\n" + "-" * 80)

            print(
                f"RANK {rank}"
            )

            print(
                f"Reranker Score: "
                f"{float(score):.4f}"
            )

            print(
                f"Source: {source}"
            )

            print(
                f"Page: {page}"
            )

            print("-" * 80)

            print(
                content[:900]
            )


        # ----------------------------------------------------
        # MANUAL CHECK
        # ----------------------------------------------------

        print("\n" + "=" * 80)

        correct = input(
            "\nWas the correct evidence retrieved "
            "after reranking? (y/n): "
        ).strip().lower()

        if correct == "y":

            while True:

                best_rank = input(
                    "What was the best reranked rank "
                    "containing the correct evidence? "
                ).strip()

                try:

                    best_rank = int(
                        best_rank
                    )

                    if (
                        1
                        <= best_rank
                        <= len(reranked_documents)
                    ):
                        break

                    print(
                        f"Enter a rank between "
                        f"1 and "
                        f"{len(reranked_documents)}."
                    )

                except ValueError:

                    print(
                        "Please enter a valid number."
                    )

            results_summary.append(
                {
                    "question": question,
                    "success": True,
                    "best_rank": best_rank
                }
            )

            print(
                f"\nRESULT: PASS | "
                f"Best reranked rank = "
                f"{best_rank}"
            )

        else:

            results_summary.append(
                {
                    "question": question,
                    "success": False,
                    "best_rank": None
                }
            )

            print(
                "\nRESULT: FAIL | "
                "Correct evidence not found."
            )

    except Exception as error:

        print(
            f"\nERROR: "
            f"{type(error).__name__}: "
            f"{error}"
        )

        results_summary.append(
            {
                "question": question,
                "success": False,
                "best_rank": None
            }
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL RERANKER BENCHMARK SUMMARY")
print("=" * 80)

successful_questions = sum(
    1
    for result in results_summary
    if result["success"]
)

total_questions = len(
    results_summary
)

success_rate = (
    successful_questions
    / total_questions
    * 100
    if total_questions
    else 0
)


for index, result in enumerate(
    results_summary,
    start=1
):

    status = (
        "PASS"
        if result["success"]
        else "FAIL"
    )

    rank = (
        result["best_rank"]
        if result["best_rank"]
        is not None
        else "-"
    )

    print(
        f"Q{index}: "
        f"{status} | "
        f"Best Rank: {rank}"
    )


print("-" * 80)

print(
    f"Successful Retrievals: "
    f"{successful_questions}/"
    f"{total_questions}"
)

print(
    f"Retrieval Success Rate: "
    f"{success_rate:.2f}%"
)

print("=" * 80)