from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

TOP_K = 10


# ============================================================
# LOAD EMBEDDINGS + VECTOR DB
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
    print("\nERROR: ChromaDB is empty.")
    print("Make sure you are running this script from:")
    print(r"C:\Users\HP\Enterprise-RAG")
    print("and make sure the vector_db folder exists there.")
    raise SystemExit


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(documents):
    unique_documents = []
    seen_content = set()

    for document in documents:
        content = document.page_content.strip()

        if content and content not in seen_content:
            seen_content.add(content)
            unique_documents.append(document)

    return unique_documents


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
print("RETRIEVAL BENCHMARK STARTED")
print("=" * 80)


for question_number, question in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {question_number}")
    print("=" * 80)

    print(question)

    try:

        # ----------------------------------------------------
        # DIRECT SIMILARITY SEARCH
        # ----------------------------------------------------

        documents = vectorstore.similarity_search(
            question,
            k=TOP_K
        )

        documents = remove_duplicates(documents)

        print(
            f"\nRetrieved {len(documents)} unique chunks."
        )

        if not documents:

            print("\nNo chunks were retrieved.")

            results_summary.append(
                {
                    "question": question,
                    "success": False,
                    "best_rank": None
                }
            )

            continue


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        for rank, document in enumerate(documents, start=1):

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

            content = document.page_content.strip()

            print("\n" + "-" * 80)

            print(f"RANK {rank}")

            print(f"Source: {source}")
            print(f"Page: {page}")

            print("-" * 80)

            # Show enough text to judge relevance
            print(content[:900])


        # ----------------------------------------------------
        # MANUAL CHECK
        # ----------------------------------------------------

        print("\n" + "=" * 80)

        correct = input(
            "\nWas the correct evidence retrieved? (y/n): "
        ).strip().lower()

        if correct == "y":

            while True:

                best_rank = input(
                    "What was the best rank containing "
                    "the correct evidence? "
                ).strip()

                try:
                    best_rank = int(best_rank)

                    if 1 <= best_rank <= len(documents):
                        break

                    print(
                        f"Please enter a rank between "
                        f"1 and {len(documents)}."
                    )

                except ValueError:
                    print("Please enter a valid number.")

            results_summary.append(
                {
                    "question": question,
                    "success": True,
                    "best_rank": best_rank
                }
            )

            print(
                f"\nRESULT: PASS | Best rank = {best_rank}"
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
            f"{type(error).__name__}: {error}"
        )

        results_summary.append(
            {
                "question": question,
                "success": False,
                "best_rank": None
            }
        )


# ============================================================
# FINAL BENCHMARK SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL RETRIEVAL BENCHMARK SUMMARY")
print("=" * 80)

successful_questions = sum(
    1
    for result in results_summary
    if result["success"]
)

total_questions = len(results_summary)

if total_questions > 0:
    success_rate = (
        successful_questions / total_questions
    ) * 100
else:
    success_rate = 0


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
        if result["best_rank"] is not None
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
    f"{successful_questions}/{total_questions}"
)

print(
    f"Retrieval Success Rate: "
    f"{success_rate:.2f}%"
)

print("=" * 80)