from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sentence_transformers import CrossEncoder
from agents.verifier import verify_answer

# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DB_PATH = "vector_db"

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

OLLAMA_BASE_URL = "http://localhost:11434"

# Initial similarity retrieval
INITIAL_K = 20

# Final chunks after reranking
FINAL_K = 8
MAX_VERIFICATION_RETRIES = 2
# Local reranker
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
# HELPER FUNCTIONS
# ============================================================

def remove_duplicate_documents(documents):
    """
    Remove duplicate or empty document chunks.
    """

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


def create_retrieval_query(question):
    """
    Expand the user's question into a retrieval-focused query.
    """

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


def rerank_documents(question, documents, reranker):
    """
    Rerank retrieved chunks using a cross-encoder.
    """

    if not documents:
        return []

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

    reranked_documents = [
        document
        for score, document
        in scored_documents[:FINAL_K]
    ]

    return reranked_documents


def build_context(documents):
    """
    Convert reranked documents into one formatted context string.
    """

    context_parts = []

    for index, document in enumerate(documents, start=1):

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

    return "\n\n---\n\n".join(context_parts)


def print_sources(documents):
    """
    Print unique source/page references used for generation.
    """

    seen = set()
    source_number = 1

    print("\nSources:")

    for document in documents:

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

        source_key = (
            source,
            page
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        print(
            f"{source_number}. "
            f"{source} — Page {page}"
        )

        source_number += 1


# ============================================================
# LOAD EMBEDDING MODEL
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
# LOAD GENERATION MODEL
# ============================================================

print("Loading Llama 3.2...")

llm = ChatOllama(
    model=GENERATION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

print("Enterprise RAG Assistant is ready!")


# ============================================================
# QUESTION-ANSWERING LOOP
# ============================================================

while True:

    question = input(
        "\nAsk a question (type 'exit' to quit): "
    ).strip()

    if question.lower() == "exit":
        print("Exiting Enterprise RAG Assistant.")
        break

    if not question:
        print("Please enter a valid question.")
        continue

    print("\nSearching documents...")

    try:

        # ----------------------------------------------------
        # RETRIEVAL QUERY
        # ----------------------------------------------------

        retrieval_query = create_retrieval_query(
            question
        )


        # ----------------------------------------------------
        # INITIAL SIMILARITY RETRIEVAL
        # ----------------------------------------------------

        documents = vectorstore.similarity_search(
            retrieval_query,
            k=INITIAL_K
        )

        documents = remove_duplicate_documents(
            documents
        )

        if not documents:

            print("\n" + "=" * 70)

            print(
                "I could not find this information "
                "in the provided documents."
            )

            print("=" * 70)

            continue


        print(
            f"Initial retrieval: "
            f"{len(documents)} candidate chunks."
        )


        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        documents = rerank_documents(
            question=question,
            documents=documents,
            reranker=reranker
        )

        print(
            f"After reranking: "
            f"{len(documents)} context chunks."
        )


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = build_context(
            documents
        )


        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )


        # ----------------------------------------------------
        # DEBUG: SHOW RERANKED PAGES
        # ----------------------------------------------------

        print("\nReranked pages:")

        for index, document in enumerate(
            documents,
            start=1
        ):

            page = document.metadata.get(
                "page_number",
                document.metadata.get(
                    "page",
                    "Unknown"
                )
            )

            print(
                f"{index}. Page {page}"
            )

                # ----------------------------------------------------
        # GENERATE + VERIFY ANSWER
        # ----------------------------------------------------

        print("\nGenerating answer...\n")

        response = llm.invoke(prompt)

        answer = getattr(
            response,
            "content",
            str(response)
        ).strip()

        if not answer:
            answer = (
                "I could not find this information "
                "in the provided documents."
            )


        # ----------------------------------------------------
        # VERIFY + REVISE LOOP
        # ----------------------------------------------------

        verification = verify_answer(
            question=question,
            answer=answer,
            context=context
        )

        best_answer = answer
        best_verification = verification

        retry_count = 0

        while (
            verification["verdict"] == "REVISE"
            and retry_count < MAX_VERIFICATION_RETRIES
        ):

            retry_count += 1

            print(
                f"\nVerifier requested revision "
                f"(attempt {retry_count}/"
                f"{MAX_VERIFICATION_RETRIES})"
            )

            print(
                "Verifier feedback:",
                verification["feedback"]
            )

            print(
                "Verifier issues:",
                verification["issues"]
            )

            revision_prompt = f"""
You are revising an Enterprise Financial Document answer.

Use ONLY the retrieved evidence below.

Original question:
{question}

Previous answer:
{answer}

Verifier feedback:
{verification["feedback"]}

Verifier issues:
{verification["issues"]}

Retrieved evidence:
{context}

Instructions:

1. Correct every issue identified by the verifier.
2. Preserve exact financial category names.
3. Preserve exact numbers, percentages, and monetary values.
4. Do not introduce outside knowledge.
5. Do not invent information.
6. Do not reverse an increase into a decrease or a decrease into an increase.
7. Preserve cause-and-effect relationships exactly as stated in the evidence.
8. Answer the user's question directly and completely.
9. If evidence is insufficient, say:
   "I could not find this information in the provided documents."

Revised answer:
"""

            revised_response = llm.invoke(
                revision_prompt
            )

            revised_answer = getattr(
                revised_response,
                "content",
                str(revised_response)
            ).strip()

            if not revised_answer:
                continue

            new_verification = verify_answer(
                question=question,
                answer=revised_answer,
                context=context
            )

            if new_verification["verdict"] == "PASS":
                answer = revised_answer
                verification = new_verification
                best_answer = revised_answer
                best_verification = new_verification
                break

            verification = new_verification


        # ----------------------------------------------------
        # RESTORE SAFE ANSWER IF ALL REVISIONS FAILED
        # ----------------------------------------------------

        if verification["verdict"] == "REVISE":
            answer = best_answer
            verification = best_verification


        # ----------------------------------------------------
        # SHOW VERIFICATION RESULT
        # ----------------------------------------------------

        print(
            f"\nVerifier verdict: "
            f"{verification['verdict']}"
        )

        if verification["issues"]:
            print(
                "Verifier issues:",
                verification["issues"]
            )


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        print("=" * 70)

        print(answer)

        print_sources(
            documents
        )

        print("=" * 70)

    except Exception as error:

        print("\nAn error occurred:")

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )