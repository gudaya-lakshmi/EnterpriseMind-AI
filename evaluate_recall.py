import os
import csv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from deepeval.metrics import ContextualRecallMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from sentence_transformers import CrossEncoder


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "evaluation_dataset.csv"
VECTOR_DB_PATH = "vector_db"

OUTPUT_FILE = "contextual_recall_7_13.csv"

EMBEDDING_MODEL = "nomic-embed-text"
JUDGE_MODEL = "gemini-2.5-flash"

OLLAMA_BASE_URL = "http://localhost:11434"

INITIAL_K = 20
FINAL_K = 8

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

METRIC_THRESHOLD = 0.5

# Evaluate ONLY questions 7 to 13
START_QUESTION = 7
END_QUESTION = 13


# ============================================================
# DATASET
# ============================================================

def load_dataset():

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {
        "question",
        "ground_truth"
    }

    missing = required_columns - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    dataset = dataset.dropna(
        subset=["question", "ground_truth"]
    ).reset_index(drop=True)

    dataset["question"] = (
        dataset["question"]
        .astype(str)
        .str.strip()
    )

    dataset["ground_truth"] = (
        dataset["ground_truth"]
        .astype(str)
        .str.strip()
    )

    return dataset


# ============================================================
# RETRIEVAL QUERY
# ============================================================

def create_retrieval_query(question):

    return f"""
Retrieve the document passages required to answer the following question.

User question:
{question}

Retrieval instructions:

- Retrieve passages that directly answer the question.
- Include supporting facts, values, dates, percentages, and explanations.
- Retrieve nearby supporting passages when the complete answer may span
  multiple chunks.
"""


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(documents):

    unique_documents = []
    seen = set()

    for document in documents:

        content = document.page_content.strip()

        if not content:
            continue

        if content in seen:
            continue

        seen.add(content)
        unique_documents.append(document)

    return unique_documents


# ============================================================
# RERANKING
# ============================================================

def rerank_documents(
    question,
    documents,
    reranker,
):

    if not documents:
        return []

    pairs = [
        [
            question,
            document.page_content
        ]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    scored_documents = list(
        zip(documents, scores)
    )

    scored_documents.sort(
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        document
        for document, score
        in scored_documents[:FINAL_K]
    ]


# ============================================================
# RETRIEVE + RERANK
# ============================================================

def retrieve_context(
    question,
    retriever,
    reranker,
):

    retrieval_query = create_retrieval_query(
        question
    )

    candidate_documents = retriever.invoke(
        retrieval_query
    )

    candidate_documents = remove_duplicates(
        candidate_documents
    )

    reranked_documents = rerank_documents(
        question=question,
        documents=candidate_documents,
        reranker=reranker,
    )

    contexts = [
        document.page_content.strip()
        for document in reranked_documents
        if document.page_content.strip()
    ]

    return contexts


# ============================================================
# OUTPUT FILE
# ============================================================

def initialize_output():

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "question_number",
                "question",
                "contextual_recall",
                "passed",
                "retrieved_chunks",
                "error"
            ]
        )


def save_result(
    question_number,
    question,
    score,
    passed,
    retrieved_chunks,
    error="",
):

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                question_number,
                question,
                score,
                passed,
                retrieved_chunks,
                error
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EnterpriseMind AI - Contextual Recall Evaluation")
    print(f"Questions {START_QUESTION} to {END_QUESTION}")
    print("=" * 70)

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not gemini_api_key:

        raise ValueError(
            "GEMINI_API_KEY was not found in .env"
        )

    print(
        "Gemini API key loaded successfully."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset()

    total_questions = len(dataset)

    if START_QUESTION < 1:
        raise ValueError(
            "START_QUESTION must be >= 1"
        )

    if END_QUESTION > total_questions:
        raise ValueError(
            f"END_QUESTION cannot exceed "
            f"{total_questions}"
        )

    if START_QUESTION > END_QUESTION:
        raise ValueError(
            "START_QUESTION cannot be greater "
            "than END_QUESTION"
        )

    selected_dataset = dataset.iloc[
        START_QUESTION - 1:END_QUESTION
    ]

    print(
        f"\nTotal dataset questions: "
        f"{total_questions}"
    )

    print(
        f"Questions selected: "
        f"{START_QUESTION}-{END_QUESTION}"
    )

    print(
        f"Number of questions this run: "
        f"{len(selected_dataset)}"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    print(
        "Connecting to ChromaDB..."
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": INITIAL_K
        },
    )

    # --------------------------------------------------------
    # CrossEncoder
    # --------------------------------------------------------

    print(
        "Loading CrossEncoder reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    # --------------------------------------------------------
    # Gemini Judge
    # --------------------------------------------------------

    print(
        f"Loading Gemini judge: "
        f"{JUDGE_MODEL}..."
    )

    judge_model = GeminiModel(
        model=JUDGE_MODEL,
        api_key=gemini_api_key,
        temperature=0,
    )

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    metric = ContextualRecallMetric(
        threshold=METRIC_THRESHOLD,
        model=judge_model,
        include_reason=False,
        async_mode=False,
    )

    initialize_output()

    scores = []
    passed_count = 0
    evaluation_errors = 0

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for index, row in selected_dataset.iterrows():

        question_number = index + 1

        question = row["question"]
        ground_truth = row["ground_truth"]

        print("\n" + "=" * 70)

        print(
            f"Question "
            f"{question_number}/{total_questions}"
        )

        print("=" * 70)

        print(question)

        try:

            # ------------------------------------------------
            # Retrieve + rerank
            # ------------------------------------------------

            contexts = retrieve_context(
                question=question,
                retriever=retriever,
                reranker=reranker,
            )

            print(
                f"Retrieved candidates: "
                f"{INITIAL_K}"
            )

            print(
                f"Reranked context chunks: "
                f"{len(contexts)}"
            )

            if not contexts:

                print(
                    "No context retrieved."
                )

                save_result(
                    question_number=question_number,
                    question=question,
                    score="",
                    passed=False,
                    retrieved_chunks=0,
                    error="No context retrieved",
                )

                evaluation_errors += 1

                continue

            # ------------------------------------------------
            # DeepEval Test Case
            # ------------------------------------------------

            test_case = LLMTestCase(
                input=question,
                actual_output=ground_truth,
                expected_output=ground_truth,
                retrieval_context=contexts,
            )

            # ------------------------------------------------
            # Measure
            # ------------------------------------------------

            metric.measure(
                test_case
            )

            score = float(
                metric.score
            )

            passed = (
                score >= METRIC_THRESHOLD
            )

            scores.append(
                score
            )

            if passed:
                passed_count += 1

            print(
                f"\nContextual Recall: "
                f"{score:.4f}"
            )

            print(
                f"Result: "
                f"{'PASS' if passed else 'FAIL'}"
            )

            save_result(
                question_number=question_number,
                question=question,
                score=f"{score:.4f}",
                passed=passed,
                retrieved_chunks=len(contexts),
                error="",
            )

            running_average = (
                sum(scores) / len(scores)
            )

            print(
                f"Running Average: "
                f"{running_average:.4f}"
            )

            print(
                f"Completed this run: "
                f"{len(scores)}/"
                f"{len(selected_dataset)}"
            )

        except KeyboardInterrupt:

            print(
                "\n\nEvaluation stopped by user."
            )

            print(
                "Completed results are saved."
            )

            break

        except Exception as error:

            evaluation_errors += 1

            print(
                "\nEvaluation failed:"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            save_result(
                question_number=question_number,
                question=question,
                score="",
                passed=False,
                retrieved_chunks=0,
                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CONTEXTUAL RECALL RESULTS "
        "FOR QUESTIONS 7-13"
    )
    print("=" * 70)

    if scores:

        average_score = (
            sum(scores)
            / len(scores)
        )

        print(
            f"Successfully evaluated: "
            f"{len(scores)}/"
            f"{len(selected_dataset)}"
        )

        print(
            f"Average Contextual Recall: "
            f"{average_score:.4f}"
        )

        print(
            f"Passed: "
            f"{passed_count}/{len(scores)}"
        )

        print(
            f"Pass Rate: "
            f"{(passed_count / len(scores)) * 100:.2f}%"
        )

    else:

        print(
            "No questions were successfully evaluated."
        )

    print(
        f"Evaluation errors: "
        f"{evaluation_errors}"
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()