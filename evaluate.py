import os
from pathlib import Path
from typing import Any

import pandas as pd


# ----------------------------------------------------------------------
# DeepEval timeout settings
# These must be set before importing DeepEval.
# ----------------------------------------------------------------------

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1800"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "2100"
os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "1"


from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

try:
    from deepeval.evaluate.configs import AsyncConfig
except ImportError:
    AsyncConfig = None

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET_PATH = "evaluation_dataset.csv"
VECTOR_DB_PATH = "vector_db"

RESULTS_FILE = "deepeval_results.txt"
FAILED_BATCHES_FILE = "failed_batches.txt"

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"
JUDGE_MODEL = "llama3.2"

OLLAMA_BASE_URL = "http://localhost:11434"

# Number of questions evaluated together
BATCH_SIZE = 3

# Similarity retrieval settings
TOP_K = 10

# A metric score equal to or greater than this value passes
METRIC_THRESHOLD = 0.5


# ======================================================================
# PROMPT
# ======================================================================

PROMPT_TEMPLATE = """
You are an Enterprise Financial Document Question-Answering Assistant.

Your task is to answer the user's question using ONLY the information
provided in the context.

Instructions:

1. Read all retrieved context carefully before answering.
2. Combine information from multiple context chunks whenever necessary.
3. Answer the user's exact question directly and completely.
4. Include all relevant facts, numbers, dates, percentages, names, and
   explanations that are explicitly present in the context.
5. The wording in the question may differ from the wording in the context.
   Use information that has the same meaning.
6. If the retrieved context completely answers the question,
   provide the complete answer and stop.
7. Only state:
   "The provided documents do not specify the remaining information."
   if an essential part of the question is genuinely missing from every
   retrieved chunk.
8. Do not infer, speculate, or assume anything that is not explicitly
   stated in the context.
9. Do not use outside knowledge.
10. Keep the answer concise, factual, focused, and well-organized.
11. When answering financial questions, preserve the values and units
    exactly as they appear in the retrieved context. If you perform
    calculations, ensure the units remain consistent with the source data.
12. Use the fallback response only when the answer is completely absent
    from every retrieved chunk.

Fallback response:

I could not find this information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# ======================================================================
# FILE AND DATASET FUNCTIONS
# ======================================================================

def validate_files() -> None:
    """
    Check whether the evaluation dataset and vector database exist.
    """

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not Path(VECTOR_DB_PATH).exists():
        raise FileNotFoundError(
            f"Vector database not found: {VECTOR_DB_PATH}"
        )


def load_dataset() -> pd.DataFrame:
    """
    Load and validate the evaluation CSV file.
    """

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {
        "question",
        "ground_truth"
    }

    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {sorted(missing_columns)}"
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

    dataset = dataset[
        (dataset["question"] != "")
        & (dataset["ground_truth"] != "")
    ].reset_index(drop=True)

    if dataset.empty:
        raise ValueError(
            "The evaluation dataset contains no valid rows."
        )

    return dataset


# ======================================================================
# RETRIEVAL FUNCTIONS
# ======================================================================

def create_retrieval_query(question: str) -> str:
    """
    Expand the question for improved semantic retrieval.
    """

    return f"""
Retrieve the document passages required to answer the following question.

User question:
{question}

Retrieval instructions:

- Retrieve passages that directly answer the question.
- Include supporting facts, values, dates, percentages, and explanations.
- If the question concerns financial performance, prioritize consolidated
  company-level results before individual business segments.
- Retrieve nearby supporting passages when the complete answer may span
  multiple chunks.
"""


def remove_duplicate_documents(documents: list) -> list:
    """
    Remove duplicate and empty chunks.
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


# ======================================================================
# DEEPEVAL METRICS
# ======================================================================
def create_metrics(judge_model: OllamaModel) -> list:
    return [
        ContextualRecallMetric(
            threshold=METRIC_THRESHOLD,
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
    ]


# ======================================================================
# DEEPEVAL RESULT FUNCTIONS
# ======================================================================

def extract_test_results(evaluation_result: Any) -> list:
    """
    Extract TestResult objects from DeepEval's return value.

    This supports multiple DeepEval return formats.
    """

    if evaluation_result is None:
        return []

    if hasattr(evaluation_result, "test_results"):
        return evaluation_result.test_results or []

    if isinstance(evaluation_result, list):
        return evaluation_result

    return []


def collect_metric_scores(
    test_results: list,
    metric_scores: dict[str, list[float]],
    metric_passes: dict[str, int],
) -> None:
    """
    Collect metric scores and pass counts.
    """

    for test_result in test_results:
        metrics_data = getattr(
            test_result,
            "metrics_data",
            None
        )

        if not metrics_data:
            continue

        for metric_data in metrics_data:
            metric_name = getattr(
                metric_data,
                "name",
                None
            )

            score = getattr(
                metric_data,
                "score",
                None
            )

            success = getattr(
                metric_data,
                "success",
                False
            )

            if metric_name not in metric_scores:
                continue

            if score is not None:
                metric_scores[metric_name].append(
                    float(score)
                )

            if success:
                metric_passes[metric_name] += 1


def calculate_average(scores: list[float]) -> float:
    """
    Calculate a safe average score.
    """

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ======================================================================
# ANSWER GENERATION
# ======================================================================

def generate_answer(
    question: str,
    retriever,
    llm: ChatOllama,
) -> tuple[str, list[str]]:
    """
    Retrieve relevant chunks and generate an answer.
    """

    retrieval_query = create_retrieval_query(question)

    documents = retriever.invoke(retrieval_query)

    documents = remove_duplicate_documents(documents)

    retrieved_context = [
        document.page_content.strip()
        for document in documents
        if document.page_content
        and document.page_content.strip()
    ]

    if not retrieved_context:
        return (
            "I could not find this information in the provided documents.",
            [],
        )

    context_parts = []

    for index, document in enumerate(documents, start=1):
        content = document.page_content.strip()

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page_number = document.metadata.get(
            "page_number",
            document.metadata.get("page", "Unknown page")
        )

        context_parts.append(
            f"[Context Chunk {index}]\n"
            f"Source: {source}\n"
            f"Page: {page_number}\n"
            f"{content}"
        )

    combined_context = "\n\n---\n\n".join(context_parts)

    formatted_prompt = prompt.invoke(
        {
            "context": combined_context,
            "question": question,
        }
    )

    response = llm.invoke(formatted_prompt)

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

    return answer, retrieved_context


# ======================================================================
# RESULT FILE FUNCTIONS
# ======================================================================

def write_header(
    total_questions: int,
    total_batches: int,
) -> None:
    """
    Create the initial output files.
    """

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("Enterprise RAG DeepEval Results\n")
        file.write("=" * 70 + "\n")
        file.write(
            f"Total questions: {total_questions}\n"
        )
        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )
        file.write(
            f"Total batches: {total_batches}\n"
        )
        file.write(
            "Retrieval type: Similarity\n"
        )
        file.write(
            f"Retrieved chunks: {TOP_K}\n"
        )
        file.write(
            f"Metric threshold: {METRIC_THRESHOLD}\n"
        )
        file.write("=" * 70 + "\n\n")

    with open(
        FAILED_BATCHES_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("Failed DeepEval Batches\n")
        file.write("=" * 70 + "\n\n")


def append_batch_result(
    batch_number: int,
    question_numbers: list[int],
    evaluation_result: Any,
) -> None:
    """
    Save a batch's raw DeepEval output.
    """

    with open(
        RESULTS_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(f"Batch {batch_number}\n")
        file.write(
            f"Questions evaluated: {question_numbers}\n"
        )
        file.write("-" * 70 + "\n")
        file.write(str(evaluation_result))
        file.write("\n\n")


def append_failed_batch(
    batch_number: int,
    question_numbers: list[int],
    error: Exception,
) -> None:
    """
    Save information about a failed batch.
    """

    with open(
        FAILED_BATCHES_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(f"Batch {batch_number}\n")
        file.write(f"Questions: {question_numbers}\n")
        file.write(
            f"Error type: {type(error).__name__}\n"
        )
        file.write(f"Error: {error}\n")
        file.write("-" * 70 + "\n\n")


def save_final_summary(
    metric_scores: dict[str, list[float]],
    metric_passes: dict[str, int],
    total_questions: int,
    test_cases_generated: int,
    generation_failures: int,
    completed_batches: int,
    failed_batches: int,
    successful_test_cases: int,
    failed_test_cases: int,
) -> None:
    """
    Print and save the final evaluation summary.
    """

    summary_lines = []

    summary_lines.append("=" * 70)
    summary_lines.append("FINAL EVALUATION SUMMARY")
    summary_lines.append("=" * 70)

    summary_lines.append(
        f"Total questions: {total_questions}"
    )

    summary_lines.append(
        f"Test cases generated: {test_cases_generated}"
    )

    summary_lines.append(
        f"Generation failures: {generation_failures}"
    )

    summary_lines.append(
        f"Completed batches: {completed_batches}"
    )

    summary_lines.append(
        f"Failed batches: {failed_batches}"
    )

    summary_lines.append(
        f"Successful test cases: {successful_test_cases}"
    )

    summary_lines.append(
        f"Failed test cases: {failed_test_cases}"
    )

    summary_lines.append("-" * 70)

    for metric_name, scores in metric_scores.items():
        average_score = calculate_average(scores)
        pass_count = metric_passes[metric_name]
        evaluated_count = len(scores)

        if evaluated_count > 0:
            pass_percentage = (
                pass_count / evaluated_count
            ) * 100
        else:
            pass_percentage = 0.0

        summary_lines.append(
            f"{metric_name}:"
        )

        summary_lines.append(
            f"  Average score: {average_score:.4f}"
        )

        summary_lines.append(
            f"  Passed: {pass_count}/{evaluated_count} "
            f"({pass_percentage:.2f}%)"
        )

    summary_lines.append("=" * 70)

    summary_lines.append(
        f"Results saved to: {RESULTS_FILE}"
    )

    summary_lines.append(
        f"Failure details saved to: {FAILED_BATCHES_FILE}"
    )

    summary_lines.append("=" * 70)

    summary = "\n".join(summary_lines)

    print("\n")
    print(summary)

    with open(
        RESULTS_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write("\n")
        file.write(summary)
        file.write("\n")


# ======================================================================
# MAIN EVALUATION
# ======================================================================
def main() -> None:
    """
    Run the complete RAG evaluation.
    """

    validate_files()

    dataset = load_dataset()

    total_questions = len(dataset)

    total_batches = (
        total_questions + BATCH_SIZE - 1
    ) // BATCH_SIZE

    write_header(
        total_questions=total_questions,
        total_batches=total_batches,
    )
    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    print("Loading embedding model...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # ------------------------------------------------------------------
    # Chroma vector database
    # ------------------------------------------------------------------

    print("Connecting to ChromaDB...")

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

    # ------------------------------------------------------------------
    # Similarity retriever
    # ------------------------------------------------------------------

    print("Creating similarity retriever...")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K,
        },
    )

    # ------------------------------------------------------------------
    # Generator model
    # ------------------------------------------------------------------

    print("Loading generator model...")

    generator_llm = ChatOllama(
        model=GENERATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    # ------------------------------------------------------------------
    # DeepEval judge model
    # ------------------------------------------------------------------

    print("Loading DeepEval judge model...")

    judge_model = OllamaModel(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    # ------------------------------------------------------------------
    # Metric score storage
    # ------------------------------------------------------------------

    metric_scores = {
        "Faithfulness": [],
        "Answer Relevancy": [],
        "Contextual Precision": [],
        "Contextual Recall": [],
    }

    metric_passes = {
        "Faithfulness": 0,
        "Answer Relevancy": 0,
        "Contextual Precision": 0,
        "Contextual Recall": 0,
    }

    completed_batches = 0
    failed_batches = 0
    generation_failures = 0
    test_cases_generated = 0
    successful_test_cases = 0
    failed_test_cases = 0

    # ------------------------------------------------------------------
    # Evaluate dataset batch by batch
    # ------------------------------------------------------------------

    for batch_start in range(
        0,
        total_questions,
        BATCH_SIZE,
    ):
        batch_end = min(
            batch_start + BATCH_SIZE,
            total_questions,
        )

        batch_number = (
            batch_start // BATCH_SIZE
        ) + 1

        batch_data = dataset.iloc[
            batch_start:batch_end
        ]

        question_numbers = list(
            range(
                batch_start + 1,
                batch_end + 1
            )
        )

        print("\n" + "=" * 70)

        print(
            f"Processing Batch "
            f"{batch_number}/{total_batches}"
        )

        print(
            f"Questions: {question_numbers}"
        )

        print("=" * 70)

        batch_test_cases = []

        # --------------------------------------------------------------
        # Generate answers
        # --------------------------------------------------------------

        for row_index, row in batch_data.iterrows():
            question_number = row_index + 1

            question = row["question"]
            ground_truth = row["ground_truth"]

            print(
                f"\nGenerating answer for "
                f"question {question_number}..."
            )

            try:
                generated_answer, retrieved_context = (
                    generate_answer(
                        question=question,
                        retriever=retriever,
                        llm=generator_llm,
                    )
                )

                test_case = LLMTestCase(
                    input=question,
                    actual_output=generated_answer,
                    expected_output=ground_truth,
                    retrieval_context=retrieved_context,
                )

                batch_test_cases.append(test_case)

                test_cases_generated += 1

                print(f"Question: {question}")
                print(f"Answer: {generated_answer}")
                print(
                    f"Retrieved chunks: "
                    f"{len(retrieved_context)}"
                )

            except Exception as error:
                generation_failures += 1

                print(
                    f"Generation failed for question "
                    f"{question_number}: {error}"
                )

                with open(
                    FAILED_BATCHES_FILE,
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(
                        f"Generation failure — "
                        f"Question {question_number}\n"
                    )

                    file.write(
                        f"Question: {question}\n"
                    )

                    file.write(
                        f"Error type: "
                        f"{type(error).__name__}\n"
                    )

                    file.write(
                        f"Error: {error}\n"
                    )

                    file.write(
                        "-" * 70 + "\n\n"
                    )

        if not batch_test_cases:
            failed_batches += 1

            print(
                f"Batch {batch_number} skipped because "
                "no test cases were generated."
            )

            continue

        # --------------------------------------------------------------
        # Run DeepEval
        # --------------------------------------------------------------

        try:
            metrics = create_metrics(judge_model)

            evaluation_arguments = {
                "test_cases": batch_test_cases,
                "metrics": metrics,
            }

            if AsyncConfig is not None:
                evaluation_arguments["async_config"] = (
                    AsyncConfig(
                        run_async=False,
                        max_concurrent=1,
                    )
                )

            evaluation_result = evaluate(
                **evaluation_arguments
            )

            batch_results = extract_test_results(
                evaluation_result
            )

            collect_metric_scores(
                test_results=batch_results,
                metric_scores=metric_scores,
                metric_passes=metric_passes,
            )

            for test_result in batch_results:
                if getattr(
                    test_result,
                    "success",
                    False,
                ):
                    successful_test_cases += 1
                else:
                    failed_test_cases += 1

            append_batch_result(
                batch_number=batch_number,
                question_numbers=question_numbers,
                evaluation_result=evaluation_result,
            )

            completed_batches += 1

            print(
                f"\nBatch {batch_number} "
                "completed successfully."
            )

        except Exception as error:
            failed_batches += 1

            print(
                f"\nBatch {batch_number} failed."
            )

            print(
                f"Error type: {type(error).__name__}"
            )

            print(f"Error: {error}")

            append_failed_batch(
                batch_number=batch_number,
                question_numbers=question_numbers,
                error=error,
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DEEPEVAL PROCESS FINISHED")
    print("=" * 70)

    save_final_summary(
        metric_scores=metric_scores,
        metric_passes=metric_passes,
        total_questions=total_questions,
        test_cases_generated=test_cases_generated,
        generation_failures=generation_failures,
        completed_batches=completed_batches,
        failed_batches=failed_batches,
        successful_test_cases=successful_test_cases,
        failed_test_cases=failed_test_cases,
    )


if __name__ == "__main__":
    main()