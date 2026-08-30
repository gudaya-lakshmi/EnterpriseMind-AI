import json
import re
from typing import Any

from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

CITATION_MODEL = "llama3.2"

OLLAMA_BASE_URL = "http://localhost:11434"

# Maximum number of citations returned
MAX_CITATIONS = 5


# ============================================================
# CITATION AGENT PROMPT
# ============================================================

CITATION_PROMPT = """
You are an Enterprise RAG Citation Agent.

Your task is to identify which retrieved document chunks directly
support the final verified answer.

You will receive:

1. The original user question.
2. The final verified answer.
3. A list of retrieved document chunks.

Your job is NOT to rewrite the answer.

Your job is ONLY to identify the evidence that directly supports
the answer.

Rules:

1. Select a chunk only if it directly supports at least one factual
   claim in the final answer.

2. Do not select a chunk merely because it discusses the same topic.

3. If several chunks support different parts of the answer,
   select all chunks necessary to support the answer.

4. If multiple chunks provide essentially the same evidence,
   prefer the strongest and most direct chunk.

5. Avoid redundant citations.

6. Do not invent source names, page numbers, or chunk IDs.

7. Only select chunk IDs that appear in the Retrieved Chunks section.

8. A citation must support the FINAL VERIFIED ANSWER,
   not merely the original question.

9. If no chunk directly supports the answer, return an empty list.

10. Do not use outside knowledge.

Return ONLY valid JSON.

Example:

{{
    "supported": true,
    "citations": [
        {{
            "chunk_id": 2,
            "reason": "Directly supports the claim about Azure revenue growth."
        }},
        {{
            "chunk_id": 5,
            "reason": "Supports the overall Intelligent Cloud revenue increase."
        }}
    ]
}}

If no supporting evidence exists:

{{
    "supported": false,
    "citations": []
}}

Question:
{question}

Final Verified Answer:
{answer}

Retrieved Chunks:
{chunks}
"""


# ============================================================
# LOAD CITATION MODEL
# ============================================================

_citation_llm = ChatOllama(
    model=CITATION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)


# ============================================================
# JSON PARSER
# ============================================================

def _extract_json(
    text: str
) -> dict[str, Any]:

    if not text:
        return {
            "supported": False,
            "citations": [],
        }

    text = text.strip()

    # Remove markdown fences if model adds them
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    # --------------------------------------------------------
    # DIRECT JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            text
        )

        if isinstance(
            result,
            dict
        ):
            return result

    except json.JSONDecodeError:
        pass


    # --------------------------------------------------------
    # EXTRACT JSON OBJECT
    # --------------------------------------------------------

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            result = json.loads(
                candidate
            )

            if isinstance(
                result,
                dict
            ):
                return result

        except json.JSONDecodeError:
            pass


    return {
        "supported": False,
        "citations": [],
    }


# ============================================================
# BUILD CHUNK TEXT
# ============================================================

def build_chunk_list(
    documents
) -> str:

    """
    Convert LangChain Documents into numbered chunks
    for the Citation Agent.
    """

    chunk_parts = []

    for index, document in enumerate(
        documents,
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

        content = document.page_content.strip()

        chunk_parts.append(
            f"[Chunk {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(
        chunk_parts
    )


# ============================================================
# REMOVE DUPLICATE CITATIONS
# ============================================================

def remove_duplicate_citations(
    citations
):

    """
    Remove duplicate source/page citations.

    If two chunks point to the same source and page,
    keep only the first one.
    """

    unique = []

    seen = set()

    for citation in citations:

        source = citation.get(
            "source"
        )

        page = citation.get(
            "page"
        )

        key = (
            source,
            page
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            citation
        )

    return unique
# ============================================================
# DETERMINISTIC CITATION FALLBACK
# ============================================================

def _extract_answer_numbers(
    text: str
) -> set[str]:

    numbers = set()

    # Percentages
    percentages = re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text
    )

    for value in percentages:
        numbers.add(
            value.replace(" ", "")
        )

    # Financial values such as 281,724
    financial_values = re.findall(
        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        text
    )

    for value in financial_values:
        numbers.add(
            value.replace(",", "")
        )

    return numbers


def _extract_chunk_numbers(
    text: str
) -> set[str]:

    numbers = set()

    # Percentages
    percentages = re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text
    )

    for value in percentages:
        numbers.add(
            value.replace(" ", "")
        )

    # Financial values such as 281,724
    financial_values = re.findall(
        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        text
    )

    for value in financial_values:
        numbers.add(
            value.replace(",", "")
        )

    return numbers


def deterministic_citation_fallback(
    answer: str,
    documents
) -> list[dict]:

    answer_numbers = _extract_answer_numbers(
        answer
    )

    if not answer_numbers:
        return []

    candidates = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        chunk_numbers = _extract_chunk_numbers(
            document.page_content
        )

        matching_numbers = (
            answer_numbers
            & chunk_numbers
        )

        if not matching_numbers:
            continue

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

        candidates.append(
            {
                "chunk_id": index,
                "source": source,
                "page": page,
                "reason": (
                    "Deterministic numeric evidence match."
                ),
                "content": document.page_content.strip(),
                "_match_count": len(matching_numbers),
            }
        )

    candidates.sort(
        key=lambda item: item["_match_count"],
        reverse=True
    )

    for candidate in candidates:
        candidate.pop(
            "_match_count",
            None
        )

    return candidates


# ============================================================
# CITATION AGENT
# ============================================================

def find_citations(
    question: str,
    answer: str,
    documents
) -> dict[str, Any]:

    """
    Identify which retrieved chunks directly support
    the final verified answer.
    """

    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    if not documents:

        return {
            "supported": False,
            "citations": [],
        }


    # --------------------------------------------------------
    # BUILD NUMBERED CHUNK LIST
    # --------------------------------------------------------

    chunk_text = build_chunk_list(
        documents
    )


    # --------------------------------------------------------
    # BUILD CITATION PROMPT
    # --------------------------------------------------------

    prompt = CITATION_PROMPT.format(
        question=question,
        answer=answer,
        chunks=chunk_text,
    )


    # --------------------------------------------------------
    # ASK CITATION AGENT
    # --------------------------------------------------------

    response = _citation_llm.invoke(
        prompt
    )

    raw_output = getattr(
        response,
        "content",
        str(response)
    )


    # --------------------------------------------------------
    # PARSE MODEL OUTPUT
    # --------------------------------------------------------

    result = _extract_json(
        raw_output
    )

    selected_citations = result.get(
        "citations",
        []
    )

    if not isinstance(
        selected_citations,
        list
    ):

        selected_citations = []


    # --------------------------------------------------------
    # VALIDATE CITATION IDS
    # --------------------------------------------------------

    final_citations = []

    for citation in selected_citations:

        if not isinstance(
            citation,
            dict
        ):
            continue

        chunk_id = citation.get(
            "chunk_id"
        )

        try:
            chunk_id = int(
                chunk_id
            )

        except (
            TypeError,
            ValueError,
        ):
            continue


        # Chunk IDs start at 1
        if (
            chunk_id < 1
            or chunk_id > len(documents)
        ):
            continue


        document = documents[
            chunk_id - 1
        ]

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

        reason = str(
            citation.get(
                "reason",
                ""
            )
        ).strip()


        final_citations.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "reason": reason,
                "content": document.page_content.strip(),
            }
        )


    # --------------------------------------------------------
    # REMOVE DUPLICATE PAGE REFERENCES
    # --------------------------------------------------------

    final_citations = remove_duplicate_citations(
        final_citations
    )


    # --------------------------------------------------------
    # DETERMINISTIC CITATION FALLBACK
    # --------------------------------------------------------

    if not final_citations:

        print(
            "[Citation Agent] LLM found no citation. "
            "Running deterministic fallback..."
        )

        final_citations = (
            deterministic_citation_fallback(
                answer,
                documents
            )
        )

        final_citations = (
            remove_duplicate_citations(
                final_citations
            )
        )

        if final_citations:

            print(
                f"[Citation Agent] Fallback found "
                f"{len(final_citations)} supporting citation(s)."
            )

        else:

            print(
                "[Citation Agent] Fallback found "
                "no supporting citation."
            )


        # --------------------------------------------------------
    # LIMIT NUMBER OF CITATIONS
    # --------------------------------------------------------

    final_citations = final_citations[
        :MAX_CITATIONS
    ]


    return {
        "supported": bool(
            final_citations
        ),
        "citations": final_citations,
    }


# ============================================================
# PRINT CITATIONS
# ============================================================

def print_citations(
    citation_result: dict[str, Any]
):

    citations = citation_result.get(
        "citations",
        []
    )

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
            f"{citation['source']} "
            f"— Page {citation['page']}"
        )


# ============================================================
# OPTIONAL CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Enterprise RAG Citation Agent"
    )

    print(
        "=" * 70
    )

    print(
        "\nThis agent is normally called "
        "from the RAG workflow."
    )

    print(
        "Use test_citation_agent.py "
        "to test citation selection."
    )