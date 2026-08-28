import json
import re
from typing import Any

from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

SUMMARIZER_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

MAX_SUMMARY_DOCUMENTS = 4


# ============================================================
# FALLBACK
# ============================================================

FALLBACK_SUMMARY = (
    "I could not find enough information "
    "in the provided documents to summarize this topic."
)


# ============================================================
# MODELS
# ============================================================

# Used for JSON-based decisions:
# chunk selection + evidence extraction
_decision_llm = ChatOllama(
    model=SUMMARIZER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)


# Used for final natural-language summary
_summary_llm = ChatOllama(
    model=SUMMARIZER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


# ============================================================
# JSON PARSER
# ============================================================

def _extract_json(
    text: str
) -> dict[str, Any]:

    if not text:
        return {}

    text = text.strip()

    # Remove markdown fences if present
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
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
    # TRY EXTRACTING JSON OBJECT
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


    return {}


# ============================================================
# SUMMARY SCOPE FILTER
# ============================================================

def filter_summary_documents(
    question: str,
    documents: list,
) -> list:

    """
    Compare all reranked chunks together and select only
    the chunks most relevant to the requested summary topic.
    """

    if not question or not question.strip():
        return []

    if not documents:
        return []

    print(
        "[Summarizer] Comparing reranked chunks "
        "for summary relevance..."
    )


    # --------------------------------------------------------
    # BUILD NUMBERED CHUNK LIST
    # --------------------------------------------------------

    chunk_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        content = document.page_content.strip()

        if not content:
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

        chunk_parts.append(
            f"[Chunk {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{content}"
        )


    chunks_text = "\n\n---\n\n".join(
        chunk_parts
    )


    # --------------------------------------------------------
    # COMPARATIVE SELECTION PROMPT
    # --------------------------------------------------------

    selection_prompt = f"""
You are selecting document evidence for an Enterprise RAG Summarizer.

User summary request:

{question}

Reranked document chunks:

{chunks_text}

Your task is to compare ALL chunks with one another and identify
the smallest set of chunks most directly useful for summarizing
the EXACT topic requested by the user.

Rules:

1. Focus strictly on the exact topic requested by the user.

2. Do not select a chunk merely because it mentions the same company,
   fiscal year, revenue, or broad business category.

3. Prefer chunks containing direct:
   - facts
   - changes
   - percentages
   - monetary values
   - causes
   - drivers
   - outcomes

   related to the requested topic.

4. Reject chunks whose main subject is a different business segment,
   product, or topic.

5. A chunk may contain mixed information.
   It may still be selected if it contains an important passage
   directly relevant to the requested topic.

6. Select approximately 2 to 4 chunks when possible.

7. If only one chunk is directly relevant, selecting one is valid.

8. Never invent chunk IDs.

9. Return IDs only from the chunks supplied above.

For example, when the request is specifically about Intelligent Cloud:

Relevant information may include:
- Intelligent Cloud
- Server products and cloud services
- Azure and other cloud services
- directly related margin or operating information

Information mainly about unrelated areas such as:
- LinkedIn
- Gaming
- Microsoft 365
- Dynamics
- advertising

should not be selected unless it directly explains the requested
Intelligent Cloud topic.

Return ONLY valid JSON:

{{
    "selected_chunk_ids": [2, 4, 6],
    "reason": "These chunks most directly support the requested summary."
}}
"""


    # --------------------------------------------------------
    # CALL MODEL
    # --------------------------------------------------------

    try:

        response = _decision_llm.invoke(
            selection_prompt
        )

        raw_output = getattr(
            response,
            "content",
            str(response)
        )

        result = _extract_json(
            raw_output
        )

    except Exception as error:

        print(
            "[Summarizer] Comparative filtering error:",
            type(error).__name__
        )

        return documents[:2]


    # --------------------------------------------------------
    # EXTRACT IDS
    # --------------------------------------------------------

    selected_ids = result.get(
        "selected_chunk_ids",
        []
    )

    if not isinstance(
        selected_ids,
        list
    ):

        selected_ids = []


    clean_ids = []

    for chunk_id in selected_ids:

        try:

            chunk_id = int(
                chunk_id
            )

        except (
            TypeError,
            ValueError,
        ):

            continue


        if (
            1 <= chunk_id <= len(documents)
            and chunk_id not in clean_ids
        ):

            clean_ids.append(
                chunk_id
            )


    clean_ids = clean_ids[
        :MAX_SUMMARY_DOCUMENTS
    ]


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not clean_ids:

        print(
            "[Summarizer] No valid chunk IDs returned. "
            "Using top 2 reranked chunks."
        )

        return documents[:2]


    # --------------------------------------------------------
    # BUILD FILTERED DOCUMENT LIST
    # --------------------------------------------------------

    filtered_documents = [
        documents[
            chunk_id - 1
        ]
        for chunk_id in clean_ids
    ]


    print(
        "[Summarizer] Selected chunks:",
        clean_ids
    )

    print(
        f"[Summarizer] Kept "
        f"{len(filtered_documents)} of "
        f"{len(documents)} reranked chunks."
    )


    return filtered_documents


# ============================================================
# RELEVANT EVIDENCE EXTRACTION
# ============================================================

def extract_relevant_evidence(
    question: str,
    context: str,
) -> str:

    """
    Extract only the facts/passages from the selected chunks
    that directly relate to the requested summary topic.

    This solves the problem where a selected chunk contains
    several different business segments.
    """

    if not question or not question.strip():
        return ""

    if not context or not context.strip():
        return ""


    print(
        "[Summarizer] Extracting topic-specific evidence..."
    )


    extraction_prompt = f"""
You are an Enterprise RAG Evidence Extraction component.

The user wants a summary of a specific topic.

User request:

{question}

Selected document evidence:

{context}

Your job is NOT to summarize yet.

Your job is to extract ONLY facts that directly relate to the exact
topic requested by the user.

============================================================
RULES
============================================================

1. Read the user's request carefully and determine the exact subject.

2. Keep only facts directly related to that subject.

3. Remove facts belonging mainly to unrelated:
   - business segments
   - products
   - departments
   - topics

4. A selected document passage may contain multiple subjects.
   Extract only the sentences or facts relevant to the user's request.

5. Preserve exact:
   - category names
   - numbers
   - percentages
   - monetary values
   - dates
   - increase/decrease directions
   - cause-and-effect relationships

6. Do not invent information.

7. Do not use outside knowledge.

8. Do not reinterpret financial categories.

9. Do not add explanations that are absent from the source.

10. Avoid duplicates.

11. Only return information appearing in the supplied evidence.

For example:

If the user asks specifically about Intelligent Cloud and the evidence
contains:

- Microsoft 365 revenue growth
- LinkedIn revenue growth
- Azure revenue growth
- Server products and cloud services revenue
- Dynamics revenue growth

then extract only the facts directly related to Intelligent Cloud,
such as Azure and Server products and cloud services, unless another
fact is explicitly necessary to explain the requested topic.

Return ONLY valid JSON using this structure:

{{
    "evidence": [
        "Relevant fact 1",
        "Relevant fact 2",
        "Relevant fact 3"
    ]
}}
"""


    # --------------------------------------------------------
    # CALL EXTRACTION MODEL
    # --------------------------------------------------------

    try:

        response = _decision_llm.invoke(
            extraction_prompt
        )

        raw_output = getattr(
            response,
            "content",
            str(response)
        )

        result = _extract_json(
            raw_output
        )

    except Exception as error:

        print(
            "[Summarizer] Evidence extraction error:",
            type(error).__name__
        )

        return context


    # --------------------------------------------------------
    # GET EXTRACTED EVIDENCE
    # --------------------------------------------------------

    evidence = result.get(
        "evidence",
        []
    )

    if not isinstance(
        evidence,
        list
    ):

        evidence = []


    clean_evidence = []

    seen = set()

    for item in evidence:

        item = str(
            item
        ).strip()

        if not item:
            continue

        normalized = item.lower()

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        clean_evidence.append(
            item
        )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not clean_evidence:

        print(
            "[Summarizer] No extracted evidence returned. "
            "Using filtered context."
        )

        return context


    print(
        f"[Summarizer] Extracted "
        f"{len(clean_evidence)} relevant fact(s)."
    )


    return "\n".join(
        f"- {fact}"
        for fact in clean_evidence
    )


# ============================================================
# FINAL SUMMARY PROMPT
# ============================================================

SUMMARIZER_PROMPT = """
You are an Enterprise Document Summarizer Agent.

Generate a focused and grounded summary of the topic requested
by the user.

User request:

{question}

Topic-specific evidence:

{context}

============================================================
RULES
============================================================

1. Stay strictly within the user's requested topic.

2. Use ONLY the topic-specific evidence supplied above.

3. Do not introduce information from other business areas.

4. Do not use outside knowledge.

5. Do not invent facts, numbers, causes, or conclusions.

6. Preserve exact:
   - financial category names
   - percentages
   - monetary values
   - dates
   - increase/decrease directions

7. Preserve important cause-and-effect relationships.

8. Do not combine categories with different meanings.

9. Remove repetition.

10. Each important fact should normally appear only once.

11. Prioritize:
    - important changes
    - important drivers
    - important figures
    - important outcomes

12. Do not create a summary and then repeat the same content under
    headings such as:
    - Key Points
    - Highlights
    - Takeaways

13. Do not mention:
    - evidence
    - context
    - retrieval
    - chunks
    - filtering
    - agents
    - verification

14. Adapt the length to the user's request.

For short / quick / concise / 30-second requests:
produce approximately 3 to 5 short points.

For normal summaries / overviews / major developments:
produce approximately 3 to 6 focused points when sufficient evidence
exists.

For detailed summaries:
provide a longer structured summary while remaining within scope.

15. If the supplied topic-specific evidence is genuinely insufficient,
respond exactly:

"I could not find enough information in the provided documents to summarize this topic."

Return ONLY the final summary.

Summary:
"""


# ============================================================
# SUMMARIZER AGENT
# ============================================================

def summarize_documents(
    question: str,
    context: str,
) -> dict[str, Any]:

    """
    Produce a focused summary.

    Flow:

    filtered chunks
        ↓
    relevant evidence extraction
        ↓
    final summary generation
    """

    if not question or not question.strip():

        return {
            "summary": FALLBACK_SUMMARY,
            "supported": False,
        }


    if not context or not context.strip():

        return {
            "summary": FALLBACK_SUMMARY,
            "supported": False,
        }


    # --------------------------------------------------------
    # EXTRACT ONLY RELEVANT FACTS
    # --------------------------------------------------------

    relevant_evidence = extract_relevant_evidence(
        question=question,
        context=context,
    )


    if not relevant_evidence.strip():

        return {
            "summary": FALLBACK_SUMMARY,
            "supported": False,
        }


    # --------------------------------------------------------
    # BUILD FINAL SUMMARY PROMPT
    # --------------------------------------------------------

    prompt = SUMMARIZER_PROMPT.format(
        question=question.strip(),
        context=relevant_evidence.strip(),
    )


    # --------------------------------------------------------
    # GENERATE FINAL SUMMARY
    # --------------------------------------------------------

    try:

        response = _summary_llm.invoke(
            prompt
        )

    except Exception as error:

        return {
            "summary": FALLBACK_SUMMARY,
            "supported": False,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }


    summary = getattr(
        response,
        "content",
        str(response)
    ).strip()


    if not summary:

        return {
            "summary": FALLBACK_SUMMARY,
            "supported": False,
        }


    return {
        "summary": summary,
        "supported": True,
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "ENTERPRISE RAG SUMMARIZER AGENT"
    )

    print(
        "=" * 70
    )

    print(
        "\nSummarizer Agent ready."
    )

    print(
        "Comparative scope filtering: ENABLED"
    )

    print(
        "Topic-specific evidence extraction: ENABLED"
    )