import json
import re
from typing import Any

from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

ROUTER_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# ROUTES
# ============================================================

ROUTE_QA = "qa"
ROUTE_SUMMARIZE = "summarize"

VALID_ROUTES = {
    ROUTE_QA,
    ROUTE_SUMMARIZE,
}


# ============================================================
# ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
You are the Router Agent for an Enterprise RAG Assistant.

Your ONLY task is to classify the user's request into exactly one route.

Available routes:

1. "qa"
2. "summarize"


============================================================
ROUTE: qa
============================================================

Choose "qa" when the user wants a specific question answered.

This includes requests asking for:

- a specific fact
- a number or percentage
- a date
- a person
- a reason or cause
- an explanation
- a comparison
- which item performed better or worse
- whether something increased or decreased
- why something increased or decreased
- factors responsible for a change
- the relationship between specific facts
- a specific conclusion based on document evidence

Examples of QA-style intent:

- What was the company's operating income?
- Why did revenue decrease?
- Which segment performed better?
- Did operating margin increase or decrease?
- What caused the decline?
- How did one segment compare with another?
- Was performance better or worse, and why?
- What factors affected profitability?


IMPORTANT:

A question can mention a broad topic such as company performance,
cloud performance, or financial performance and STILL be "qa"
if the user is asking a specific question about that topic.

For example:

"Was the company's cloud performance better or worse, and why?"

This is "qa" because the user wants a specific judgment/comparison
and an explanation.


============================================================
ROUTE: summarize
============================================================

Choose "summarize" when the user wants information condensed into
a broad overview rather than one specific question answered.

This includes requests asking for:

- a summary
- an overview
- a recap
- key points
- highlights
- main takeaways
- major developments
- important findings
- a brief breakdown
- a condensed explanation
- the overall picture
- what they should know about a broad topic

Examples of summarization-style intent:

- Summarize the company's financial performance.
- Give me the key points from the report.
- Walk me through the major developments during the year.
- What are the main takeaways from this section?
- Give me a quick recap of the company's performance.
- Tell me the most important developments in the report.
- I do not need every detail. Give me the overall picture.


============================================================
CRITICAL DECISION RULES
============================================================

Rule 1:

Specific question beats broad topic.

If the user mentions a broad topic but asks a specific question
about it, choose "qa".

Example:

"Was cloud performance better or worse in 2025, and why?"

Route:
qa


Rule 2:

Broad condensation requests use "summarize".

Example:

"Walk me through the major changes in the cloud business."

Route:
summarize


Rule 3:

Questions containing "why", "what caused", "what drove",
"better or worse", "increase or decrease", "which performed better",
or similar specific analytical requests normally use "qa".


Rule 4:

Words such as "performance", "financial", "report", or "business"
DO NOT automatically mean summarization.


Rule 5:

Do not route based only on keywords.

Understand the meaning of the entire request.


Rule 6:

Handle negation correctly.

Example:

"Do not summarize the report. Just tell me why revenue decreased."

Route:
qa

The presence of the word "summarize" does NOT make this a
summarization request because the user explicitly says not to summarize.


Rule 7:

If a request asks for broad highlights AND then narrows the request
to one specific question, prefer "qa".

Example:

"Give me the main findings, but focus specifically on what caused
the decline in server products revenue."

Route:
qa


Rule 8:

If the user asks for several important developments, key findings,
major changes, or an overall picture without one specific factual
question, choose "summarize".


Rule 9:

If the intent remains genuinely ambiguous after applying all rules,
default to "qa".


============================================================
OUTPUT RULES
============================================================

Do NOT answer the user's question.

Do NOT summarize any documents.

Do NOT retrieve information.

Do NOT provide additional text.

Return ONLY valid JSON using this structure:

{{
    "route": "qa",
    "reason": "The request asks for a specific question to be answered."
}}

User request:

{question}
"""


# ============================================================
# LOAD ROUTER MODEL
# ============================================================

_router_llm = ChatOllama(
    model=ROUTER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text: str) -> dict[str, Any]:

    if not text:
        return {
            "route": ROUTE_QA,
            "reason": (
                "Empty Router Agent response. "
                "Defaulting to QA."
            ),
        }

    text = text.strip()

    # Remove possible markdown fences
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
    # DIRECT JSON PARSING
    # --------------------------------------------------------

    try:

        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # EXTRACT JSON OBJECT
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = text[start:end + 1]

        try:

            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "route": ROUTE_QA,
        "reason": (
            "Router response could not be parsed. "
            "Defaulting to QA."
        ),
    }


# ============================================================
# ROUTER AGENT
# ============================================================

def route_request(question: str) -> dict[str, Any]:

    """
    Classify a user request into:

        qa
        summarize

    The Router Agent does not answer the question.
    It only decides which specialized workflow should handle it.
    """

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not question or not question.strip():

        return {
            "route": ROUTE_QA,
            "reason": (
                "Empty request. "
                "Defaulting to QA."
            ),
        }

    question = question.strip()

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = ROUTER_PROMPT.format(
        question=question
    )

    # --------------------------------------------------------
    # CALL ROUTER LLM
    # --------------------------------------------------------

    try:

        response = _router_llm.invoke(prompt)

        raw_output = getattr(
            response,
            "content",
            str(response)
        )

    except Exception as error:

        return {
            "route": ROUTE_QA,
            "reason": (
                f"Router model error: "
                f"{type(error).__name__}. "
                f"Defaulting to QA."
            ),
        }

    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    result = _extract_json(raw_output)

    route = str(
        result.get(
            "route",
            ROUTE_QA
        )
    ).strip().lower()

    reason = str(
        result.get(
            "reason",
            ""
        )
    ).strip()

    # --------------------------------------------------------
    # NORMALIZE ROUTE ALIASES
    # --------------------------------------------------------

    route_aliases = {

        # QA aliases
        "q&a": ROUTE_QA,
        "qna": ROUTE_QA,
        "question_answer": ROUTE_QA,
        "question-answer": ROUTE_QA,
        "question answering": ROUTE_QA,
        "normal_qna": ROUTE_QA,
        "normal_qa": ROUTE_QA,

        # Summary aliases
        "summary": ROUTE_SUMMARIZE,
        "summarization": ROUTE_SUMMARIZE,
        "summarisation": ROUTE_SUMMARIZE,
        "summarise": ROUTE_SUMMARIZE,
        "summarizer": ROUTE_SUMMARIZE,
    }

    route = route_aliases.get(
        route,
        route
    )

    # --------------------------------------------------------
    # VALIDATE ROUTE
    # --------------------------------------------------------

    if route not in VALID_ROUTES:

        route = ROUTE_QA

        reason = (
            "Invalid route returned by Router Agent. "
            "Defaulting to QA."
        )

    # --------------------------------------------------------
    # ENSURE REASON EXISTS
    # --------------------------------------------------------

    if not reason:

        if route == ROUTE_SUMMARIZE:

            reason = (
                "The request asks for a broad summary "
                "or condensed overview."
            )

        else:

            reason = (
                "The request asks for a specific "
                "question to be answered."
            )

    # --------------------------------------------------------
    # RETURN DECISION
    # --------------------------------------------------------

    return {
        "route": route,
        "reason": reason,
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ENTERPRISE RAG ROUTER AGENT")
    print("=" * 70)

    while True:

        question = input(
            "\nEnter request "
            "(type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":

            print(
                "Exiting Router Agent."
            )

            break

        if not question:

            print(
                "Please enter a valid request."
            )

            continue

        result = route_request(
            question
        )

        print(
            "\nRoute:",
            result["route"]
        )

        print(
            "Reason:",
            result["reason"]
        )