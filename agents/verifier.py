import ast
import json
import re
from typing import Any

from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

VERIFIER_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# SEMANTIC VERIFIER PROMPT
# ============================================================

SEMANTIC_PROMPT = """
You are an Enterprise RAG semantic support verifier.

Your task is to check whether the claims in the generated answer
are supported by the retrieved evidence.

Do NOT judge answer completeness here.

Rules:

1. Every factual claim must be supported by the retrieved evidence.

2. Preserve the exact scope of financial categories.

3. Treat these as different categories:
   - "Server products and cloud services"
   - "Server products"

4. Do not confuse revenue with:
   - gross margin
   - operating income
   - cost of revenue
   - another financial metric

5. Preserve cause-and-effect relationships exactly.

6. Paraphrasing is allowed when the meaning remains the same.

7. Do not invent errors.

8. Do not treat values belonging to different categories
   as contradictions.

9. Do not reject an answer merely because it is concise.

10. Before marking a statement unsupported, check whether the
    retrieved evidence expresses the same meaning in different wording.

11. If every claim is supported and there is no contradiction,
    return PASS.

Return ONLY valid JSON.

PASS example:

{{
  "verdict": "PASS",
  "issues": [],
  "feedback": "The answer is fully supported by the evidence."
}}

REVISE example:

{{
  "verdict": "REVISE",
  "issues": ["The answer contradicts the evidence."],
  "feedback": "Correct the unsupported claim."
}}

Do not use markdown.
Do not use code fences.
Do not write anything outside the JSON object.

Question:
{question}

Generated Answer:
{answer}

Retrieved Evidence:
{context}
"""


# ============================================================
# LOAD VERIFIER MODEL
# ============================================================

_verifier_llm = ChatOllama(
    model=VERIFIER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text: str) -> dict[str, Any]:

    if not text:
        return {}

    text = text.strip()

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

    # Direct JSON
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Extract JSON-like object
    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:
            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Python dictionary fallback
        try:
            result = ast.literal_eval(
                candidate
            )

            if isinstance(result, dict):
                return result

        except (
            ValueError,
            SyntaxError,
        ):
            pass

    return {}


# ============================================================
# NUMBER EXTRACTION
# ============================================================

# ============================================================
# NUMBER EXTRACTION
# ============================================================

def _normalize_number(
    value: str
) -> str:

    """
    Normalize numeric formatting so values such as:

    $281,724 million
    $ 281,724
    281,724

    all share the same numeric core:

    281724
    """

    value = value.strip().lower()

    value = value.replace(
        "$",
        ""
    )

    value = value.replace(
        ",",
        ""
    )

    value = value.replace(
        "million",
        ""
    )

    value = value.replace(
        "billion",
        ""
    )

    value = value.replace(
        "%",
        ""
    )

    return value.strip()


def extract_numeric_values(
    text: str
) -> set[str]:

    values = set()

    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    percentages = re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text,
        flags=re.IGNORECASE
    )

    for value in percentages:

        normalized = _normalize_number(
            value
        )

        values.add(
            f"percent:{normalized}"
        )

    # --------------------------------------------------------
    # DOLLAR VALUES
    #
    # Examples:
    # $281,724
    # $ 281,724
    # $281,724 million
    # $168.9 billion
    # --------------------------------------------------------

    money_values = re.findall(
        r"\$\s*"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?:\s*(?:million|billion))?",
        text,
        flags=re.IGNORECASE
    )

    for value in money_values:

        normalized = _normalize_number(
            value
        )

        values.add(
            f"number:{normalized}"
        )

    # --------------------------------------------------------
    # COMMA-FORMATTED TABLE VALUES
    #
    # Important for PDF tables where:
    #
    # (In millions)
    # Revenue 281,724 245,122 211,915
    #
    # The currency symbol/unit may not be repeated
    # beside every number after PDF extraction.
    # --------------------------------------------------------

    table_numbers = re.findall(
        r"(?<![\d.])"
        r"\d{1,3}(?:,\d{3})+"
        r"(?:\.\d+)?"
        r"(?![\d.])",
        text,
        flags=re.IGNORECASE
    )

    for value in table_numbers:

        normalized = _normalize_number(
            value
        )

        values.add(
            f"number:{normalized}"
        )

    return values


# ============================================================
# NUMERIC SUPPORT CHECK
# ============================================================

def check_numeric_support(
    answer: str,
    context: str,
) -> list[str]:

    answer_values = extract_numeric_values(
        answer
    )

    context_values = extract_numeric_values(
        context
    )

    issues = []

    for value in sorted(
        answer_values
    ):

        if value not in context_values:

            # Convert internal representation
            # into readable verifier feedback.

            if value.startswith(
                "percent:"
            ):

                display_value = (
                    value.replace(
                        "percent:",
                        ""
                    )
                    + "%"
                )

            else:

                display_value = value.replace(
                    "number:",
                    ""
                )

            issues.append(
                f"Numeric value '{display_value}' "
                "is not present in the retrieved evidence."
            )

    return issues


# ============================================================
# DIRECTION HELPERS
# ============================================================

INCREASE_WORDS = [
    "increase",
    "increased",
    "grew",
    "growth",
    "rose",
]

DECREASE_WORDS = [
    "decrease",
    "decreased",
    "decline",
    "declined",
    "fell",
]


def _get_direction(
    sentence: str
) -> str | None:

    has_increase = any(
        word in sentence
        for word in INCREASE_WORDS
    )

    has_decrease = any(
        word in sentence
        for word in DECREASE_WORDS
    )

    if (
        has_increase
        and not has_decrease
    ):
        return "increase"

    if (
        has_decrease
        and not has_increase
    ):
        return "decrease"

    return None


# ============================================================
# DIRECTION CHECK
# ============================================================

def check_direction_conflicts(
    answer: str,
    context: str,
) -> list[str]:

    issues = []

    answer_lower = answer.lower()
    context_lower = context.lower()

    categories = [
        "server products and cloud services revenue",
        "azure and other cloud services revenue",
        "intelligent cloud revenue",
        "server products revenue",
        "operating income",
        "net income",
    ]

    answer_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?\n]+",
            answer_lower
        )
        if sentence.strip()
    ]

    context_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?\n]+",
            context_lower
        )
        if sentence.strip()
    ]

    for category in categories:

        answer_related = [
            sentence
            for sentence in answer_sentences
            if category in sentence
        ]

        context_related = [
            sentence
            for sentence in context_sentences
            if category in sentence
        ]

        if not answer_related:
            continue

        if not context_related:
            continue

        for answer_sentence in answer_related:

            answer_direction = _get_direction(
                answer_sentence
            )

            if answer_direction is None:
                continue

            context_directions = set()

            for context_sentence in context_related:

                direction = _get_direction(
                    context_sentence
                )

                if direction is not None:
                    context_directions.add(
                        direction
                    )

            if (
                context_directions == {"increase"}
                and answer_direction == "decrease"
            ):

                issues.append(
                    f"Direction conflict for '{category}': "
                    "answer says decrease while evidence says increase."
                )

            elif (
                context_directions == {"decrease"}
                and answer_direction == "increase"
            ):

                issues.append(
                    f"Direction conflict for '{category}': "
                    "answer says increase while evidence says decrease."
                )

    return list(
        dict.fromkeys(
            issues
        )
    )


# ============================================================
# FALSE ABSENCE / MISSING-INFORMATION CLAIM CHECK
# ============================================================

def check_absence_claims(
    answer: str,
    context: str,
) -> list[str]:

    issues = []

    answer_lower = answer.lower()

    # --------------------------------------------------------
    # CLAIM: NO SUPPORTING FIGURES
    # --------------------------------------------------------

    no_figure_phrases = [
        "no supporting figures",
        "no figures are available",
        "no supporting figures are explicitly available",
        "no figures were provided",
    ]

    context_values = extract_numeric_values(
        context
    )

    if (
        any(
            phrase in answer_lower
            for phrase in no_figure_phrases
        )
        and context_values
    ):

        issues.append(
            "The answer claims that supporting figures are unavailable, "
            "but the retrieved evidence contains numeric supporting figures."
        )

    # --------------------------------------------------------
    # CLAIM: NO PERCENTAGE
    # --------------------------------------------------------

    no_percentage_phrases = [
        "no specific percentage",
        "no percentage is mentioned",
        "no percentage mentioned",
        "no percentage is available",
    ]

    context_has_percentage = bool(
        re.search(
            r"\b\d+(?:\.\d+)?%",
            context
        )
    )

    if (
        any(
            phrase in answer_lower
            for phrase in no_percentage_phrases
        )
        and context_has_percentage
    ):

        issues.append(
            "The answer claims that no percentage is available, "
            "but the retrieved evidence contains percentage values."
        )

    # --------------------------------------------------------
    # CLAIM: INFORMATION NOT PROVIDED
    # --------------------------------------------------------

    missing_information_phrases = [
        "the provided documents do not specify",
        "the documents do not specify",
        "information is not provided",
    ]

    if (
        any(
            phrase in answer_lower
            for phrase in missing_information_phrases
        )
        and context_values
    ):

        issues.append(
            "The answer claims that information is not provided, "
            "but the retrieved evidence contains potentially relevant "
            "numeric information."
        )

    return list(
        dict.fromkeys(
            issues
        )
    )


# ============================================================
# SEMANTIC SUPPORT CHECK
# ============================================================

def semantic_verify(
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:

    prompt = SEMANTIC_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )

    response = _verifier_llm.invoke(
        prompt
    )

    raw_output = getattr(
        response,
        "content",
        str(response)
    )

    result = _extract_json(
        raw_output
    )

    if not result:

        return {
            "verdict": "REVISE",
            "issues": [
                "Semantic verifier response could not be parsed."
            ],
            "feedback": (
                "Regenerate the answer using only supported evidence."
            ),
        }

    verdict = str(
        result.get(
            "verdict",
            "REVISE"
        )
    ).upper()

    if verdict not in {
        "PASS",
        "REVISE",
    }:
        verdict = "REVISE"

    issues = result.get(
        "issues",
        []
    )

    if not isinstance(
        issues,
        list
    ):
        issues = [
            str(issues)
        ]

    issues = [
        str(issue).strip()
        for issue in issues
        if str(issue).strip()
    ]

    feedback = str(
        result.get(
            "feedback",
            ""
        )
    ).strip()

    return {
        "verdict": verdict,
        "issues": issues,
        "feedback": feedback,
    }


# ============================================================
# FINAL HYBRID VERIFIER
# ============================================================

def verify_answer(
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # CHECK 1: NUMERIC SUPPORT
    # --------------------------------------------------------

    numeric_issues = check_numeric_support(
        answer,
        context
    )

    # --------------------------------------------------------
    # CHECK 2: DIRECTION
    # --------------------------------------------------------

    direction_issues = check_direction_conflicts(
        answer,
        context
    )

    # --------------------------------------------------------
    # CHECK 3: FALSE ABSENCE CLAIMS
    # --------------------------------------------------------

    absence_issues = check_absence_claims(
        answer,
        context
    )

    # --------------------------------------------------------
    # CHECK 4: SEMANTIC SUPPORT
    #
    # Advisory only.
    # It does NOT control PASS / REVISE because llama3.2
    # can occasionally produce false-positive judgments.
    # --------------------------------------------------------

    semantic_result = semantic_verify(
        question=question,
        answer=answer,
        context=context,
    )

    semantic_issues = semantic_result.get(
        "issues",
        []
    )

    semantic_feedback = semantic_result.get(
        "feedback",
        ""
    )

    # --------------------------------------------------------
    # BLOCKING ISSUES
    #
    # Only deterministic checks control the final verdict.
    # --------------------------------------------------------

    blocking_issues = (
        numeric_issues
        + direction_issues
        + absence_issues
    )

    blocking_issues = list(
        dict.fromkeys(
            blocking_issues
        )
    )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    if blocking_issues:
        verdict = "REVISE"
    else:
        verdict = "PASS"

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    feedback_parts = []

    if numeric_issues:
        feedback_parts.append(
            "Correct unsupported numeric values."
        )

    if direction_issues:
        feedback_parts.append(
            "Correct increase/decrease contradictions."
        )

    if absence_issues:
        feedback_parts.append(
            "Do not state that figures or percentages are unavailable "
            "when they are present in the retrieved evidence."
        )

    if verdict == "REVISE":

        if feedback_parts:
            feedback = " ".join(
                feedback_parts
            )

        else:
            feedback = (
                "Revise the answer so every claim "
                "matches the retrieved evidence."
            )

    else:

        feedback = (
            "Deterministic grounding checks passed."
        )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "verdict": verdict,

        "supported": (
            verdict == "PASS"
        ),

        "issues": blocking_issues,

        "feedback": feedback,

        "checks": {
            "numeric_check": (
                len(numeric_issues) == 0
            ),

            "direction_check": (
                len(direction_issues) == 0
            ),

            "absence_check": (
                len(absence_issues) == 0
            ),

            "semantic_check": (
                semantic_result.get(
                    "verdict"
                ) == "PASS"
            ),
        },

        "semantic_advisory": {
            "verdict": semantic_result.get(
                "verdict",
                "UNKNOWN"
            ),

            "issues": semantic_issues,

            "feedback": semantic_feedback,
        },
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Enterprise RAG Hybrid Verifier"
    )

    print("=" * 70)

    question = input(
        "\nQuestion: "
    ).strip()

    answer = input(
        "\nGenerated answer: "
    ).strip()

    print(
        "\nPaste retrieved evidence below."
    )

    print(
        "Type END on a new line "
        "when finished.\n"
    )

    evidence_lines = []

    while True:

        line = input()

        if line.strip() == "END":
            break

        evidence_lines.append(
            line
        )

    context = "\n".join(
        evidence_lines
    )

    result = verify_answer(
        question=question,
        answer=answer,
        context=context,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VERIFICATION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )