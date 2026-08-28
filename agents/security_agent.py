import re
from typing import Dict, List


# ============================================================
# SECURITY CONFIGURATION
# ============================================================

BLOCK_THRESHOLD = 4


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize user input for security analysis.
    """

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# SAFE / EDUCATIONAL CONTEXT DETECTION
# ============================================================

def is_security_discussion(query: str) -> bool:
    """
    Detect when suspicious phrases are being discussed
    academically rather than executed.

    Example:

    "Explain why 'ignore previous instructions'
    is a prompt injection attack."

    This should be allowed.
    """

    educational_patterns = [

        r"explain why .* prompt injection",

        r"what is prompt injection",

        r"what does .* prompt injection mean",

        r"why is .* considered .* prompt injection",

        r"give examples of prompt injection",

        r"how does prompt injection work",

        r"describe prompt injection",

        r"define prompt injection",

        r"discuss prompt injection",

        r"detect prompt injection",

        r"prevent prompt injection",

        r"protect against prompt injection",

        r"what are prompt injection attacks",
    ]

    for pattern in educational_patterns:

        if re.search(pattern, query):

            return True

    return False


# ============================================================
# HIGH CONFIDENCE ATTACK PATTERNS
# ============================================================

HIGH_RISK_PATTERNS = [

    # --------------------------------------------------------
    # Instruction override
    # --------------------------------------------------------

    r"ignore .*previous .*instructions",

    r"ignore .*previous .*rules",

    r"disregard .*instructions",

    r"disregard .*rules",

    r"override .*instructions",

    r"override .*rules",

    r"do not follow .*rules",

    r"do not follow .*instructions",

    r"forget .*instructions",

    r"forget .*rules",

    # --------------------------------------------------------
    # Security bypass
    # --------------------------------------------------------

    r"bypass .*security",

    r"disable .*security",

    r"turn off .*security",

    r"remove .*security",

    r"disable .*checks",

    r"bypass .*checks",

    r"ignore .*security",

    # --------------------------------------------------------
    # System prompt extraction
    # --------------------------------------------------------

    r"(show|give|print|reveal|display|provide).*system prompt",

    r"(show|give|print|reveal|display|provide).*hidden prompt",

    r"(show|give|print|reveal|display|provide).*secret prompt",

    r"(show|give|print|reveal|display|provide).*hidden instructions",

    r"(show|give|print|reveal|display|provide).*secret instructions",

    r"(show|give|print|reveal|display|provide).*internal instructions",

    # Indirect prompt leakage attempts

    r"(show|give|print|reveal|display|provide).*exact instructions.*(provided|given).*before",

    r"(show|give|print|reveal|display|provide).*instructions.*before (my|this) question",

    r"(show|give|print|reveal|display|provide).*instructions.*before.*message",
    # --------------------------------------------------------
    # Raw context extraction
    # --------------------------------------------------------

    r"(show|print|reveal|display|give|provide).*raw context",

    r"(show|print|reveal|display|give|provide).*internal context",

    r"(show|print|reveal|display|give|provide).*hidden context",

    r"(inspect|show|print|reveal).*protected context",

    # --------------------------------------------------------
    # Retrieved chunk extraction
    # --------------------------------------------------------

    r"(output|print|show|reveal).*retrieved chunk",

    r"(output|print|show|reveal).*retrieved document",

    r"every retrieved chunk",

    r"all retrieved chunks",

    # --------------------------------------------------------
    # Private / internal data exfiltration
    # --------------------------------------------------------

    r"(return|show|print|reveal|dump|give).*private internal data",

    r"(return|show|print|reveal|dump|give).*confidential internal data",

    r"(return|show|print|reveal|dump|give).*protected data",

    r"dump .*internal data",

    r"leak .*internal data",

    # --------------------------------------------------------
    # Role hijacking
    # --------------------------------------------------------

    r"pretend .*no restrictions",

    r"act .*no restrictions",

    r"you are now .*unrestricted",

    r"pretend .*rules do not apply",

    # --------------------------------------------------------
    # Priority manipulation
    # --------------------------------------------------------

    r"my instructions .*higher priority",

    r"my instructions .*override",

    r"these instructions .*higher priority",

    r"new instructions .*higher priority",
]


# ============================================================
# SUSPICIOUS INTENT GROUPS
# ============================================================

OVERRIDE_TERMS = [
    "ignore",
    "disregard",
    "override",
    "forget",
    "bypass",
    "disable",
    "do not follow",
    "higher priority",
    "no restrictions",
]


PROTECTED_INFORMATION_TERMS = [
    "system prompt",
    "hidden prompt",
    "secret prompt",
    "internal prompt",
    "hidden instructions",
    "secret instructions",
    "internal instructions",
    "raw context",
    "internal context",
    "hidden context",
    "protected context",
    "retrieved chunks",
    "retrieved chunk",
    "private internal data",
    "confidential data",
]


EXTRACTION_TERMS = [
    "show",
    "reveal",
    "print",
    "display",
    "give",
    "provide",
    "return",
    "output",
    "dump",
    "reproduce",
    "inspect",
]


AUTHORITY_MANIPULATION_TERMS = [
    "system administrator",
    "administrator",
    "developer",
    "security auditor",
    "security audit",
    "authorized",
    "i authorize",
]


EXACT_EXTRACTION_TERMS = [
    "word for word",
    "exactly",
    "verbatim",
    "complete",
    "entire",
    "all",
    "every",
]


# ============================================================
# HELPER
# ============================================================

def contains_any(
    query: str,
    terms: List[str]
) -> bool:

    return any(
        term in query
        for term in terms
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

def calculate_risk_score(
    query: str
) -> Dict:

    score = 0

    reasons = []


    # --------------------------------------------------------
    # High confidence patterns
    # --------------------------------------------------------

    for pattern in HIGH_RISK_PATTERNS:

        if re.search(pattern, query):

            score += 5

            reasons.append(
                "High-confidence attack pattern detected."
            )

            break


    # --------------------------------------------------------
    # Instruction override
    # --------------------------------------------------------

    if contains_any(
        query,
        OVERRIDE_TERMS
    ):

        score += 2

        reasons.append(
            "Instruction override language detected."
        )


    # --------------------------------------------------------
    # Protected information
    # --------------------------------------------------------

    protected = contains_any(
        query,
        PROTECTED_INFORMATION_TERMS
    )

    if protected:

        score += 2

        reasons.append(
            "Request references protected internal information."
        )


    # --------------------------------------------------------
    # Extraction intention
    # --------------------------------------------------------

    extraction = contains_any(
        query,
        EXTRACTION_TERMS
    )

    if extraction:

        score += 1


    # Extraction + protected information is dangerous

    if extraction and protected:

        score += 2

        reasons.append(
            "Attempt to extract protected information detected."
        )


    # --------------------------------------------------------
    # Authority manipulation
    # --------------------------------------------------------

    authority = contains_any(
        query,
        AUTHORITY_MANIPULATION_TERMS
    )

    if authority:

        score += 1

        reasons.append(
            "Authority or authorization claim detected."
        )


    if authority and protected:

        score += 2

        reasons.append(
            "Authority claim combined with protected-data request."
        )


    # --------------------------------------------------------
    # Exact reproduction
    # --------------------------------------------------------

    exact_request = contains_any(
        query,
        EXACT_EXTRACTION_TERMS
    )

    if exact_request and protected:

        score += 2

        reasons.append(
            "Exact reproduction of protected content requested."
        )


    return {
        "score": score,
        "reasons": reasons
    }


# ============================================================
# MAIN SECURITY DETECTOR
# ============================================================

def detect_prompt_injection(
    query: str
) -> Dict:

    normalized_query = normalize_text(
        query
    )


    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not normalized_query:

        return {
            "allowed": False,
            "security_status": "BLOCK",
            "security_reason": "Empty user query.",
            "risk_score": 10
        }


    # --------------------------------------------------------
    # Legitimate security discussion
    # --------------------------------------------------------

    if is_security_discussion(
        normalized_query
    ):

        return {
            "allowed": True,
            "security_status": "ALLOW",
            "security_reason":
                "Security-related content appears educational or analytical.",
            "risk_score": 0
        }


    # --------------------------------------------------------
    # Risk analysis
    # --------------------------------------------------------

    analysis = calculate_risk_score(
        normalized_query
    )

    score = analysis["score"]

    reasons = analysis["reasons"]


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if score >= BLOCK_THRESHOLD:

        reason = "; ".join(
            dict.fromkeys(reasons)
        )

        return {
            "allowed": False,
            "security_status": "BLOCK",
            "security_reason": reason,
            "risk_score": score
        }


    return {
        "allowed": True,
        "security_status": "ALLOW",
        "security_reason":
            "No significant prompt-injection or protected-data extraction risk detected.",
        "risk_score": score
    }


# ============================================================
# LANGGRAPH SECURITY AGENT
# ============================================================

def security_agent(
    state: dict
) -> dict:

    question = state.get(
        "question",
        ""
    )

    result = detect_prompt_injection(
        question
    )

    print("\n[Security Agent]")

    print(
        f"Decision: {result['security_status']}"
    )

    print(
        f"Risk Score: {result['risk_score']}"
    )

    print(
        f"Reason: {result['security_reason']}"
    )


    return {

        **state,

        "security_status":
            result["security_status"],

        "security_reason":
            result["security_reason"],

        "security_allowed":
            result["allowed"],

        "security_risk_score":
            result["risk_score"]
    }


# ============================================================
# SAFE REJECTION NODE
# ============================================================

def security_rejection_node(
    state: dict
) -> dict:

    safe_response = (
        "I cannot provide hidden system instructions, "
        "internal prompts, protected retrieval context, "
        "private internal data, or comply with attempts "
        "to override the system's security controls."
    )

    print("\n[Security Agent]")
    print("Request blocked.")

    return {

        **state,

        "answer": safe_response,

        "final_answer": safe_response,

        "verification": "BLOCKED",

        "citations": []
    }


# ============================================================
# LANGGRAPH ROUTING
# ============================================================

def route_after_security(
    state: dict
) -> str:

    if state.get(
        "security_allowed",
        False
    ):

        return "allowed"

    return "blocked"