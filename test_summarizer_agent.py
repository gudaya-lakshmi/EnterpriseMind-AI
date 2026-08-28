from agents.summarizer_agent import summarize_documents


# ============================================================
# TEST DATA
# ============================================================

question = (
    "Summarize Intelligent Cloud performance "
    "in fiscal year 2025."
)


context = """
[Context Chunk 1]

Intelligent Cloud revenue increased in fiscal year 2025.
Azure and other cloud services revenue grew 34%, driven by
demand for our portfolio of services.

---

[Context Chunk 2]

Microsoft Cloud gross margin percentage decreased to 69%,
driven by the impact of scaling AI infrastructure, offset
in part by efficiency gains in Azure.

---

[Context Chunk 3]

Server products revenue decreased 3%, driven by a decrease
in transactional purchasing with continued customer shift
to cloud offerings.
"""


# ============================================================
# RUN SUMMARIZER AGENT
# ============================================================

print("=" * 70)
print("SUMMARIZER AGENT TEST")
print("=" * 70)


result = summarize_documents(
    question=question,
    context=context,
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print(
    "\nSupported:",
    result["supported"]
)

print("\nSummary:\n")

print(
    result["summary"]
)

print(
    "\n" + "=" * 70
)