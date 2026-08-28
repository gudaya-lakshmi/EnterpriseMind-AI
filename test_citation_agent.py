from langchain_core.documents import Document

from agents.citation_agent import (
    find_citations,
    print_citations,
)


# ----------------------------------------------------
# TEST QUESTION
# ----------------------------------------------------

question = (
    "What drove Intelligent Cloud revenue growth "
    "in fiscal year 2025?"
)


# ----------------------------------------------------
# FINAL VERIFIED ANSWER
# Contains TWO claims supported by different chunks
# ----------------------------------------------------

answer = (
    "Intelligent Cloud revenue increased by 21%. "
    "Azure and other cloud services revenue grew 34%, "
    "driven by demand for the company's portfolio of services."
)


# ----------------------------------------------------
# SAMPLE RERANKED DOCUMENTS
# ----------------------------------------------------

documents = [

    # Supports claim 1
    Document(
        page_content=(
            "Intelligent Cloud revenue increased 21% "
            "in fiscal year 2025."
        ),
        metadata={
            "source": "data/pdfs/AnnualReport.pdf",
            "page": 24,
        },
    ),

    # Unrelated
    Document(
        page_content=(
            "Productivity and Business Processes revenue "
            "increased during fiscal year 2025."
        ),
        metadata={
            "source": "data/pdfs/AnnualReport.pdf",
            "page": 21,
        },
    ),

    # Supports claim 2
    Document(
        page_content=(
            "Azure and other cloud services revenue grew "
            "34%, driven by demand for our portfolio "
            "of services."
        ),
        metadata={
            "source": "data/pdfs/AnnualReport.pdf",
            "page": 26,
        },
    ),

    # Unrelated
    Document(
        page_content=(
            "Operating expenses increased during "
            "the fiscal year."
        ),
        metadata={
            "source": "data/pdfs/AnnualReport.pdf",
            "page": 62,
        },
    ),
]


# ----------------------------------------------------
# RUN CITATION AGENT
# ----------------------------------------------------

print("=" * 70)
print("MULTIPLE CITATION TEST")
print("=" * 70)

result = find_citations(
    question=question,
    answer=answer,
    documents=documents,
)


# ----------------------------------------------------
# SHOW RESULT
# ----------------------------------------------------

print(
    "\nSupported:",
    result["supported"]
)

print_citations(
    result
)

print("\nSelected citation details:")

for citation in result["citations"]:

    print(
        f"\nChunk ID: {citation['chunk_id']}"
    )

    print(
        f"Source: {citation['source']}"
    )

    print(
        f"Page: {citation['page']}"
    )

    print(
        f"Reason: {citation['reason']}"
    )

print("\n" + "=" * 70)