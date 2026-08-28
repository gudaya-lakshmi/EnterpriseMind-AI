from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

query = """
Find the consolidated summary results of operations for fiscal year 2025
compared with fiscal year 2024.

Retrieve the total company revenue, net income, percentage changes,
and management's stated reasons for the changes.

Prioritize consolidated company results, not individual segment results.
"""

results = vectorstore.similarity_search(
    query,
    k=10
)

for index, document in enumerate(results, start=1):
    print("\n" + "=" * 70)
    print(f"RESULT {index}")
    print("=" * 70)

    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page_number"))

    print("-" * 70)
    print(document.page_content)