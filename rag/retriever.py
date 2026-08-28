from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("Loading embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Connecting to ChromaDB...")

vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

print("Connected successfully!")

# Ask a question
question = input("\nEnter your question: ")

print("\nSearching...\n")

results = vectorstore.similarity_search(
    question,
    k=3
)

print("=" * 70)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 70)
    print("Page:", doc.metadata["page"] + 1)
    print()
    print(doc.page_content[:600])

print("=" * 70)