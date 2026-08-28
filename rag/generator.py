from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

print("Loading embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Connecting to ChromaDB...")

vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

print("Loading Llama 3.2...")

llm = ChatOllama(
    model="llama3.2"
)

print("Ready!")

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    print("\nSearching documents...")

    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an Enterprise RAG Assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, say:

"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    print("\nGenerating answer...\n")

    response = llm.invoke(prompt)

    print("=" * 70)
    print(response.content)
    print("=" * 70)