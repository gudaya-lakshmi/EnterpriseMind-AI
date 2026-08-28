from langchain_ollama import OllamaEmbeddings

print("Loading embedding model...")

emb = OllamaEmbeddings(model="nomic-embed-text")

print("Creating embeddings for multiple texts...")

texts = [
    "Artificial Intelligence",
    "Microsoft Azure",
    "Cloud Computing"
]

vectors = emb.embed_documents(texts)

print("\nSuccess!")
print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))
print("\nFirst 5 values of first vector:")
print(vectors[0][:5])