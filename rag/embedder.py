from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

print("Step 1: Loading PDF...")

# Load PDF
loader = PyPDFLoader("data/pdfs/AnnualReport.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

print("\nStep 2: Chunking...")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("\nStep 3: Loading Embedding Model...")

# Load embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model loaded successfully.")

print("\nStep 4: Creating embedding for the first chunk...")

# Generate embedding for first chunk
vector = embeddings.embed_query(chunks[0].page_content)

print("Embedding created successfully!")

print(f"\nVector Length: {len(vector)}")

print("\nFirst 10 values of the vector:")
print(vector[:10])