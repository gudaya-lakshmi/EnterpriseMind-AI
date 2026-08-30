import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/pdfs/AnnualReport.pdf"
VECTOR_DB_PATH = "vector_db"

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Improved chunking for annual reports
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BATCH_SIZE = 50


# ============================================================
# STEP 1: REMOVE OLD VECTOR DATABASE
# ============================================================

print("Step 1: Checking existing vector database...")

if os.path.exists(VECTOR_DB_PATH):
    shutil.rmtree(VECTOR_DB_PATH)
    print("Old vector database removed.")
else:
    print("No old vector database found.")


# ============================================================
# STEP 2: LOAD PDF
# ============================================================

print("\nStep 2: Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# ============================================================
# STEP 3: ADD USEFUL METADATA
# ============================================================

print("\nStep 3: Adding metadata...")

for page_number, document in enumerate(documents, start=1):
    document.metadata["source"] = PDF_PATH
    document.metadata["page_number"] = page_number
    document.metadata["document_type"] = "annual_report"

    # RBAC metadata
    document.metadata["category"] = "financial"

print("Metadata added successfully.")


# ============================================================
# STEP 4: CHUNK DOCUMENTS
# ============================================================

print("\nStep 4: Chunking PDF...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# ============================================================
# STEP 5: ADD CHUNK METADATA
# ============================================================

print("\nStep 5: Adding chunk metadata...")

for chunk_id, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = chunk_id
    chunk.metadata["chunk_size"] = len(chunk.page_content)

print("Chunk metadata added successfully.")


# ============================================================
# STEP 6: LOAD EMBEDDING MODEL
# ============================================================

print("\nStep 6: Loading embedding model...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

print("Embedding model loaded.")


# ============================================================
# STEP 7: CREATE CHROMA DATABASE
# ============================================================

print("\nStep 7: Creating ChromaDB...")

first_batch = chunks[:BATCH_SIZE]

vectorstore = Chroma.from_documents(
    documents=first_batch,
    embedding=embeddings,
    persist_directory=VECTOR_DB_PATH,
    collection_metadata={
        "hnsw:space": "cosine"
    }
)

for start_index in range(BATCH_SIZE, len(chunks), BATCH_SIZE):

    end_index = min(
        start_index + BATCH_SIZE,
        len(chunks)
    )

    print(
        f"Storing chunks {start_index} to {end_index}..."
    )

    vectorstore.add_documents(
        chunks[start_index:end_index]
    )


# ============================================================
# STEP 8: VERIFY DATABASE
# ============================================================

print("\nStep 8: Verifying stored data...")

stored_count = vectorstore._collection.count()

print(f"Total chunks created: {len(chunks)}")
print(f"Total chunks stored: {stored_count}")

if stored_count == len(chunks):
    print("All chunks were stored successfully.")
else:
    print("Warning: Some chunks may not have been stored.")


print("\nVector database created successfully.")
print(f"Database location: {VECTOR_DB_PATH}")