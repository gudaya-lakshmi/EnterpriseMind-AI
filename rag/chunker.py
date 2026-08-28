from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/pdfs/AnnualReport.pdf")
documents = loader.load()

print(f"Documents Loaded: {len(documents)}")

# Create Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split into chunks
chunks = text_splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)