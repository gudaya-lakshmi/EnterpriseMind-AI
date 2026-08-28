from langchain_community.document_loaders import PyPDFLoader

print("Step 1: Import successful")

pdf_path = "data/pdfs/AnnualReport.pdf"
print("Step 2: PDF Path =", pdf_path)

loader = PyPDFLoader(pdf_path)
print("Step 3: Loader created")

documents = loader.load()
print("Step 4: PDF Loaded Successfully")

print(f"Total Pages: {len(documents)}")

print("\nFirst Page Metadata:")
print(documents[0].metadata)

print("\nFirst Page Preview:")
print("Length:", len(documents[0].page_content))
print(repr(documents[0].page_content[:500]))
print("\nSecond Page Preview:\n")
print(documents[1].page_content[:500])