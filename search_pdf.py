from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("data/pdfs/AnnualReport.pdf")
documents = loader.load()

search_terms = [
    "net income increased",
    "net income was",
    "revenue increased",
    "revenue was"
]

for page_number, document in enumerate(documents, start=1):

    text = document.page_content.lower()

    for term in search_terms:

        if term in text:

            print("\n" + "=" * 70)
            print("PAGE:", page_number)
            print("FOUND:", term)
            print("=" * 70)

            position = text.find(term)

            start = max(0, position - 500)
            end = min(len(document.page_content), position + 1500)

            print(document.page_content[start:end])