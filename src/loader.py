from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(folder_path):
    all_docs = []

    for pdf_file in Path(folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        all_docs.extend(docs)

    return all_docs