from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, get_retriever
from src.rag_chain import create_rag_chain

# Load docs
docs = load_documents("rag_experiment")


# Split docs
chunks = split_documents(docs)

# Embeddings
embeddings = get_embeddings()

# Vector Store
vector_store = create_vector_store(
    chunks=chunks,
    embeddings=embeddings
)

# Retriever
retriever = get_retriever(vector_store)

# API Key
provider = input(
    "Choose provider (groq/gemini): "
).lower()

if provider == "groq":
    api_key = input("Enter Groq API Key: ")

elif provider == "gemini":
    api_key = input("Enter Gemini API Key: ")

else:
    print("Invalid provider")
    exit()

# RAG Chain
chain = create_rag_chain(
    retriever=retriever,
    provider=provider,
    api_key=api_key
)

# Chat Loop
while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(question)

    print("\nAnswer:")
    print(answer)