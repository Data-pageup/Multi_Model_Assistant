from langchain_chroma import Chroma


def create_vector_store(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store

def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever