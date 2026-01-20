# query.py
# Swiggy Annual Report RAG using Ollama (LangChain 1.2.6 compatible)

from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are answering questions strictly from the Swiggy Annual Report context below.
Do NOT use outside knowledge.

If the answer is not found in the context, reply exactly:
"Information not available in the report."

Context:
{context}

Question:
{question}

Answer:
"""


def setup_rag_chain():
    # Embeddings 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Load FAISS index
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    print("✅ FAISS index loaded")

    # Ollama LLM 
    llm = Ollama(
        model="llama3.2",   
        temperature=0.2
    )

    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # LCEL RAG pipeline (no chains)
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
    )

    return rag_chain, retriever


def query_rag(rag_chain, retriever):
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = rag_chain.invoke({"question": query})

        print("\n📌 Final Answer:")
        print(answer)

        print("\n📚 Supporting Context:")
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "N/A")
            print(f"\nChunk {i} (Page {page}):")
            print(doc.page_content[:300])
            print("-" * 50)


if __name__ == "__main__":
    rag_chain, retriever = setup_rag_chain()
    query_rag(rag_chain, retriever)
