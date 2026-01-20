# app.py
# Swiggy RAG – Chat UI + Source Highlighting + Faster Retrieval

import streamlit as st
from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# -------------------------
# Config
# -------------------------
INDEX_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are an assistant answering questions ONLY from the Swiggy Annual Report context.

Rules:
- Do not use outside knowledge
- If answer is missing, say exactly:
  "Information not available in the report."

Context:
{context}

Question:
{question}

Answer:
"""


# -------------------------
# Load RAG (cached)
# -------------------------
@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ⚡ Faster & better retrieval (MMR)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    llm = OllamaLLM(
        model="llama3.2",   # or phi3 / llama3 / mistral
        temperature=0.2
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
    )

    return rag_chain, retriever


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Swiggy RAG Chat",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔 Swiggy Annual Report – RAG Chat")
st.caption("-- by AyushM")

rag_chain, retriever = load_rag()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask a question about Swiggy Annual Report...")

if user_query:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke({"question": user_query})
            docs = retriever.invoke(user_query)

        st.markdown(answer)

        # 📎 Source highlighting
        with st.expander("📚 View source chunks"):
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Chunk {i} (Page {page})**")
                st.write(doc.page_content)
                st.markdown("---")

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
