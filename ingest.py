# This handles document loading, chunking, embedding, and storing in FAISS vector store.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data\Annual-Report-FY-2023-24.pdf"  

def ingest_pdf():
    # Step 1: Load the PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the PDF.")

    # Step 2: Preprocess and split into chunks
    # Basic cleaning: Strip extra whitespace (optional, can add more advanced cleaning if needed)
    for doc in documents:
        doc.page_content = ' '.join(doc.page_content.split())  # Remove multiple spaces/newlines

    # Split into meaningful chunks with metadata (e.g., page numbers preserved)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      
        chunk_overlap=200,   
        length_function=len,  
        add_start_index=True  # Add metadata for start index in original doc
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Step 3: Generate embeddings using open-source model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight, effective model
        model_kwargs={'device': 'cpu'}  # Use CPU; change to 'cuda' if GPU available
    )

    # Step 4: Store embeddings in FAISS vector database
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")  # Save to local folder for reuse
    print("Vector store created and saved to 'faiss_index'.")

if __name__ == "__main__":
    ingest_pdf()