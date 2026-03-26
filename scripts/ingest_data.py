import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
DATA_PATH = "data/" 
DB_PATH = "db/awaz_e_nisa_db"

def build_vector_db():
    print("Step 1: Files are loading & Pre-processing metadata...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: '{DATA_PATH}' folder not found!")
        return

    # Loading PDFs
    loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # --- METADATA ENRICHMENT (Preprocessing Step) ---
    for doc in documents:
        # Extracting source filename for better reference
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        doc.metadata["file_name"] = source_file
        # Clean text a bit (Removing redundant whitespace)
        doc.page_content = " ".join(doc.page_content.split())

    print(f"Total Pages Processed: {len(documents)}")

    print("Step 2: Strategic Chunking (Recursive Splitting)...")
    # 1000 chunk size with 150 overlap ensures no legal context is cut in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""] # Priority based splitting
    )
    splits = text_splitter.split_documents(documents)

    print("Step 3: Generating Neural Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Step 4: Saving to ChromaDB (Module 2: Vector Space)...")
    vector_db = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"✅ Database saved successfully with {len(splits)} optimized chunks.")

if __name__ == "__main__":
    build_vector_db()