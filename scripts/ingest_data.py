import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Path to returned pdfs
DATA_PATH = "data/" 
# Path to save vector database
DB_PATH = "db/awaz_e_nisa_db"

def build_vector_db():
    print("Step 1: Files are loading...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: '{DATA_PATH}' folder not found!")
        return
# Load all PDFs from the data directory

    loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
# 1.File reading
# 2. Text extraction
# 3. Metadata creation (filename, page number)
    documents = loader.load()
    print(f"Total Pages Loaded: {len(documents)}")

    print("Step 2: Text is being split into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    print("Step 3: Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Step 4: Saving to database...")
    vector_db = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"Database saved with {len(splits)} chunks.")

if __name__ == "__main__":
    build_vector_db()