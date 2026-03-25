import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
DB_PATH = "db/awaz_e_nisa_db"

# 2. LOAD BRAIN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. LLM SETUP (Gemini 3 Flash Preview)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1, # Low temperature for more factual responses
    max_output_tokens=2048
)

# 4. STRICT PROMPT DESIGN (The Guardrail)
template = """
You are 'Awaz-e-Nisa', a specialized AI Legal Assistant for women's rights in Pakistan.

### STRICT RULES:
1. ONLY answer questions related to Pakistani Law, Women's Rights, Maintenance (Kharch-e-Paandan), Khula, Nikkah, and Dower (Mehar).
2. If the user asks a question that is NOT related to law or the provided legal context (e.g., general knowledge, science, cooking, or 'how to make honey from ice'), you must politely state that your expertise is limited to legal assistance for women in Pakistan.
3. If you don't know the answer based on the context, say that you don't have enough information from the legal records but provide general legal guidance for that topic.

### LEGAL CONTEXT:
{context}

### USER QUESTION: 
{question}

### Analysis (Include Merits, Demerits, and Success Rate ONLY if relevant to law):
"""

prompt = ChatPromptTemplate.from_template(template)

# 5. THE CHAIN
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LangChain Expression Language
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. EXECUTION
if __name__ == "__main__":
    # Test with the "Honey" query to see the guardrail in action
    user_issue = "How to make honey from ice?"
    
    print("\n--- Awaz-e-Nisa: Analyzing Case... ---")
    
    response = rag_chain.invoke(user_issue)
    print("\n" + response)