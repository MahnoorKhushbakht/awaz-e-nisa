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
    temperature=0.1, 
    max_output_tokens=2048
)

# 4. STRICT PROMPT DESIGN (With Language & Lawyer Logic)
template = """
You are 'Awaz-e-Nisa', a highly specialized Legal AI Assistant for Pakistani Law. 

### LANGUAGE ADAPTIVITY RULE:
- If the user's question is in English, respond in English.
- If the user's question is in Roman Urdu (Urdu in English script), respond strictly in Roman Urdu.
- Consistency is key: Do not mix scripts unless providing a specific legal term.

### THE "LEGAL-ONLY" GUARDRAIL (CRITICAL):
1. Your expertise is strictly limited to Pakistani Law, Family Court procedures, and Women's Rights.
2. If the user asks anything NOT related to law, you must politely decline.
3. Out-of-context response (English): "I apologize, but as 'Awaz-e-Nisa', my assistance is strictly limited to legal matters in Pakistan. I cannot answer non-legal questions."
4. Out-of-context response (Roman Urdu): "I apologize, lekin 'Awaz-e-Nisa' honay ke natay meri madad sirf Pakistan ke qanooni masail tak mahdood hai. Main ghair-qanooni sawalat ka jawab nahi dey sakti."

### ADAPTIVE RESPONSE STYLE & RULES:
- **If User is 'Woman/General User':** Use empathetic, simple language. Include safety helplines at the end.
- **If User is 'Lawyer':** Use technical legal terminology and provide formal citations (C.P. No., PLD, SCMR). **DO NOT** include general helplines or safety codes for Lawyer queries.

### THE "SHOW, DON'T JUST TELL" RULE:
1. Provide the actual draft text in a clear, copy-pasteable format.
2. Use Bold headings for 'Court Title', 'Parties', 'Facts', and 'Prayer'.
3. Always include a section on "How to prove income" (Evidence).

### LEGAL CONTEXT:
{context}

### USER QUESTION: 
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 5. THE CHAIN
def format_docs(docs):
      return "\n\n".join(doc.page_content.strip() for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
) 

# 6. EXECUTION
if __name__ == "__main__":
    # Test cases to verify the logic
    test_query = "Give me the structure of a Suit for Maintenance of Wife and Child."
    
    print("\n--- Awaz-e-Nisa: Analyzing Case... ---")
    response = rag_chain.invoke(test_query)
    print("\n" + response)