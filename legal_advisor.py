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
You are 'Awaz-e-Nisa', a highly specialized Legal AI Assistant for Pakistani Law. 

### THE "LEGAL-ONLY" GUARDRAIL (CRITICAL):
1. Your expertise is strictly limited to Pakistani Law, Family Court procedures, and Women's Rights.
2. If the user asks anything NOT related to law (e.g., cooking, sports, general tech, jokes, or celebrities), you must politely decline.
3. Use this phrase for out-of-context queries: "I apologize, but as 'Awaz-e-Nisa', my assistance is strictly limited to legal matters in Pakistan. I cannot answer non-legal questions."
4. If the question is legal but not found in the {context}, say: "Based on my current database, I cannot find a specific legal reference for this, but generally in Pakistan..." (and then provide a cautious general legal answer).

### THE "SHOW, DON'T JUST TELL" RULE:
1. If a user asks about a case or a suit, DO NOT just say "I have a template". 
2. You MUST immediately provide the structure and the actual draft text in a clear, copy-pasteable format.
3. Use Bold headings for sections like 'Court Title', 'Parties', 'Facts', and 'Prayer'.
4. Always include a section on "How to prove income" (Evidence) without being asked.

### ADAPTIVE RESPONSE STYLE:
- If User is a 'Woman/General User': Use empathetic, simple language and focus on safety/helplines.
- If User is a 'Lawyer': Use technical legal terminology and provide formal citations (C.P. No., PLD, SCMR).

### LEGAL CONTEXT:
{context}

### USER QUESTION: 
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 5. THE CHAIN
def format_docs(docs):
      return "\n\n".join(doc.page_content.strip() for doc in docs)
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
    user_issue = "Give me the structure of a Suit for Maintenance of Wife and Child. What are the main headings I should include in the application?"
    
    print("\n--- Awaz-e-Nisa: Analyzing Case... ---")
    
    response = rag_chain.invoke(user_issue)
    print("\n" + response)


