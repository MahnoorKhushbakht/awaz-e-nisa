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

# 3. LLM SETUP
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1,
    max_output_tokens=2048
)

# ─────────────────────────────────────────────
# 4. MAIN RAG CHAIN (unchanged)
# ─────────────────────────────────────────────
main_template = """
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

prompt = ChatPromptTemplate.from_template(main_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content.strip() for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# 5. MERITS & DEMERITS CHAIN  (NEW)
# ─────────────────────────────────────────────
merits_template = """
You are 'Awaz-e-Nisa', a Pakistani Law specialist.

Analyze the following legal query and the provided context, then produce a **Case Strength Report**.

### OUTPUT FORMAT (use exactly these headings):
## ✅ MERITS (Strengths of the Case)
- [Point 1]
- [Point 2]
...

## ❌ DEMERITS (Weaknesses / Risks)
- [Point 1]
- [Point 2]
...

## ⚖️ OVERALL ASSESSMENT
One sentence verdict on case viability (Strong / Moderate / Weak).

### RULES:
- Be honest and balanced.
- Base analysis on Pakistani Law, Family Courts Act 1964, MFLO 1961, and relevant case law.
- Use the same language as the query (English or Roman Urdu).
- Keep each bullet concise (1-2 lines).

### LEGAL CONTEXT:
{context}

### CASE QUERY:
{question}
"""

merits_prompt = ChatPromptTemplate.from_template(merits_template)

merits_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | merits_prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# 6. OPPOSITION COUNTER ARGUMENTS CHAIN  (NEW)
# ─────────────────────────────────────────────
opposition_template = """
You are 'Awaz-e-Nisa', a Pakistani Law specialist playing Devil's Advocate.

Given the user's legal case/query, generate the **strongest possible counter-arguments** the OPPOSING PARTY might raise in court.

### OUTPUT FORMAT:
## 🔴 OPPOSITION'S LIKELY ARGUMENTS

**Argument 1: [Title]**
[Explanation — legal basis, likely evidence they'll use]

**Argument 2: [Title]**
[Explanation]

(continue for 3-5 arguments)

## 🛡️ HOW TO REBUT THESE
For each argument above, give a 1-line rebuttal strategy.

### RULES:
- Be realistic — think like an opposing lawyer.
- Cite relevant Pakistani law sections where applicable.
- Use the same language as the query (English or Roman Urdu).

### LEGAL CONTEXT:
{context}

### CASE QUERY:
{question}
"""

opposition_prompt = ChatPromptTemplate.from_template(opposition_template)

opposition_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | opposition_prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# 7. SMART TIMELINE ESTIMATOR CHAIN  (NEW)
# ─────────────────────────────────────────────
timeline_template = """
You are 'Awaz-e-Nisa', a Pakistani Law specialist with expertise in court procedures and timelines.

Based on the legal query, generate a **Realistic Court Timeline Estimate** for Pakistani courts (Family Court / High Court / Supreme Court as relevant).

### OUTPUT FORMAT:
## 📅 ESTIMATED CASE TIMELINE

| Stage | Description | Estimated Duration |
|-------|-------------|-------------------|
| 1 | Filing & Admission | X weeks |
| 2 | Notice to Respondent | X weeks |
| 3 | Written Statement | X weeks |
| 4 | Evidence / Framing of Issues | X months |
| 5 | Arguments | X months |
| 6 | Judgment | X months |
| **TOTAL** | **Full Case Duration** | **X - Y months** |

## ⚡ FACTORS THAT CAN SPEED THIS UP
- [Factor 1]
- [Factor 2]

## 🐢 FACTORS THAT CAN CAUSE DELAYS
- [Factor 1]
- [Factor 2]

## 💡 PRO TIP
One actionable tip to manage timeline expectations.

### RULES:
- Base estimates on realistic Pakistani court backlogs (Family Courts typically 1-3 years, High Court 2-5 years).
- Be honest about delays — do not give optimistic estimates.
- Use the same language as the query (English or Roman Urdu).
- If it's a Family Court matter, reference the Family Courts Act 1964 Section 12 (90-day target, rarely met).

### LEGAL CONTEXT:
{context}

### CASE QUERY:
{question}
"""

timeline_prompt = ChatPromptTemplate.from_template(timeline_template)

timeline_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | timeline_prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# 8. LEGAL DRAFT GENERATOR CHAIN  (NEW)
# ─────────────────────────────────────────────
draft_template = """
You are 'Awaz-e-Nisa', a Pakistani Law specialist. Generate a complete, formal **Legal Draft** based on the user's query.

### DRAFT FORMAT:
IN THE COURT OF [RELEVANT COURT], [CITY]

**SUIT/APPLICATION NO. ______ OF 20____**

**[CASE TYPE]**

---

**IN THE MATTER OF:**

**PLAINTIFF/PETITIONER:**
[Full Name], [Relation], Resident of [Address]
                                              ...Plaintiff/Petitioner

**VERSUS**

**DEFENDANT/RESPONDENT:**
[Full Name], [Relation], Resident of [Address]
                                              ...Defendant/Respondent

---

**FACTS OF THE CASE:**
1. [Fact 1]
2. [Fact 2]
3. [Fact 3]

**LEGAL GROUNDS:**
1. [Ground 1 with law citation]
2. [Ground 2 with law citation]

**RELIEF SOUGHT / PRAYER:**
It is, therefore, most respectfully prayed that this Honorable Court may be pleased to:
(a) [Relief 1]
(b) [Relief 2]
(c) Grant any other relief deemed just and equitable.

**VERIFICATION:**
I, [Plaintiff Name], do hereby solemnly affirm that the contents of this plaint are true to the best of my knowledge and belief.

Date: ____________
Place: ____________

[Plaintiff/Petitioner's Signature]
[Advocate's Signature & Stamp]

---

### RULES:
- Fill in all [placeholders] with realistic/standard language.
- Include relevant Pakistani law citations (MFLO 1961, CPC, CrPC, Family Courts Act 1964, etc.).
- Make it court-ready and professional.
- Use English for the draft regardless of query language (legal drafts are in English).

### LEGAL CONTEXT:
{context}

### DRAFT REQUEST:
{question}
"""

draft_prompt = ChatPromptTemplate.from_template(draft_template)

draft_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | draft_prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "I want to file for khula from my husband who refuses to divorce me."
    print("\n--- MAIN RESPONSE ---")
    print(rag_chain.invoke(test_query))
    print("\n--- MERITS/DEMERITS ---")
    print(merits_chain.invoke(test_query))
    print("\n--- TIMELINE ---")
    print(timeline_chain.invoke(test_query))
