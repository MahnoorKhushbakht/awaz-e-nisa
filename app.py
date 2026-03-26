import streamlit as st
import os

# --- 1. PAGE CONFIGURATION (SAB SE PEHLE) ---
st.set_page_config(
    page_title="Awaz-e-Nisa | Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded" # Sidebar ko force-open rakhta hai
)

# --- 2. ADVANCED CSS (VISIBILITY & THEMING) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #000000; }

    /* SIDEBAR EMERGENCY FIX */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #FF2E7E !important;
        display: block !important;
    }
    
    /* Making Sidebar Text & Icons White */
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }

    /* Fix for Sidebar Toggle Button (The Arrow) */
    [data-testid="bundle-header"] button {
        color: #FF2E7E !important;
        background: rgba(255, 46, 126, 0.1) !important;
    }

    /* Clean Headers */
    h1, h2, h3 { 
        color: #FF2E7E !important; 
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
    }

    /* Feature Cards Styling */
    .feature-card {
        background-color: #080808;
        border: 1px solid #1A1A1A;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        transition: 0.3s;
        height: 180px;
        margin-bottom: 20px;
    }
    .feature-card:hover {
        border-color: #FF2E7E;
        box-shadow: 0px 0px 20px rgba(255, 46, 126, 0.15);
    }
    .feature-icon { font-size: 30px; margin-bottom: 10px; }

    /* Hide default Streamlit elements but keep Sidebar Toggle */
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODE & SESSION LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "GENERAL USER (Woman)"

# --- 4. SIDEBAR CONTENT ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 24px; color: #FF2E7E;'>AWAZ-E-NISA</h1>", unsafe_allow_html=True)
    st.caption("AI LEGAL ASSISTANT v6.0")
    st.divider()
    
    st.subheader("⚙️ SYSTEM SETTINGS")
    selected_mode = st.radio(
        "SELECT OPERATIONAL MODE:",
        ["GENERAL USER (Woman)", "LEGAL PRO"],
        index=0 if st.session_state.current_mode == "GENERAL USER (Woman)" else 1
    )
    
    # Mode Switch Logic
    if selected_mode != st.session_state.current_mode:
        st.session_state.current_mode = selected_mode
        st.session_state.messages = [] # Clear chat on mode switch
        st.rerun()

    st.divider()
    st.info(f"Current Phase: Module 4 (Cognitive UI)")
    
    if st.button("🗑️ RESET SESSION"):
        st.session_state.messages = []
        st.rerun()

# --- 5. MAIN DASHBOARD (Only shows if no messages) ---
if not st.session_state.messages:
    st.markdown("<h1 style='text-align: center; margin-top: 30px;'>NISA INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Bridging the Justice Gap with AI-Powered Legal Support for Women in Pakistan.</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='feature-card'><h3 style='color:#FF2E7E'>🧠</h3><b>SEMANTIC INTERPRETER</b><p style='font-size:12px; color:#888'>Advanced NLP for Roman Urdu.</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='feature-card'><h3 style='color:#FF2E7E'>⚖️</h3><b>STRATEGIC RAG</b><p style='font-size:12px; color:#888'>Real-time Legal Retrieval.</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='feature-card'><h3 style='color:#FF2E7E'>📄</h3><b>SMART DRAFTING</b><p style='font-size:12px; color:#888'>Court-ready petitions.</p></div>", unsafe_allow_html=True)

# --- 6. CHAT INTERFACE ---
for message in st.session_state.messages:
    avatar = "⚖️" if message["role"] == "assistant" else "👩‍💼"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a legal question (English or Roman Urdu)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Assistant Response Logic
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("Searching Legal Repository...", expanded=False) as status:
            # Fake logic for demo or connect your real rag_chain.invoke() here
            try:
                # response = rag_chain.invoke(st.session_state.messages[-1]["content"])
                response = "This is a placeholder. Please connect your RAG chain logic."
                status.update(label="Analysis Complete", state="complete")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")