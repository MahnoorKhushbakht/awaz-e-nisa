import streamlit as st
import os
from database import init_db, add_user, verify_user, save_chat_message, get_chat_history

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Awaz-e-Nisa", page_icon="⚖️", layout="wide")

# --- 2. CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #FF2E7E !important; }
    h1, h2, h3 { color: #FF2E7E !important; }
    .feature-card {
        background-color: #080808; border: 1px solid #1A1A1A; padding: 20px;
        border-radius: 12px; text-align: center; height: 180px;
    }
    .court-box {
        background-color: #0A0A0A;
        border-left: 5px solid #FF2E7E;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .mode-tag {
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Initialize Database
init_db()

# Initialize Session States
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "messages" not in st.session_state: st.session_state.messages = []
if "current_mode" not in st.session_state: st.session_state.current_mode = "GENERAL USER (Woman)"

# --- 3. LOGIN GATE ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>AWAZ-E-NISA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>AI-Powered Legal Guidance System for Pakistan</p>", unsafe_allow_html=True)
    st.divider()

    col_info, col_login = st.columns([1.2, 1], gap="large")

    with col_info:
        st.subheader("🏛️ Judicial Jurisdictions")
        st.markdown("""
        <div class='court-box'>
            <b>SUPREME COURT OF PAKISTAN</b><br>
            <small>The highest judicial forum in Pakistan, serving as the ultimate arbiter of legal and constitutional disputes.</small>
        </div>
        <div class='court-box'>
            <b>LAHORE HIGH COURT (LHC)</b><br>
            <small>The oldest high court in Pakistan, exercising jurisdiction over the province of Punjab.</small>
        </div>
        <div class='court-box'>
            <b>ISLAMABAD HIGH COURT (IHC)</b><br>
            <small>The principal court of the federal capital, handling constitutional and administrative matters of Islamabad.</small>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 Awaz-e-Nisa uses RAG technology to retrieve precedents specifically from these bodies.")

    with col_login:
        st.subheader("🔐 Secure Access")
        
        # User selects mode at login
        start_mode = st.selectbox("I am a:", ["GENERAL USER (Woman)", "LEGAL PRO"])
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("l"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if verify_user(u, p):
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        st.session_state.current_mode = start_mode
                        # Load history from DB (including mode tags)
                        st.session_state.messages = get_chat_history(u)
                        st.rerun()
                    else: 
                        st.error("Invalid Login")
                        
        with tab2:
            with st.form("s"):
                nu = st.text_input("New Username")
                np = st.text_input("New Password", type="password")
                if st.form_submit_button("Register"):
                    if add_user(nu, np): 
                        st.success("Account Created! You can now login.")
                    else:
                        st.error("Registration failed.")

# --- 4. MAIN APP (After Login) ---
else:
    with st.sidebar:
        st.markdown(f"<h1 style='font-size: 24px; color: #FF2E7E;'>AWAZ-E-NISA</h1>", unsafe_allow_html=True)
        st.write(f"👤 User: **{st.session_state.username}**")
        st.divider()
        
        selected_mode = st.radio(
            "OPERATIONAL MODE:", 
            ["GENERAL USER (Woman)", "LEGAL PRO"],
            index=0 if st.session_state.current_mode == "GENERAL USER (Woman)" else 1
        )
        
        if selected_mode != st.session_state.current_mode:
            st.session_state.current_mode = selected_mode
            st.toast(f"Switched to {selected_mode} Mode")
            st.rerun()
            
        st.divider()
        if st.button("LOGOUT"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.messages = []
            st.rerun()

    if "rag" not in st.session_state:
        with st.spinner("Loading AI Brain..."):
            from legal_advisor import rag_chain
            st.session_state.rag = rag_chain

    if len(st.session_state.messages) == 0:
        st.markdown("<h1 style='text-align: center; margin-top: 30px;'>NISA INTELLIGENCE</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown("<div class='feature-card'><h3>🧠</h3><b>SEMANTIC INTERPRETER</b></div>", unsafe_allow_html=True)
        with col2: st.markdown("<div class='feature-card'><h3>⚖️</h3><b>STRATEGIC RAG</b></div>", unsafe_allow_html=True)
        with col3: st.markdown("<div class='feature-card'><h3>📄</h3><b>SMART DRAFTING</b></div>", unsafe_allow_html=True)

    # Display Chat History with Tags
    for msg in st.session_state.messages:
        avatar = "⚖️" if msg["role"] == "assistant" else "👩‍💼"
        with st.chat_message(msg["role"], avatar=avatar):
            # Displaying the Mode Tag
            m = msg.get("mode", "GENERAL USER (Woman)")
            tag_clr = "#FF2E7E" if "LEGAL" in m else "#888888"
            st.markdown(f"<div class='mode-tag' style='color:{tag_clr};'>[{m}]</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Enter Query..."):
        mode_at_time = st.session_state.current_mode
        # Save User Message with current mode
        st.session_state.messages.append({"role": "user", "content": prompt, "mode": mode_at_time})
        save_chat_message(st.session_state.username, "user", prompt, mode_at_time)
        st.rerun()

    # Assistant Logic
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="⚖️"):
            with st.status("Consulting Database..."):
                try:
                    mode_at_time = st.session_state.current_mode
                    query = st.session_state.messages[-1]["content"]
                    response = st.session_state.rag.invoke(query)
                    # Save Assistant Message with current mode
                    st.session_state.messages.append({"role": "assistant", "content": response, "mode": mode_at_time})
                    save_chat_message(st.session_state.username, "assistant", response, mode_at_time)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")