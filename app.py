import streamlit as st
import os
import shutil
import pytesseract
import pdfplumber 
from PIL import Image
import cv2
import numpy as np
import tempfile
import whisper
from fpdf import FPDF
from database import init_db, add_user, verify_user, save_chat_message, get_chat_history
from streamlit_mic_recorder import mic_recorder

# --- 0. FFMPEG & TESSERACT CONFIGURATION ---
def configure_paths():
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    if not shutil.which("ffmpeg"):
        winget_base = os.path.join(os.environ.get('LOCALAPPDATA', ''), r'Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.WinGet.Source_8wekyb3d8bbwe')
        found_path = None
        if os.path.exists(winget_base):
            for folder in os.listdir(winget_base):
                if folder.startswith("ffmpeg-"):
                    bin_path = os.path.join(winget_base, folder, 'bin')
                    if os.path.exists(bin_path):
                        found_path = bin_path
                        break
        possible_ffmpeg_paths = [found_path, r'C:\ffmpeg\bin', r'C:\Program Files\ffmpeg\bin', os.path.join(os.environ.get('USERPROFILE', ''), r'scoop\shims')]
        for path in possible_ffmpeg_paths:
            if path and os.path.exists(path):
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                break

configure_paths()

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Awaz-e-Nisa", page_icon="⚖️", layout="wide")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

init_db()

# --- 2. SESSION STATE ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "messages" not in st.session_state: st.session_state.messages = []
if "current_mode" not in st.session_state: st.session_state.current_mode = "GENERAL USER (Woman)"
if "last_audio_id" not in st.session_state: st.session_state.last_audio_id = None
# NEW: Track expanded analysis panels per message index
if "expanded_panels" not in st.session_state: st.session_state.expanded_panels = {}

# --- 3. HELPER FUNCTIONS ---

def create_pdf(text, filename="Legal_Document"):
    """Convert any text/draft to a downloadable PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    clean_text = text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text, align='L')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name

def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: full_text += page_text + "\n"
        import string
        readable_chars = set(string.printable)
        full_text = "".join(filter(lambda x: x in readable_chars, full_text))
        return full_text if full_text.strip() else "⚠️ PDF is empty or contains only images."
    except Exception as e: return f"Error reading PDF: {e}"

def extract_text_from_image(uploaded_file):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_img = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        custom_config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(processed_img, lang='eng', config=custom_config)
        text = "".join([c if ord(c) < 128 else "" for c in text])
        return text if text.strip() else "⚠️ OCR failed to extract meaningful text."
    except Exception as e: return f"Error reading document: {e}"


# ─────────────────────────────────────────────
# NEW HELPER: Run Analysis Chains
# ─────────────────────────────────────────────
def run_merits_analysis(query):
    return st.session_state.merits_chain.invoke(query)

def run_opposition_analysis(query):
    return st.session_state.opposition_chain.invoke(query)

def run_timeline_analysis(query):
    return st.session_state.timeline_chain.invoke(query)

def run_draft_generation(query):
    return st.session_state.draft_chain.invoke(query)


# ─────────────────────────────────────────────
# NEW UI COMPONENT: Analysis Panel (shown below each AI message)
# ─────────────────────────────────────────────
def render_analysis_panel(msg_index, original_query, mode):
    """
    Renders the 4 feature buttons below an AI response.
    Each button is a toggle — click once to generate & show, click again to hide.
    """
    panel_key = f"panel_{msg_index}"
    if panel_key not in st.session_state.expanded_panels:
        st.session_state.expanded_panels[panel_key] = {
            "merits": False,
            "opposition": False,
            "timeline": False,
            "draft": False,
            "merits_result": None,
            "opposition_result": None,
            "timeline_result": None,
            "draft_result": None,
        }
    
    panel = st.session_state.expanded_panels[panel_key]

    st.markdown("---")
    st.markdown("<p style='color:#888; font-size:12px; margin-bottom:8px;'>🔍 DEEP ANALYSIS TOOLS</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # --- Button 1: Merits & Demerits ---
    with col1:
        btn_label_m = "✅ Hide Merits" if panel["merits"] else "✅ Case Merits / Demerits"
        if st.button(btn_label_m, key=f"btn_merits_{msg_index}", use_container_width=True):
            if not panel["merits"]:
                if not panel["merits_result"]:
                    with st.spinner("Analyzing case strength..."):
                        panel["merits_result"] = run_merits_analysis(original_query)
                panel["merits"] = True
            else:
                panel["merits"] = False
            st.session_state.expanded_panels[panel_key] = panel
            st.rerun()

    # --- Button 2: Opposition Arguments ---
    with col2:
        btn_label_o = "🔴 Hide Opposition" if panel["opposition"] else "🔴 Opposition Arguments"
        if st.button(btn_label_o, key=f"btn_opp_{msg_index}", use_container_width=True):
            if not panel["opposition"]:
                if not panel["opposition_result"]:
                    with st.spinner("Generating counter-arguments..."):
                        panel["opposition_result"] = run_opposition_analysis(original_query)
                panel["opposition"] = True
            else:
                panel["opposition"] = False
            st.session_state.expanded_panels[panel_key] = panel
            st.rerun()

    # --- Button 3: Timeline Estimator ---
    with col3:
        btn_label_t = "📅 Hide Timeline" if panel["timeline"] else "📅 Timeline Estimator"
        if st.button(btn_label_t, key=f"btn_time_{msg_index}", use_container_width=True):
            if not panel["timeline"]:
                if not panel["timeline_result"]:
                    with st.spinner("Estimating court timeline..."):
                        panel["timeline_result"] = run_timeline_analysis(original_query)
                panel["timeline"] = True
            else:
                panel["timeline"] = False
            st.session_state.expanded_panels[panel_key] = panel
            st.rerun()

    # --- Button 4: Legal Draft ---
    with col4:
        btn_label_d = "📄 Hide Draft" if panel["draft"] else "📄 Generate Legal Draft"
        if st.button(btn_label_d, key=f"btn_draft_{msg_index}", use_container_width=True):
            if not panel["draft"]:
                if not panel["draft_result"]:
                    with st.spinner("Drafting legal document..."):
                        panel["draft_result"] = run_draft_generation(original_query)
                panel["draft"] = True
            else:
                panel["draft"] = False
            st.session_state.expanded_panels[panel_key] = panel
            st.rerun()

    # --- RESULTS DISPLAY ---

    if panel["merits"] and panel["merits_result"]:
        with st.container():
            st.markdown("""
                <div style='background:#0d1a0d; border:1px solid #2d5a2d; border-radius:8px; padding:16px; margin-top:10px;'>
                <p style='color:#4CAF50; font-size:13px; font-weight:bold; margin:0 0 8px 0;'>⚖️ CASE STRENGTH ANALYSIS</p>
            """, unsafe_allow_html=True)
            st.markdown(panel["merits_result"])
            st.markdown("</div>", unsafe_allow_html=True)

    if panel["opposition"] and panel["opposition_result"]:
        with st.container():
            st.markdown("""
                <div style='background:#1a0d0d; border:1px solid #5a2d2d; border-radius:8px; padding:16px; margin-top:10px;'>
                <p style='color:#FF5252; font-size:13px; font-weight:bold; margin:0 0 8px 0;'>🔴 OPPOSITION COUNTER-ARGUMENTS</p>
            """, unsafe_allow_html=True)
            st.markdown(panel["opposition_result"])
            st.markdown("</div>", unsafe_allow_html=True)

    if panel["timeline"] and panel["timeline_result"]:
        with st.container():
            st.markdown("""
                <div style='background:#0d0d1a; border:1px solid #2d2d5a; border-radius:8px; padding:16px; margin-top:10px;'>
                <p style='color:#7C9EFF; font-size:13px; font-weight:bold; margin:0 0 8px 0;'>📅 SMART TIMELINE ESTIMATE</p>
            """, unsafe_allow_html=True)
            st.markdown(panel["timeline_result"])
            st.markdown("</div>", unsafe_allow_html=True)

    if panel["draft"] and panel["draft_result"]:
        with st.container():
            st.markdown("""
                <div style='background:#0d0f1a; border:1px solid #3a3a6e; border-radius:8px; padding:16px; margin-top:10px;'>
                <p style='color:#FFD700; font-size:13px; font-weight:bold; margin:0 0 8px 0;'>📄 LEGAL DRAFT</p>
            """, unsafe_allow_html=True)
            st.markdown(panel["draft_result"])
            
            # ─── DOWNLOAD BUTTON ───
            pdf_path = create_pdf(panel["draft_result"])
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=f,
                    file_name=f"Legal_Draft_{msg_index}.pdf",
                    mime="application/pdf",
                    key=f"dl_draft_{msg_index}",
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)


# --- 4. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
    
    .stApp { background-color: #000000; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #FF2E7E !important; }
    h1, h2, h3, .stSubheader { color: #FF2E7E !important; font-family: 'Rajdhani', sans-serif !important; }
    .stMarkdown, p, span { color: #E0E0E0 !important; }
    .court-box { background-color: #0A0A0A; border-left: 5px solid #FF2E7E; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
    .mode-tag { font-size: 10px; font-weight: bold; color: #FF2E7E; text-transform: uppercase; margin-bottom: 5px; }
    .stChatInput input { background-color: #000 !important; color: #fff !important; border: 1px solid #333 !important; }
    
    /* Analysis buttons styling */
    div[data-testid="stHorizontalBlock"] .stButton button {
        background-color: #0a0a0a !important;
        border: 1px solid #333 !important;
        color: #ccc !important;
        font-size: 12px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton button:hover {
        border-color: #FF2E7E !important;
        color: #FF2E7E !important;
        background-color: #1a0010 !important;
    }
    
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIN GATE ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>AWAZ-E-NISA</h1>", unsafe_allow_html=True)
    st.divider()
    col_info, col_login = st.columns([1.2, 1], gap="large")
    with col_info:
        st.subheader("🏛️ Judicial Jurisdictions")
        st.markdown("<div class='court-box'><b>SUPREME COURT OF PAKISTAN</b></div>", unsafe_allow_html=True)
        st.markdown("<div class='court-box'><b>LAHORE HIGH COURT (LHC)</b></div>", unsafe_allow_html=True)
    with col_login:
        st.subheader("🔐 Secure Access")
        start_mode = st.selectbox("I am a:", ["GENERAL USER (Woman)", "LEGAL PRO"])
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            with st.form("l"):
                u, p = st.text_input("Username"), st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if verify_user(u, p):
                        st.session_state.logged_in, st.session_state.username = True, u
                        st.session_state.current_mode = start_mode
                        st.session_state.messages = get_chat_history(u)
                        st.rerun()
        with tab2:
            with st.form("s"):
                nu, np_ = st.text_input("New Username"), st.text_input("New Password", type="password")
                if st.form_submit_button("Register"):
                    if add_user(nu, np_): st.success("Account Created!")

# --- 6. MAIN APP ---
else:
    # Load all chains once
    if "rag" not in st.session_state:
        from legal_advisor import rag_chain, merits_chain, opposition_chain, timeline_chain, draft_chain
        st.session_state.rag = rag_chain
        st.session_state.merits_chain = merits_chain
        st.session_state.opposition_chain = opposition_chain
        st.session_state.timeline_chain = timeline_chain
        st.session_state.draft_chain = draft_chain

    def get_adaptive_response(user_input):
        helpline_instruction = ""
        if st.session_state.current_mode == "GENERAL USER (Woman)":
            helpline_instruction = "End with relevant Pakistani women legal helplines (1043, 15, etc.)."
        else:
            helpline_instruction = "Do NOT provide any helplines. Use technical legal terminology and citations (PLD/SCMR)."
        final_prompt = (
            f"Query: {user_input}\n"
            f"Rules:\n"
            f"- If query is English, respond in English.\n"
            f"- If query is Urdu/Roman Urdu, respond in Roman Urdu.\n"
            f"- Mode: {st.session_state.current_mode}.\n"
            f"- {helpline_instruction}"
        )
        return st.session_state.rag.invoke(final_prompt)

    with st.sidebar:
        st.markdown(f"<h1 style='font-size: 24px;'>AWAZ-E-NISA</h1>", unsafe_allow_html=True)
        st.write(f"👤 User: **{st.session_state.username}**")
        st.divider()
        
        st.subheader("🎤 Voice Query")
        audio_data = mic_recorder(key="audio_recorder", start_prompt="Record Audio Query", stop_prompt="Stop Recording")
        
        st.divider()
        selected_mode = st.radio("OPERATIONAL MODE:", ["GENERAL USER (Woman)", "LEGAL PRO"])
        if selected_mode != st.session_state.current_mode:
            st.session_state.current_mode = selected_mode
            st.rerun()
        
        st.divider()
        st.subheader("📷 Document Scanner")
        uploaded_docs = st.file_uploader("Upload PDF or Image", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        combined_text = ""
        if uploaded_docs:
            for doc in uploaded_docs:
                if doc.type == "application/pdf":
                    with st.spinner(f"Reading PDF: {doc.name}"): combined_text += f"\n--- {doc.name} ---\n" + extract_text_from_pdf(doc)
                else:
                    with st.spinner(f"Reading Image: {doc.name}"): combined_text += f"\n--- {doc.name} ---\n" + extract_text_from_image(doc)
            st.text_area("Extracted Content Preview:", combined_text, height=150)
            if st.button("Send to AI for Analysis"):
                if combined_text.strip():
                    analysis_prompt = f"Analyze these documents and respond in the same language as detected: \n\n {combined_text}"
                    st.session_state.messages.append({"role": "user", "content": f"Uploaded {len(uploaded_docs)} document(s).", "mode": st.session_state.current_mode, "query": analysis_prompt})
                    with st.chat_message("assistant", avatar="⚖️"):
                        with st.status("Analyzing Legal Text...", expanded=False) as status:
                            response = st.session_state.rag.invoke(analysis_prompt)
                            status.update(label="Analysis Complete", state="complete")
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "mode": st.session_state.current_mode, "query": analysis_prompt})
                        save_chat_message(st.session_state.username, "assistant", response, st.session_state.current_mode)
                    st.rerun()

        st.divider()
        if st.button("LOGOUT"):
            st.session_state.logged_in = False
            st.rerun()

    # ─── CHAT DISPLAY ───
    st.header(f"⚖️ {st.session_state.current_mode}")
    
    # Track the last user query to associate with AI response
    last_user_query = ""
    
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else "👩‍💼"):
            st.markdown(f"<div class='mode-tag'>[{msg.get('mode', 'GENERAL')}]</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])

            # Track user queries for pairing with assistant responses
            if msg["role"] == "user":
                last_user_query = msg.get("query", msg["content"])

            # ─── ANALYSIS PANEL: show below every AI response ───
            if msg["role"] == "assistant":
                # Find the preceding user query
                original_query = ""
                for j in range(i - 1, -1, -1):
                    if st.session_state.messages[j]["role"] == "user":
                        original_query = st.session_state.messages[j].get("query", st.session_state.messages[j]["content"])
                        break
                
                if original_query:
                    render_analysis_panel(i, original_query, msg.get("mode", "GENERAL"))

    # ─── TEXT INPUT ───
    if prompt := st.chat_input("Enter Legal Query..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "mode": st.session_state.current_mode, "query": prompt})
        save_chat_message(st.session_state.username, "user", prompt, st.session_state.current_mode)
        with st.chat_message("assistant", avatar="⚖️"):
            with st.status("Consulting Precedents...", expanded=False) as status:
                response = get_adaptive_response(prompt)
                status.update(label="Consultation Finished", state="complete")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "mode": st.session_state.current_mode, "query": prompt})
            save_chat_message(st.session_state.username, "assistant", response, st.session_state.current_mode)
        st.rerun()

    # ─── AUDIO INPUT ───
    if audio_data and audio_data.get('id') != st.session_state.last_audio_id:
        whisper_model = load_whisper_model()
        with st.status("🎧 Processing Audio...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data['bytes'])
                tmp_path = tmp_file.name
            result = whisper_model.transcribe(tmp_path)
            transcribed_text = result["text"].strip()
            os.remove(tmp_path)
            st.session_state.last_audio_id = audio_data['id']
            status.update(label="✅ Audio Transcribed", state="complete")
        if transcribed_text:
            st.session_state.messages.append({"role": "user", "content": f"🎤 {transcribed_text}", "mode": st.session_state.current_mode, "query": transcribed_text})
            save_chat_message(st.session_state.username, "user", f"🎤 {transcribed_text}", st.session_state.current_mode)
            with st.chat_message("assistant", avatar="⚖️"):
                with st.status("Analyzing Voice Query...", expanded=False) as status:
                    response = get_adaptive_response(transcribed_text)
                    status.update(label="Response Ready", state="complete")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "mode": st.session_state.current_mode, "query": transcribed_text})
                save_chat_message(st.session_state.username, "assistant", response, st.session_state.current_mode)
            st.rerun()
