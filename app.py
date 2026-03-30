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
from fpdf import FPDF # PDF generation ke liye
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

# --- 3. HELPER FUNCTIONS ---
def create_pdf(text):
    """Legal draft ko PDF mein convert karne ka function"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # PDF handles latin-1 better, ignore special symbols that crash it
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

# --- 4. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #FF2E7E !important; }
    h1, h2, h3, .stSubheader { color: #FF2E7E !important; font-family: 'Inter', sans-serif; }
    .stMarkdown, p, span { color: #E0E0E0 !important; }
    .court-box { background-color: #0A0A0A; border-left: 5px solid #FF2E7E; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
    .mode-tag { font-size: 10px; font-weight: bold; color: #FF2E7E; text-transform: uppercase; margin-bottom: 5px; }
    .stChatInput input { background-color: #000 !important; color: #fff !important; border: 1px solid #333 !important; }
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
                nu, np = st.text_input("New Username"), st.text_input("New Password", type="password")
                if st.form_submit_button("Register"):
                    if add_user(nu, np): st.success("Account Created!")

# --- 6. MAIN APP ---
else:
    if "rag" not in st.session_state:
        from legal_advisor import rag_chain
        st.session_state.rag = rag_chain

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
                    st.session_state.messages.append({"role": "user", "content": f"Uploaded {len(uploaded_docs)} document(s).", "mode": st.session_state.current_mode})
                    
                    with st.chat_message("assistant", avatar="⚖️"):
                        with st.status("Analyzing Legal Text...", expanded=False) as status:
                            response = st.session_state.rag.invoke(analysis_prompt)
                            status.update(label="Analysis Complete", state="complete")
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response, "mode": st.session_state.current_mode})
                        save_chat_message(st.session_state.username, "assistant", response, st.session_state.current_mode)
                    st.rerun()

        st.divider()
        if st.button("LOGOUT"):
            st.session_state.logged_in = False
            st.rerun()

    # Chat Display
    st.header(f"⚖️ {st.session_state.current_mode}")
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else "👩‍💼"):
            st.markdown(f"<div class='mode-tag'>[{msg.get('mode', 'GENERAL')}]</div>", unsafe_allow_html=True)
            st.markdown(msg["content"])
            
            # Agar AI ne draft banaya hai toh Download button dikhayen
            if msg["role"] == "assistant" and "DRAFT" in msg["content"].upper():
                pdf_path = create_pdf(msg["content"])
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📄 Download Legal Draft (PDF)",
                        data=f,
                        file_name=f"Legal_Draft_{i}.pdf",
                        mime="application/pdf",
                        key=f"btn_{i}"
                    )

    # 1. Text Input logic
    if prompt := st.chat_input("Enter Legal Query..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "mode": st.session_state.current_mode})
        save_chat_message(st.session_state.username, "user", prompt, st.session_state.current_mode)
        
        with st.chat_message("assistant", avatar="⚖️"):
            with st.status("Consulting Precedents...", expanded=False) as status:
                response = get_adaptive_response(prompt)
                status.update(label="Consultation Finished", state="complete")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "mode": st.session_state.current_mode})
            save_chat_message(st.session_state.username, "assistant", response, st.session_state.current_mode)
        st.rerun()

    # 2. Audio Input logic
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
            st.session_state.messages.append({"role": "user", "content": f"🎤 {transcribed_text}", "mode": st.session_state.current_mode})
            save_chat_message(st.session_state.username, "user", f"🎤 {transcribed_text}", st.session_state.current_mode)
            
            with st.chat_message("assistant", avatar="⚖️"):
                with st.status("Analyzing Voice Query...", expanded=False) as status:
                    response = get_adaptive_response(transcribed_text)
                    status.update(label="Response Ready", state="complete")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "mode": st.session_state.current_mode})
                save_chat_message(st.session_state.username, "assistant", response, st.session_state.current_mode)
            st.rerun()