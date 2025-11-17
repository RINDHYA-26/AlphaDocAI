import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import wikipedia
import numpy as np
import io
import tempfile
import os
from faster_whisper import WhisperModel
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
model_name = "llama-3.1-8b-instant"
print("Loaded client successfully!")

def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())

def load_whisper_model():
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = WhisperModel("base", device="cpu")
    return st.session_state.whisper_model

def load_pdf_text(files):
    full_text = ""

    failed_files = []

    for f in files:
        pdf = PdfReader(f)
        file_text = ""
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                file_text += txt + " "
        if file_text.strip() == "":
            failed_files.append(f.name)
        else:
            full_text += file_text

    if failed_files:
        st.warning(f"⚠️ Could not extract text from: {', '.join(failed_files)}")

    return clean_text(full_text)

def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(model, chunks):
    return model.encode(chunks, convert_to_tensor=False)

def get_top_k_chunks(query, model, chunks, chunk_embeddings, k=3):
    query_emb = model.encode([query])[0]
    scores = np.dot(chunk_embeddings, query_emb)
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]

def ask_model(client, model_name, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="AlphaDoc AI Chatbot", page_icon="🔍", layout="wide")

st.header(" 📚 🤖 AlphaDoc AI Chatbot")
st.markdown("#### 🤖 Hi Buddy! How can i assist you ?")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

st.markdown("""
<style>
.stApp {
     background-image: linear-gradient(rgba(20,30,48,0.70), rgba(25,38,72,0.70)),
                      url('https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    font-family: 'Segoe UI', sans-serif;
    color: #e6eef8;
}
.user-bubble, .bot-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 10px 0;
    max-width: 75%;
    font-size: 17px;
    line-height: 1.5;
    white-space: pre-wrap;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.user-bubble {
    background-color: rgba(40, 100, 80, 0.2);
    border: 1px solid rgba(40,160,80,0.3);
    margin-left: auto;
}
.bot-bubble {
    background-color: rgba(10, 10, 10, 0.4);
    border: 1px solid rgba(255,255,255,0.1);
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SIDEBAR
# ----------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Uploads")

    model_name = st.text_input(
        "🤖 Model Name (GROQ Models)",
        "llama-3.1-8b-instant"
    )

    uploaded_files = st.file_uploader(
        "📄 Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🗑 Clear Chat"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
whisper_model = load_whisper_model()


embedder = None
chunks = []
chunk_embeddings = np.array([])
if uploaded_files:
    st.success("📚 PDFs uploaded successfully!✔")

    st.markdown("### 📄 Uploaded Files:")
    for file in uploaded_files:
        st.markdown(f"- 📄 **{file.name}**")

    with st.spinner("📄 Reading documents..."):
        full_text = load_pdf_text(uploaded_files)

    if full_text.strip() == "":
        st.error("⚠️ Could not read any text from the PDF.")
        st.stop()

    chunks = split_text(full_text)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with st.spinner("🧠 Processing document..."):
        chunk_embeddings = embed_chunks(embedder, chunks)

# -----------------------------------------------------
# 🎤 VOICE QUESTION
# -----------------------------------------------------
st.subheader("🎤 Voice Question")

audio_file = st.audio_input("Click to record your voice question")

transcribed_text = ""

if audio_file is not None:
    audio_bytes = audio_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    segments, info = whisper_model.transcribe(temp_audio_path)
    texts = [seg.text if hasattr(seg, "text") else seg[2] for seg in segments]
    transcribed_text = " ".join(texts)

    st.success(f"Transcribed: **{transcribed_text}**")


    # If PDF uploaded -> search PDF first
    if uploaded_files and len(chunks) > 0:
        top_chunks = get_top_k_chunks(transcribed_text, embedder, chunks, chunk_embeddings)
        context = "\n\n".join(top_chunks)

        prompt = f"""
You are an AI assistant. Answer the question ONLY using the provided document context.
If the answer is not found, respond: "Information not in document."

CONTEXT:
{context}

QUESTION:
{transcribed_text}
"""
        answer = ask_model(client, model_name, prompt)

        # ⭐ UPDATED → Wikipedia fallback if doc has no answer
        if "Information not in document" in answer:
            try:
                wiki_summary = wikipedia.summary(transcribed_text, sentences=4)
                answer = f"It’s not mentioned in the document, but here’s what I found:\n\n{wiki_summary}" #added ##$
            except:
               #wiki_summary = "No Wikipedia results found."
               model_explanation = ask_model(
                   client,
                   model_name,
                   f"Explain this topic simply: {transcribed_text}"
               )
               answer = (
                "It’s not mentioned in the document, "
                "and no Wikipedia results were found.\n\n"
                "Here’s a simplified explanation instead:\n\n"
                f"{model_explanation}"
            )  # ⭐ UPDATED

    else:
        # Wikipedia fallback
        try:
            wiki_summary = wikipedia.summary(transcribed_text, sentences=4)
        except:
            wiki_summary = "No Wikipedia results found."
        prompt = f"""
Use ONLY this Wikipedia information to answer:

{wiki_summary}

Question: {transcribed_text}
"""
        answer = ask_model(client, model_name, prompt)

    st.session_state["chat"].append(("user", transcribed_text))
    st.session_state["chat"].append(("assistant", answer))
# ----------------------------------
# DOCUMENT MODE (TEXT INPUT)
# ----------------------------------
if uploaded_files:
    st.markdown("### 💬 Ask your question")
    query = st.text_input("Type your question here:")


    if query:
        st.session_state["doc_query"] = "" #updated line248
        top_chunks = get_top_k_chunks(query, embedder, chunks, chunk_embeddings)
        context = "\n\n".join(top_chunks)

        prompt = f"""
You are an AI assistant. Answer the question ONLY using the provided document context.
If the answer is not found, respond: "Information not in document."

CONTEXT:
{context}

QUESTION:
{query}
"""

        answer = ask_model(client, model_name, prompt)
        if "Information not in document" in answer: #updated
            try:
                wiki_summary = wikipedia.summary(query, sentences=4)
            except:
                wiki_summary = "No Wikipedia results found."

            answer = f"It’s not mentioned in the document, but here’s the Wikipedia answer:\n\n{wiki_summary}"
        st.session_state["chat"].append(("user", query))
        st.session_state["chat"].append(("assistant", answer))
    with st.expander("✨ Suggested Prompts"): #updated
        st.markdown("""
            - **Summarize the document in 5 bullet points**
            - **What are the key topics covered?**
            - **Explain the concept in simple words**
            - **List definitions from the document**
            - **What does the document say about <topic>?**
            """)

# ----------------------------------
# WIKIPEDIA MODE
# ----------------------------------
# ----------------------------------
# WIKIPEDIA + GENERAL KNOWLEDGE MODE
# ----------------------------------
else:
    query = st.text_input("Ask anything:")
    if query:
        st.session_state["general_query"] = ""

        try:
            wiki_summary = wikipedia.summary(query, sentences=4)
            prompt = f"""
        You are an AI assistant. Use the following Wikipedia information to answer the question factually and directly.


{wiki_summary}

Question: {query}
"""
            answer = ask_model(client, model_name, prompt)

        except:
            # Wikipedia failed → fallback to general explanation
            prompt = f"""
            You are an AI assistant. Provide a direct and accurate answer to this general knowledge question.


            {query}
            """
            answer = ask_model(client, model_name, prompt)
        st.session_state["chat"].append(("user", query))
        st.session_state["chat"].append(("assistant", answer))

# ----------------------------------
# CHAT DISPLAY
# ----------------------------------
st.markdown("---")
st.markdown("### 💬 Chat History")

for role, message in st.session_state["chat"]:
    if role == "user":
        st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'><b>🤖 AI:</b><br>{message}</div>", unsafe_allow_html=True)