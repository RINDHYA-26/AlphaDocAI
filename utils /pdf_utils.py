from PyPDF2 import PdfReader
import streamlit as st

def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())

def load_pdf_text(files):
    full_text = ""
    failed_files = []

    for f in files:
        try:
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

        except Exception:
            failed_files.append(getattr(f, "name", "unknown"))

    if failed_files:
        st.warning(f"⚠️ Could not extract text from: {', '.join(failed_files)}")

    return clean_text(full_text)
