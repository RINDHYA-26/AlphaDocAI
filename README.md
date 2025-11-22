🤖 AlphaDoc AI Chatbot
AlphaDoc is an intelligent, multi-mode document assistant built with Streamlit, powered by Groq’s LLaMA-3 models, and enhanced with semantic search, PDF understanding, and speech-to-text capabilities.
It provides accurate responses based on PDF content, voice queries, Wikipedia knowledge, and fallback AI reasoning — all inside a clean, modern UI.

✨ Key Features
📄 Document-Based Q&A
Upload one or more PDFs and ask questions from their content.
AlphaDoc extracts text using PyPDF2, splits it into semantic chunks, and retrieves the most relevant parts using embeddings from SentenceTransformers.

🎤 Voice Input Processing
Ask questions using your voice.
Faster-Whisper transcribes your audio and feeds it to the Groq model.

🌐 Wikipedia Fallback
If the answer is not found inside your documents, AlphaDoc automatically checks Wikipedia and provides a synthesized response.

🧠 AI Reasoning Fallback
If both the PDF and Wikipedia fail, AlphaDoc intelligently answers using the Groq LLaMA model’s own knowledge.

💬 Chat History
All interactions — document, Wikipedia, voice, and general queries — are displayed as beautifully styled chat bubbles.

🎨 Custom UI Styling

AlphaDoc includes a visually appealing UI:
Gradient background
Subtle geometric overlay
Modern chat bubble design
Sidebar with settings and uploads

🛠️ Tech Stack

Streamlit	 : Interactive UI framework
Groq API : 	LLaMA-3.1-8B Instant for lightning-fast responses
Faster-Whisper	: Efficient and accurate speech-to-text
PyPDF2	: PDF text extraction
SentenceTransformers	: Embedding + semantic chunk matching
Wikipedia API	: External fallback knowledge source
NumPy	: Embedding similarity scoring
![Intelexi logo](assets/intelexi_logo.png)
![Intelexi UI](assets/home_screen.png)
