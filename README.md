🤖 INTELEXI — Ask. Understand. Intelexi.

Intelexi is an intelligent, multi-mode document assistant built using Streamlit, powered by Groq LLaMA-3 models, and enhanced with semantic search, PDF understanding, and speech-to-text capabilities.

It provides highly accurate responses based on:
PDF content
Voice queries
Wikipedia fallback
AI reasoning
All within a clean, modern, responsive UI.

✨ Key Features
📄 1. Document-Based Q&A
        1.Upload one or more PDFs
        2.Intelexi extracts text using PyPDF2
        3.Splits content into chunks using semantic text splitters
        4.Embeds them using SentenceTransformers
        5.Retrieves the most relevant content for your question

🎤 2. Voice Input Processing
        1.Ask questions through audio files
        2.Faster-Whisper transcribes speech with high accuracy
        3.The transcription is passed directly to the LLaMA model

🌐 3. Wikipedia Fallback
        If a question cannot be answered from your documents, Intelexi automatically queries Wikipedia and synthesizes a helpful response.

🧠 4. AI Reasoning Fallback
        If the document and Wikipedia both fail, Intelexi uses Groq LLaMA-3.1-8B Instant for fast, intelligent responses.

💬 5. Chat History
        All interactions appear as beautifully styled chat bubbles — for both user and AI messages.

🎨 6. Custom UI Styling
Intelexi features:
        1.Gradient background
        2.Soft geometric textures
        3.Clean side navigation    
        4.Modern chat UI with readable spacing
        5.Icons, colors, and a polished visual experience
        
🛠️ Tech Stack

🛠️ Tech Stack
Component	Technology
User Interface	: Streamlit
LLM Backend	: Groq LLaMA-3.1-8B Instant
Speech-to-Text	: Whisper
PDF Parsing	: PyPDF2
Embeddings	: SentenceTransformers
Semantic Search	: NumPy cosine similarity
External Knowledge	: Wikipedia API
UI	: Custom CSS + HTML styling
![Intelexi logo](assets/intelexi_logo.png)
![Intelexi UI](assets/home_screen.png)

Architecture

                    ┌──────────────────┐
                    │      USER        │
                    │  (Text / Voice)  │
                    └───────┬──────────┘
                            │
                ┌───────────▼────────────┐
                │      Streamlit UI       │
                │ Chat, Uploads, Sidebar  │
                └───────────┬────────────┘
                            │
         ┌──────────────────▼────────────────────┐
         │          Request Handler               │
         │ (Identify: PDF? Voice? Wikipedia?)     │
         └───────────┬───────────────────────────┘
                     │
     ┌───────────────┼──────────────────────────────┐
     
     │               │                               │
     
┌────▼─────┐   ┌─────▼───────┐                ┌─────▼──────┐
│ PDF Flow │   │ Voice Flow   │                │ Wiki Flow  │
└────┬─────┘   └─────┬────────┘                └─────┬──────┘

     │               │                               │
     │       ┌───────▼───────────┐                   │
     │       │ Faster Whisper STT│                   │
     │       └───────┬───────────┘                   │
     │               │                               │
┌────▼─────────┐     │                      ┌────────▼──────────┐
│ PDF Extractor│     │                      │ Wikipedia Fetcher  │
└────┬─────────┘     │                      └────────┬──────────┘
     │               │                               │
┌────▼─────────┐     │                      ┌────────▼──────────┐
│Text Splitter │     │                      │ Summary Generator │
└────┬─────────┘     │                      └────────┬──────────┘
     │               │                               │
┌────▼───────────────▼──────┐              ┌────────▼──────────┐
│ Embed + Vector Similarity  │              │ Answer Synthesizer │
└────┬───────────────────────┘              └────────┬──────────┘
     │                                               │
     └──────────────┬────────────────────────────────┘
                    │
           ┌────────▼───────────┐
           │ Groq LLaMA-3.1 API │ (Reasoning, final response)
           └────────┬───────────┘
                    │
           ┌────────▼───────────┐
           │   Streamlit UI     │
           │   (Chat Output)    │
           └────────────────────┘
